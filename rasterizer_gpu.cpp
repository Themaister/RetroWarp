#include <context.hpp>
#include "rasterizer_gpu.hpp"
#include "context.hpp"
#include "device.hpp"
#include <stdexcept>
#include "math.hpp"
#include "stb_image_write.h"

using namespace Granite;
using namespace Vulkan;

namespace RetroWarp
{
struct BBox
{
	int min_x, max_x, min_y, max_y;
};

struct RasterizerGPU::Impl
{
	Context context;
	Device device;
	ImageHandle image;
	BufferHandle color_buffer;
	BufferHandle depth_buffer;
	unsigned width = 0;
	unsigned height = 0;

	struct
	{
		// First pass, bin at 64x64.
		BufferHandle mask_buffer_low_res;

		// Bin at 16x16.
		BufferHandle mask_buffer;

		// Groups group of 32 primitives into one 1 bit for faster rejection in raster.
		BufferHandle mask_buffer_coarse;
	} binning;

	struct
	{
		// Prefix sum of primitives active inside a tile.
		BufferHandle tile_prefix_sum;
		// Total primitives for a tile.
		BufferHandle tile_total;
		// Horizontal prefix sum of tile_total.
		BufferHandle horiz_prefix_sum;
		// Totals for horizontal scan.
		BufferHandle horiz_total;
		// Vertical scan of horiz_total.
		BufferHandle vert_prefix_sum;
		// Final resolved tile offsets.
		BufferHandle tile_offset;
	} tile_count;

	struct
	{
		BufferHandle color;
		BufferHandle depth;
		BufferHandle flags;
	} tile_instance_data;

	struct
	{
		BufferHandle positions;
		BufferHandle attributes;
		PrimitiveSetupPos *mapped_positions = nullptr;
		PrimitiveSetupAttr *mapped_attributes = nullptr;
		unsigned count = 0;
		unsigned num_conservative_tile_instances = 0;
	} staging;

	struct
	{
		BufferHandle item_count_per_variant;
		BufferHandle work_list_per_variant;
	} raster_work;

	struct
	{
		int x, y, width, height;
	} scissor;

	void reset_staging();
	void begin_staging();
	void end_staging();

	void init_binning_buffers();
	void init_prefix_sum_buffers();
	void init_tile_buffers();
	void init_raster_work_buffers();
	void flush();

	void queue_primitive(const PrimitiveSetup &setup);
	unsigned compute_num_conservative_tiles(const PrimitiveSetup &setup) const;

	BBox compute_bbox(const PrimitiveSetup &setup) const;
	bool clip_bbox_scissor(BBox &clipped_bbox, const BBox &bbox) const;

	void set_fb_info(CommandBuffer &cmd);
	void clear_indirect_buffer(CommandBuffer &cmd);
	void binning_low_res_prepass(CommandBuffer &cmd);
	void binning_full_res(CommandBuffer &cmd);
	void build_coarse_mask(CommandBuffer &cmd);
	void run_per_tile_prefix_sum(CommandBuffer &cmd);
	void run_horiz_prefix_sum(CommandBuffer &cmd);
	void run_vert_prefix_sum(CommandBuffer &cmd);
	void finalize_tile_offsets(CommandBuffer &cmd);
	void distribute_combiner_work(CommandBuffer &cmd);
	void run_rop(CommandBuffer &cmd);

	void test_prefix_sum();
};

struct FBInfo
{
	int32_t scissor_x, scissor_y, scissor_width, scissor_height;
	uvec2 resolution;
	uvec2 resolution_tiles;
	uint32_t fb_stride;
	uint32_t primitive_count;
	uint32_t primitive_count_32;
	uint32_t primitive_count_1024;
};

constexpr int MAX_PRIMITIVES = 0x4000;
constexpr int TILE_BINNING_STRIDE = MAX_PRIMITIVES / 32;
constexpr int TILE_BINNING_STRIDE_COARSE = TILE_BINNING_STRIDE / 32;
constexpr int MAX_WIDTH = 2048;
constexpr int MAX_HEIGHT = 2048;
constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;
constexpr int TILE_WIDTH_LOG2 = 4;
constexpr int TILE_HEIGHT_LOG2 = 4;
constexpr int MAX_TILES_X = MAX_WIDTH / TILE_WIDTH;
constexpr int MAX_TILES_Y = MAX_HEIGHT / TILE_HEIGHT;
constexpr int MAX_TILES_X_LOW_RES = MAX_WIDTH / (4 * TILE_WIDTH);
constexpr int MAX_TILES_Y_LOW_RES = MAX_HEIGHT / (4 * TILE_HEIGHT);
constexpr int MAX_NUM_TILE_INSTANCES = 0xffff;
const int RASTER_ROUNDING = (1 << (SUBPIXELS_LOG2 + 16)) - 1;

struct PerTileData
{
	uint32_t color;
	uint16_t depth;
	uint16_t flags;
};

struct TileRasterWork
{
	uint16_t tile_x, tile_y;
	uint16_t tile_instance;
	uint16_t primitive;
};

void RasterizerGPU::Impl::reset_staging()
{
	staging.positions.reset();
	staging.attributes.reset();
	staging.mapped_attributes = nullptr;
	staging.mapped_positions = nullptr;
	staging.count = 0;
	staging.num_conservative_tile_instances = 0;
}

BBox RasterizerGPU::Impl::compute_bbox(const PrimitiveSetup &setup) const
{
	int lo_x = std::numeric_limits<int>::max();
	int hi_x = std::numeric_limits<int>::min();
	int lo_y = std::numeric_limits<int>::max();
	int hi_y = std::numeric_limits<int>::min();

	lo_x = std::min(lo_x, setup.pos.x_a);
	lo_x = std::min(lo_x, setup.pos.x_b);
	lo_x = std::min(lo_x, setup.pos.x_c);
	hi_x = std::max(hi_x, setup.pos.x_a);
	hi_x = std::max(hi_x, setup.pos.x_b);
	hi_x = std::max(hi_x, setup.pos.x_c);

	int end_point_a = setup.pos.x_a + setup.pos.dxdy_a * (setup.pos.y_hi - setup.pos.y_lo);
	int end_point_b = setup.pos.x_b + setup.pos.dxdy_b * (setup.pos.y_mid - setup.pos.y_lo);
	int end_point_c = setup.pos.x_c + setup.pos.dxdy_c * (setup.pos.y_hi - setup.pos.y_mid);

	lo_x = std::min(lo_x, end_point_a);
	lo_x = std::min(lo_x, end_point_b);
	lo_x = std::min(lo_x, end_point_c);
	hi_x = std::max(hi_x, end_point_a);
	hi_x = std::max(hi_x, end_point_b);
	hi_x = std::max(hi_x, end_point_c);

	BBox bbox = {};
	bbox.min_x = (lo_x + RASTER_ROUNDING) >> (16 + SUBPIXELS_LOG2);
	bbox.max_x = (hi_x - 1) >> (16 + SUBPIXELS_LOG2);
	bbox.min_y = (setup.pos.y_lo + (1 << SUBPIXELS_LOG2) - 1) >> SUBPIXELS_LOG2;
	bbox.max_y = (setup.pos.y_hi - 1) >> SUBPIXELS_LOG2;
	return bbox;
}

bool RasterizerGPU::Impl::clip_bbox_scissor(BBox &clipped_bbox, const BBox &bbox) const
{
	clipped_bbox.min_x = std::max(scissor.x, bbox.min_x);
	clipped_bbox.max_x = std::min(scissor.x + scissor.width - 1, bbox.max_x);
	clipped_bbox.min_y = std::max(scissor.y, bbox.min_y);
	clipped_bbox.max_y = std::min(scissor.y + scissor.height - 1, bbox.max_y);

	return clipped_bbox.min_x <= clipped_bbox.max_x && clipped_bbox.min_y <= clipped_bbox.max_y;
}

unsigned RasterizerGPU::Impl::compute_num_conservative_tiles(const PrimitiveSetup &setup) const
{
	auto bbox = compute_bbox(setup);
	BBox clipped_bbox;
	if (!clip_bbox_scissor(clipped_bbox, bbox))
		return 0;

	int start_tile_x = clipped_bbox.min_x >> TILE_WIDTH_LOG2;
	int end_tile_x = clipped_bbox.max_x >> TILE_WIDTH_LOG2;
	int start_tile_y = clipped_bbox.min_y >> TILE_HEIGHT_LOG2;
	int end_tile_y = clipped_bbox.max_y >> TILE_HEIGHT_LOG2;
	return (end_tile_x - start_tile_x + 1) * (end_tile_y - start_tile_y + 1);
}

void RasterizerGPU::Impl::begin_staging()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Host;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupPos);
	staging.positions = device.create_buffer(info);
	info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupAttr);
	staging.attributes = device.create_buffer(info);

	staging.mapped_positions = static_cast<PrimitiveSetupPos *>(
			device.map_host_buffer(*staging.positions,
			                       MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_attributes = static_cast<PrimitiveSetupAttr *>(
			device.map_host_buffer(*staging.attributes,
			                       MEMORY_ACCESS_WRITE_BIT));

	staging.count = 0;
	staging.num_conservative_tile_instances = 0;
}

void RasterizerGPU::Impl::end_staging()
{
	if (staging.mapped_positions)
		device.unmap_host_buffer(*staging.positions, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_attributes)
		device.unmap_host_buffer(*staging.attributes, MEMORY_ACCESS_WRITE_BIT);

	staging.mapped_positions = nullptr;
	staging.mapped_attributes = nullptr;
}

void RasterizerGPU::Impl::clear_indirect_buffer(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/clear_indirect_buffers.comp");
	cmd.set_storage_buffer(0, 0, *raster_work.item_count_per_variant);
	cmd.dispatch(1, 1, 1);
}

void RasterizerGPU::Impl::binning_low_res_prepass(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/binning_low_res.comp");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer_low_res);
	cmd.set_storage_buffer(0, 1, *staging.positions);
	cmd.dispatch((staging.count + 63) / 64,
	             (width + 4 * TILE_WIDTH - 1) / (4 * TILE_WIDTH),
	             (height + 4 * TILE_HEIGHT - 1) / (4 * TILE_HEIGHT));

}

void RasterizerGPU::Impl::binning_full_res(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/binning.comp");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer);
	cmd.set_storage_buffer(0, 1, *staging.positions);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer_low_res);
	cmd.dispatch((staging.count + 63) / 64,
	             (width + TILE_WIDTH - 1) / TILE_WIDTH,
	             (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
}

void RasterizerGPU::Impl::build_coarse_mask(CommandBuffer &cmd)
{
	uint32_t num_masks = (staging.count + 31) / 32;
	cmd.set_program("assets://shaders/build_coarse_mask.comp");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer);
	cmd.set_storage_buffer(0, 1, *binning.mask_buffer_coarse);
	cmd.push_constants(&num_masks, 0, sizeof(num_masks));
	cmd.dispatch((num_masks + 63) / 64,
	             (width + TILE_WIDTH - 1) / TILE_WIDTH,
	             (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
}

void RasterizerGPU::Impl::run_per_tile_prefix_sum(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/tile_prefix_sum.comp");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer);
	cmd.set_storage_buffer(0, 1, *tile_count.tile_prefix_sum);
	cmd.set_storage_buffer(0, 2, *tile_count.tile_total);

	unsigned tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
	unsigned tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	cmd.dispatch(tiles_x, tiles_y, 1);
}

void RasterizerGPU::Impl::run_horiz_prefix_sum(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/horiz_prefix_sum.comp");
	cmd.set_storage_buffer(0, 0, *tile_count.tile_total);
	cmd.set_storage_buffer(0, 1, *tile_count.horiz_prefix_sum);
	cmd.set_storage_buffer(0, 2, *tile_count.horiz_total);
	unsigned tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	cmd.dispatch(1, tiles_y, 1);
}

void RasterizerGPU::Impl::run_vert_prefix_sum(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/vert_prefix_sum.comp");
	cmd.set_storage_buffer(0, 0, *tile_count.horiz_total);
	cmd.set_storage_buffer(0, 1, *tile_count.vert_prefix_sum);
	cmd.dispatch(1, 1, 1);
}

void RasterizerGPU::Impl::finalize_tile_offsets(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/finalize_tile_offsets.comp");
	cmd.set_storage_buffer(0, 0, *tile_count.horiz_prefix_sum);
	cmd.set_storage_buffer(0, 1, *tile_count.vert_prefix_sum);
	cmd.set_storage_buffer(0, 2, *tile_count.tile_offset);
	cmd.dispatch(MAX_TILES_X / 8, MAX_TILES_Y / 8, 1);
}

void RasterizerGPU::Impl::distribute_combiner_work(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/distribute_combiner_work.comp");
	cmd.set_storage_buffer(0, 0, *tile_count.tile_offset);
	cmd.set_storage_buffer(0, 1, *tile_count.tile_prefix_sum);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer);
	cmd.set_storage_buffer(0, 3, *raster_work.item_count_per_variant);
	cmd.set_storage_buffer(0, 4, *raster_work.work_list_per_variant);

	unsigned num_tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
	unsigned num_tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	cmd.dispatch(num_tiles_x, num_tiles_y, 1);
}

void RasterizerGPU::Impl::set_fb_info(CommandBuffer &cmd)
{
	auto *fb_info = cmd.allocate_typed_constant_data<FBInfo>(2, 0, 1);
	fb_info->scissor_x = 0;
	fb_info->scissor_y = 0;
	fb_info->scissor_width = width;
	fb_info->scissor_height = height;
	fb_info->resolution.x = width;
	fb_info->resolution.y = height;
	fb_info->resolution_tiles.x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
	fb_info->resolution_tiles.y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	fb_info->fb_stride = width;
	fb_info->primitive_count = staging.count;
	uint32_t num_masks = (staging.count + 31) / 32;
	fb_info->primitive_count_32 = num_masks;
	fb_info->primitive_count_1024 = (staging.count + 1023) / 1024;
}

void RasterizerGPU::Impl::run_rop(CommandBuffer &cmd)
{
	cmd.set_program("assets://shaders/rasterize.comp");
	cmd.set_storage_buffer(0, 0, *staging.positions);
	cmd.set_storage_buffer(0, 1, *staging.attributes);
	cmd.set_storage_buffer(0, 2, *color_buffer);
	cmd.set_storage_buffer(0, 3, *depth_buffer);
	cmd.set_storage_buffer(0, 4, *binning.mask_buffer);
	cmd.set_storage_buffer(0, 5, *binning.mask_buffer_coarse);
	cmd.set_texture(1, 0, image->get_view());
	cmd.dispatch((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1);
}

void RasterizerGPU::Impl::flush()
{
	end_staging();
	if (staging.count == 0)
		return;

	device.next_frame_context();
	auto cmd = device.request_command_buffer();

	set_fb_info(*cmd);

	// This part can overlap with previous flush.
	// Clear indirect buffer.
	clear_indirect_buffer(*cmd);
	// Binning low-res prepass.
	binning_low_res_prepass(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Binning at full-resolution.
	binning_full_res(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Merge coarse mask.
	// Run per-tile prefix sum.
	run_per_tile_prefix_sum(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Run horizontal prefix sum.
	run_horiz_prefix_sum(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Run vertical prefix sum.
	// This job is very small, so run coarse mask building in parallel to avoid starving GPU completely.
	run_vert_prefix_sum(*cmd);
	build_coarse_mask(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Finalize offsets per tile.
	finalize_tile_offsets(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Distribute work.
	distribute_combiner_work(*cmd);

	// ROP.
	run_rop(*cmd);

	device.submit(cmd);
	reset_staging();
}

void RasterizerGPU::Impl::init_binning_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint32_t);
	binning.mask_buffer = device.create_buffer(info);

	info.size = MAX_TILES_X_LOW_RES * MAX_TILES_Y_LOW_RES * TILE_BINNING_STRIDE * sizeof(uint32_t);
	binning.mask_buffer_low_res = device.create_buffer(info);

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE_COARSE * sizeof(uint32_t);
	binning.mask_buffer_coarse = device.create_buffer(info);
}

void RasterizerGPU::Impl::init_prefix_sum_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint16_t);
	tile_count.tile_prefix_sum = device.create_buffer(info);

	info.size = MAX_TILES_X * MAX_TILES_Y * sizeof(uint16_t);
	tile_count.tile_total = device.create_buffer(info);
	tile_count.horiz_prefix_sum = device.create_buffer(info);
	tile_count.tile_offset = device.create_buffer(info);

	info.size = MAX_TILES_Y * sizeof(uint16_t);
	tile_count.horiz_total = device.create_buffer(info);
	tile_count.vert_prefix_sum = device.create_buffer(info);
}

void RasterizerGPU::Impl::init_tile_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint32_t);
	tile_instance_data.color = device.create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint16_t);
	tile_instance_data.depth = device.create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint16_t);
	tile_instance_data.flags = device.create_buffer(info);
}

void RasterizerGPU::Impl::init_raster_work_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	// Round MAX_NUM_TILE_INSTANCES up to 0x10000.
	info.size = (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork) * 16;
	raster_work.work_list_per_variant = device.create_buffer(info);

	info.size = 16 * (4 * sizeof(uint32_t));
	raster_work.item_count_per_variant = device.create_buffer(info);
}

template <typename T>
static std::vector<T> readback_buffer(Device &device, const Buffer &buffer)
{
	std::vector<T> result(buffer.get_create_info().size / sizeof(T));

	BufferCreateInfo info;
	info.domain = BufferDomain::CachedHost;
	info.size = buffer.get_create_info().size;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	auto readback_buffer = device.create_buffer(info);

	auto cmd = device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
	             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	cmd->copy_buffer(*readback_buffer, buffer);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	const void *mapped = device.map_host_buffer(*readback_buffer, MEMORY_ACCESS_READ_BIT);
	memcpy(result.data(), mapped, result.size() * sizeof(T));
	device.unmap_host_buffer(*readback_buffer, MEMORY_ACCESS_READ_BIT);

	return result;
}

void RasterizerGPU::Impl::test_prefix_sum()
{
	auto cmd = device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
	             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT);

	width = 1024;
	height = 1024;
	staging.count = 100 * 32;

	set_fb_info(*cmd);
	clear_indirect_buffer(*cmd);

	auto *ptr = static_cast<uint32_t *>(
			cmd->update_buffer(*binning.mask_buffer,
			                   0,
			                   MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint32_t)));

	memset(ptr, 0, MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint32_t));
	for (int y = 0; y < MAX_TILES_Y; y++)
	{
		for (int x = 0; x < MAX_TILES_X; x++)
		{
			auto *base = &ptr[(y * MAX_TILES_X + x) * TILE_BINNING_STRIDE];
			base[0] = 0x1000;
			base[70] = 0x80000;
		}
	}

	cmd->fill_buffer(*tile_count.tile_prefix_sum, -1u);
	cmd->fill_buffer(*tile_count.tile_total, -1u);
	cmd->fill_buffer(*tile_count.horiz_total, -1u);
	cmd->fill_buffer(*tile_count.horiz_prefix_sum, -1u);
	cmd->fill_buffer(*tile_count.vert_prefix_sum, -1u);
	cmd->fill_buffer(*tile_count.tile_offset, -1u);

	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Prefix sum.
	run_per_tile_prefix_sum(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Run horizontal prefix sum.
	run_horiz_prefix_sum(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Run vertical prefix sum.
	// This job is very small, so run coarse mask building in parallel to avoid starving GPU completely.
	run_vert_prefix_sum(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Finalize offsets per tile.
	finalize_tile_offsets(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	device.submit(cmd);

	auto tile_prefix_sum = readback_buffer<uint16_t>(device, *tile_count.tile_prefix_sum);
	auto tile_totals = readback_buffer<uint16_t>(device, *tile_count.tile_total);
	auto horiz_prefix_sum = readback_buffer<uint16_t>(device, *tile_count.horiz_prefix_sum);
	auto horiz_total = readback_buffer<uint16_t>(device, *tile_count.horiz_total);
	auto vert_prefix_sum = readback_buffer<uint16_t>(device, *tile_count.vert_prefix_sum);
	auto tile_offset = readback_buffer<uint16_t>(device, *tile_count.tile_offset);

	reset_staging();
}

RasterizerGPU::RasterizerGPU()
{
	impl.reset(new Impl);
	if (!Context::init_loader(nullptr))
		throw std::runtime_error("Failed to init context.");
	if (!impl->context.init_instance_and_device(nullptr, 0, nullptr, 0))
		throw std::runtime_error("Failed to init instance and device.");

	impl->device.set_context(impl->context);

	auto &features = impl->device.get_device_features();
	if (!features.storage_8bit_features.storageBuffer8BitAccess)
		throw std::runtime_error("8-bit storage not supported.");
	if (!features.storage_16bit_features.storageBuffer16BitAccess)
		throw std::runtime_error("16-bit storage not supported.");

	impl->init_binning_buffers();
	impl->init_prefix_sum_buffers();
	impl->init_tile_buffers();
	impl->init_raster_work_buffers();

	impl->test_prefix_sum();
}

RasterizerGPU::~RasterizerGPU()
{
}

void RasterizerGPU::upload_texture(const TextureFormatLayout &layout)
{
	auto staging = impl->device.create_image_staging_buffer(layout);
	auto info = ImageCreateInfo::immutable_2d_image(layout.get_width(), layout.get_height(), VK_FORMAT_R8G8B8A8_UINT);
	impl->image = impl->device.create_image_from_staging_buffer(info, &staging);
}

void RasterizerGPU::resize(unsigned width, unsigned height)
{
	impl->width = width;
	impl->height = height;

	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.size = width * height * sizeof(uint32_t);
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	impl->color_buffer = impl->device.create_buffer(info);

	info.size = (width * height * sizeof(uint16_t) + 3) & ~3u;
	impl->depth_buffer = impl->device.create_buffer(info);
}

void RasterizerGPU::clear_depth(uint16_t z)
{
	impl->flush();
	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->fill_buffer(*impl->depth_buffer, (uint32_t(z) << 16) | z);
	impl->device.submit(cmd);
	impl->device.next_frame_context();
}

void RasterizerGPU::clear_color(uint32_t rgba)
{
	impl->flush();
	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->fill_buffer(*impl->color_buffer, rgba);
	impl->device.submit(cmd);
	impl->device.next_frame_context();
}

void RasterizerGPU::Impl::queue_primitive(const PrimitiveSetup &setup)
{
	unsigned num_conservative_tiles = compute_num_conservative_tiles(setup);

	if (staging.count == MAX_PRIMITIVES)
		flush();
	else if (staging.num_conservative_tile_instances + num_conservative_tiles > MAX_NUM_TILE_INSTANCES)
		flush();

	if (staging.count == 0)
		begin_staging();

	staging.mapped_positions[staging.count] = setup.pos;
	staging.mapped_attributes[staging.count] = setup.attr;

	staging.count++;
	staging.num_conservative_tile_instances += num_conservative_tiles;
}

void RasterizerGPU::rasterize_primitives(const RetroWarp::PrimitiveSetup *setup, size_t count)
{
	for (size_t i = 0; i < count; i++)
		impl->queue_primitive(setup[i]);
}

float RasterizerGPU::get_binning_ratio(size_t count)
{
	impl->flush();
	impl->device.next_frame_context();

	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_READ_BIT);

	BufferCreateInfo info;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.domain = BufferDomain::CachedHost;
	info.size = impl->binning.mask_buffer->get_create_info().size;
	auto dst_buffer = impl->device.create_buffer(info);
	info.size = impl->binning.mask_buffer_coarse->get_create_info().size;
	auto dst_buffer_coarse = impl->device.create_buffer(info);
	cmd->copy_buffer(*dst_buffer, *impl->binning.mask_buffer);
	cmd->copy_buffer(*dst_buffer_coarse, *impl->binning.mask_buffer_coarse);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	Fence fence;
	impl->device.submit(cmd, &fence);

	fence->wait();
	auto *ptr = static_cast<uint32_t *>(impl->device.map_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT));
	auto *ptr_coarse = static_cast<uint32_t *>(impl->device.map_host_buffer(*dst_buffer_coarse, MEMORY_ACCESS_READ_BIT));

	unsigned num_tiles_x = (impl->width + TILE_WIDTH - 1) / TILE_WIDTH;
	unsigned num_tiles_y = (impl->height + TILE_HEIGHT - 1) / TILE_HEIGHT;

	unsigned max_count = 0;
	unsigned total_count = 0;
	for (unsigned y = 0; y < num_tiles_y; y++)
	{
		for (unsigned x = 0; x < num_tiles_x; x++)
		{
			auto *p = &ptr[TILE_BINNING_STRIDE * (y * MAX_TILES_X + x)];
			auto *p_coarse = &ptr_coarse[TILE_BINNING_STRIDE_COARSE * (y * MAX_TILES_X + x)];
			for (size_t i = 0; i < (count + 31) / 32; i++)
			{
				uint32_t mask = p[i];

				if (mask != 0)
					assert((p_coarse[i >> 5] & (1u << (i & 31))) != 0);
				else
					assert((p_coarse[i >> 5] & (1u << (i & 31))) == 0);

				//unsigned mask_count = __builtin_popcount(mask);
				unsigned mask_count = 0;
				total_count += mask_count;
			}
			max_count += count;
		}
	}

	impl->device.unmap_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT);
	impl->device.unmap_host_buffer(*dst_buffer_coarse, MEMORY_ACCESS_READ_BIT);
	return float(total_count) / float(max_count);
}

bool RasterizerGPU::save_canvas(const char *path)
{
	impl->flush();
	impl->device.next_frame_context();

	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_READ_BIT);

	BufferCreateInfo info;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.domain = BufferDomain::CachedHost;
	info.size = impl->width * impl->height * sizeof(uint32_t);
	auto dst_buffer = impl->device.create_buffer(info);
	cmd->copy_buffer(*dst_buffer, *impl->color_buffer);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	Fence fence;
	impl->device.submit(cmd, &fence);
	impl->device.next_frame_context();

	fence->wait();
	auto *ptr = static_cast<uint32_t *>(impl->device.map_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT | MEMORY_ACCESS_WRITE_BIT));
	for (unsigned i = 0; i < impl->width * impl->height; i++)
		ptr[i] |= 0xff000000u;
	bool res = stbi_write_png(path, impl->width, impl->height, 4, ptr, impl->width * 4);
	impl->device.unmap_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT | MEMORY_ACCESS_WRITE_BIT);
	return res;
}

}