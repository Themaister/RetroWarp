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

	BufferHandle tile_instance_data;

	struct
	{
		BufferHandle positions;
		BufferHandle attributes;
		PrimitiveSetupPos *mapped_positions = nullptr;
		PrimitiveSetupAttr *mapped_attributes = nullptr;
		unsigned count = 0;
		unsigned num_conservative_tile_instances = 0;
	} staging;

	void reset_staging();
	void begin_staging();
	void end_staging();

	void init_binning_buffers();
	void init_prefix_sum_buffers();
	void init_tile_buffers();
	void flush();

	void queue_primitive(const PrimitiveSetup &setup);
	unsigned compute_num_conservative_tiles(const PrimitiveSetup &setup) const;
};

struct Registers
{
	uvec2 resolution;
	uint32_t primitive_count;
	uint32_t fb_stride;
	int32_t scissor_x, scissor_y, scissor_width, scissor_height;
};

constexpr unsigned MAX_PRIMITIVES = 0x4000;
constexpr unsigned TILE_BINNING_STRIDE = MAX_PRIMITIVES / 32;
constexpr unsigned TILE_BINNING_STRIDE_COARSE = TILE_BINNING_STRIDE / 32;
constexpr unsigned MAX_WIDTH = 2048;
constexpr unsigned MAX_HEIGHT = 2048;
constexpr unsigned TILE_WIDTH = 16;
constexpr unsigned TILE_HEIGHT = 16;
constexpr unsigned MAX_TILES_X = MAX_WIDTH / TILE_WIDTH;
constexpr unsigned MAX_TILES_Y = MAX_HEIGHT / TILE_HEIGHT;
constexpr unsigned MAX_TILES_X_LOW_RES = MAX_WIDTH / (4 * TILE_WIDTH);
constexpr unsigned MAX_TILES_Y_LOW_RES = MAX_HEIGHT / (4 * TILE_HEIGHT);
constexpr unsigned MAX_NUM_TILE_INSTANCES = 0x10000;

struct PerTileData
{
	uint32_t color;
	uint16_t depth;
	uint16_t flags;
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

unsigned RasterizerGPU::Impl::compute_num_conservative_tiles(const PrimitiveSetup &setup) const
{
	return 0;
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

void RasterizerGPU::Impl::flush()
{
	end_staging();
	if (staging.count == 0)
		return;

	Registers reg = {};
	reg.fb_stride = width;
	reg.primitive_count = staging.count;
	reg.resolution.x = width;
	reg.resolution.y = height;
	reg.scissor_x = 0;
	reg.scissor_y = 0;
	reg.scissor_width = width;
	reg.scissor_height = height;

	device.next_frame_context();
	auto cmd = device.request_command_buffer();

	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Binning low-res prepass
	cmd->set_program("assets://shaders/binning_low_res.comp");
	cmd->set_storage_buffer(0, 0, *binning.mask_buffer_low_res);
	cmd->set_storage_buffer(0, 1, *staging.positions);
	cmd->push_constants(&reg, 0, sizeof(reg));
	cmd->dispatch((staging.count + 63) / 64,
	              (width + 4 * TILE_WIDTH - 1) / (4 * TILE_WIDTH),
	              (height + 4 * TILE_HEIGHT - 1) / (4 * TILE_HEIGHT));

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Binning
	cmd->set_program("assets://shaders/binning.comp");
	cmd->set_storage_buffer(0, 0, *binning.mask_buffer);
	cmd->set_storage_buffer(0, 1, *staging.positions);
	cmd->set_storage_buffer(0, 2, *binning.mask_buffer_low_res);
	cmd->push_constants(&reg, 0, sizeof(reg));
	cmd->dispatch((staging.count + 63) / 64,
	              (width + TILE_WIDTH - 1) / TILE_WIDTH,
	              (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	auto t2 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Merge coarse mask
	cmd->set_program("assets://shaders/build_coarse_mask.comp");
	cmd->set_storage_buffer(0, 0, *binning.mask_buffer);
	cmd->set_storage_buffer(0, 1, *binning.mask_buffer_coarse);
	uint32_t num_masks = (staging.count + 31) / 32;
	cmd->push_constants(&num_masks, 0, sizeof(num_masks));
	cmd->dispatch((num_masks + 63) / 64,
	              (width + TILE_WIDTH - 1) / TILE_WIDTH,
	              (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	auto t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Rasterization
	cmd->set_program("assets://shaders/rasterize.comp");
	cmd->set_storage_buffer(0, 0, *staging.positions);
	cmd->set_storage_buffer(0, 1, *staging.attributes);
	cmd->set_storage_buffer(0, 2, *color_buffer);
	cmd->set_storage_buffer(0, 3, *depth_buffer);
	cmd->set_storage_buffer(0, 4, *binning.mask_buffer);
	cmd->set_storage_buffer(0, 5, *binning.mask_buffer_coarse);
	cmd->set_texture(1, 0, image->get_view());

	cmd->push_constants(&reg, 0, sizeof(reg));
	cmd->dispatch((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1);

	auto t4 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	device.submit(cmd);
	reset_staging();

	device.wait_idle();

	if (t0->is_signalled() && t1->is_signalled() && t2->is_signalled() && t3->is_signalled() && t4->is_signalled())
	{
		double time0 = t0->get_timestamp();
		double time1 = t1->get_timestamp();
		double time2 = t2->get_timestamp();
		double time3 = t3->get_timestamp();
		double time4 = t4->get_timestamp();
		LOGI("Pre-binning time: %.6f ms\n", 1000.0 * (time1 - time0));
		LOGI("Binning time: %.6f ms\n", 1000.0 * (time2 - time1));
		LOGI("Merge coarse mask time: %.6f ms\n", 1000.0 * (time3 - time2));
		LOGI("Rasterize time: %.6f ms\n", 1000.0 * (time4 - time3));
	}
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

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint32_t);
	tile_count.tile_prefix_sum = device.create_buffer(info);

	info.size = MAX_TILES_X * MAX_TILES_Y * sizeof(uint32_t);
	tile_count.tile_total = device.create_buffer(info);
	tile_count.horiz_prefix_sum = device.create_buffer(info);
	tile_count.tile_offset = device.create_buffer(info);

	info.size = MAX_TILES_Y * sizeof(uint32_t);
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

	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(PerTileData);
	tile_instance_data = device.create_buffer(info);
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

				unsigned mask_count = __builtin_popcount(mask);
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