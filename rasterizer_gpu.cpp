#include <context.hpp>
#include "rasterizer_gpu.hpp"
#include "context.hpp"
#include "device.hpp"
#include <stdexcept>
#include "math.hpp"
#include "stb_image_write.h"
#include <string.h>

using namespace Granite;
using namespace Vulkan;

namespace RetroWarp
{
struct BBox
{
	int min_x, max_x, min_y, max_y;
};

constexpr unsigned MAX_NUM_SHADER_STATE_INDICES = 64;
constexpr unsigned MAX_NUM_RENDER_STATE_INDICES = 1024;

struct RasterizerGPU::Impl
{
	Device *device;
	BufferHandle color_buffer;
	BufferHandle depth_buffer;
	unsigned width = 0;
	unsigned height = 0;
	bool subgroup = false;
	bool ubershader = false;
	bool async_compute = false;
	unsigned num_state_indices = 0;

	struct
	{
		// First pass, bin at 64x64.
		BufferHandle mask_buffer_low_res;

		// Bin at 16x16.
		BufferHandle mask_buffer[2];

		// Groups group of 32 primitives into one 1 bit for faster rejection in raster.
		BufferHandle mask_buffer_coarse[2];
	} binning;

	struct
	{
		// Final resolved tile offsets.
		BufferHandle tile_offset[2];
	} tile_count;

	struct
	{
		BufferHandle color[2];
		BufferHandle depth[2];
		BufferHandle flags[2];
		unsigned index = 0;
		Semaphore rop_complete[2];
	} tile_instance_data;

	struct RenderState
	{
		int16_t scissor_x = 0;
		int16_t scissor_y = 0;
		int16_t scissor_width = 0;
		int16_t scissor_height = 0;
		uint8_t constant_color[4] = {};
		uint8_t depth_state = uint8_t(DepthWrite::On);
		uint8_t blend_state = uint8_t(BlendState::Replace);
		uint8_t combiner_state = 0;
		uint8_t alpha_threshold = 0;
	};
	static_assert(sizeof(RenderState) == 16, "Sizeof render state must be 16.");

	struct
	{
		BufferHandle positions;
		BufferHandle attributes;
		BufferHandle state_index;
		BufferHandle render_state_index;
		BufferHandle render_state;
		BufferHandle positions_gpu;
		BufferHandle attributes_gpu;
		BufferHandle state_index_gpu;
		BufferHandle render_state_index_gpu;
		BufferHandle render_state_gpu;
		PrimitiveSetupPos *mapped_positions = nullptr;
		PrimitiveSetupAttr *mapped_attributes = nullptr;
		uint8_t *mapped_state_index = nullptr;
		uint16_t *mapped_render_state_index = nullptr;
		RenderState *mapped_render_state = nullptr;
		unsigned count = 0;
		unsigned num_conservative_tile_instances = 0;
		bool host_visible = false;
	} staging;

	struct
	{
		BufferHandle item_count_per_variant;
		BufferHandle work_list_per_variant;
	} raster_work;

	struct
	{
		const ImageView *image_views[64] = {};
		unsigned state_count = 0;
		const ImageView *current_image = nullptr;
		RenderState last_render_state;
		RenderState current_render_state;
		unsigned render_state_count = 0;
	} state;

	void init(Device &device, bool subgroup, bool ubershader, bool async_compute);

	void reset_staging();
	void begin_staging();
	void end_staging();

	void init_binning_buffers();
	void init_prefix_sum_buffers();
	void init_tile_buffers();
	void init_raster_work_buffers();
	void flush();
	void flush_ubershader();
	void flush_split();
	ImageHandle copy_to_framebuffer();

	void queue_primitive(const PrimitiveSetup &setup);
	unsigned compute_num_conservative_tiles(const PrimitiveSetup &setup) const;

	BBox compute_bbox(const PrimitiveSetup &setup) const;
	bool clip_bbox_scissor(BBox &clipped_bbox, const BBox &bbox) const;

	void set_fb_info(CommandBuffer &cmd);
	void clear_indirect_buffer(CommandBuffer &cmd);
	void binning_low_res_prepass(CommandBuffer &cmd);
	void binning_full_res(CommandBuffer &cmd, bool ubershader);
	void dispatch_combiner_work(CommandBuffer &cmd);
	void run_rop(CommandBuffer &cmd);
	void run_rop_ubershader(CommandBuffer &cmd);

	bool can_support_minimum_subgroup_size(unsigned size) const;
	bool supports_subgroup_size_control() const;
};

struct FBInfo
{
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
constexpr int TILE_WIDTH = 8;
constexpr int TILE_HEIGHT = 8;
constexpr int TILE_WIDTH_LOG2 = 3;
constexpr int TILE_HEIGHT_LOG2 = 3;
constexpr int MAX_TILES_X = MAX_WIDTH / TILE_WIDTH;
constexpr int MAX_TILES_Y = MAX_HEIGHT / TILE_HEIGHT;
constexpr int MAX_TILES_X_LOW_RES = MAX_WIDTH / (4 * TILE_WIDTH);
constexpr int MAX_TILES_Y_LOW_RES = MAX_HEIGHT / (4 * TILE_HEIGHT);
constexpr int MAX_NUM_TILE_INSTANCES = 0xffff;
const int RASTER_ROUNDING = (1 << (SUBPIXELS_LOG2 + 16)) - 1;

struct TileRasterWork
{
	uint16_t tile_x, tile_y;
	uint16_t tile_instance;
	uint16_t primitive;
};

void RasterizerGPU::Impl::reset_staging()
{
	staging = {};
	state.render_state_count = 0;
	state.state_count = 0;
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
	int scissor_x = state.current_render_state.scissor_x;
	int scissor_y = state.current_render_state.scissor_y;
	int scissor_width = state.current_render_state.scissor_width;
	int scissor_height = state.current_render_state.scissor_height;

	clipped_bbox.min_x = std::max(scissor_x, bbox.min_x);
	clipped_bbox.max_x = std::min(scissor_x + scissor_width - 1, bbox.max_x);
	clipped_bbox.min_y = std::max(scissor_y, bbox.min_y);
	clipped_bbox.max_y = std::min(scissor_y + scissor_height - 1, bbox.max_y);

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
	info.domain = BufferDomain::Device;

	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupPos);
	staging.positions_gpu = device->create_buffer(info);
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupAttr);
	staging.attributes_gpu = device->create_buffer(info);
	info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.size = MAX_PRIMITIVES * sizeof(uint8_t);
	staging.state_index_gpu = device->create_buffer(info);
	info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.size = MAX_PRIMITIVES * sizeof(uint16_t);
	staging.render_state_index_gpu = device->create_buffer(info);
	info.size = MAX_NUM_RENDER_STATE_INDICES * sizeof(RenderState);
	staging.render_state_gpu = device->create_buffer(info);

	staging.mapped_positions = static_cast<PrimitiveSetupPos *>(
			device->map_host_buffer(*staging.positions_gpu,
			                       MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_attributes = static_cast<PrimitiveSetupAttr *>(
			device->map_host_buffer(*staging.attributes_gpu,
			                       MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_state_index = static_cast<uint8_t *>(
			device->map_host_buffer(*staging.state_index_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_render_state = static_cast<RenderState *>(
			device->map_host_buffer(*staging.render_state_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_render_state_index = static_cast<uint16_t *>(
			device->map_host_buffer(*staging.render_state_index_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));

	if (staging.mapped_positions && staging.mapped_attributes && staging.mapped_state_index &&
	    staging.mapped_render_state && staging.mapped_render_state_index)
	{
		staging.positions = staging.positions_gpu;
		staging.attributes = staging.attributes_gpu;
		staging.state_index = staging.state_index_gpu;
		staging.render_state = staging.render_state_gpu;
		staging.render_state_index = staging.render_state_index_gpu;
		staging.host_visible = true;
	}
	else
	{
		info.domain = BufferDomain::Host;
		info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

		info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupPos);
		staging.positions = device->create_buffer(info);

		info.size = MAX_PRIMITIVES * sizeof(PrimitiveSetupAttr);
		staging.attributes = device->create_buffer(info);

		info.size = MAX_PRIMITIVES * sizeof(uint8_t);
		staging.state_index = device->create_buffer(info);

		info.size = MAX_PRIMITIVES * sizeof(uint16_t);
		staging.render_state_index = device->create_buffer(info);

		info.size = MAX_NUM_RENDER_STATE_INDICES * sizeof(RenderState);
		staging.render_state = device->create_buffer(info);

		staging.mapped_positions = static_cast<PrimitiveSetupPos *>(
				device->map_host_buffer(*staging.positions,
				                        MEMORY_ACCESS_WRITE_BIT));
		staging.mapped_attributes = static_cast<PrimitiveSetupAttr *>(
				device->map_host_buffer(*staging.attributes,
				                        MEMORY_ACCESS_WRITE_BIT));
		staging.mapped_state_index = static_cast<uint8_t *>(
				device->map_host_buffer(*staging.state_index,
				                        MEMORY_ACCESS_WRITE_BIT));

		staging.mapped_render_state = static_cast<RenderState *>(
				device->map_host_buffer(*staging.render_state,
				                        MEMORY_ACCESS_WRITE_BIT));
		staging.mapped_render_state_index = static_cast<uint16_t *>(
				device->map_host_buffer(*staging.render_state_index,
				                        MEMORY_ACCESS_WRITE_BIT));

		staging.host_visible = false;
	}

	staging.count = 0;
	staging.num_conservative_tile_instances = 0;
}

void RasterizerGPU::Impl::end_staging()
{
	if (staging.mapped_positions)
		device->unmap_host_buffer(*staging.positions, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_attributes)
		device->unmap_host_buffer(*staging.attributes, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_state_index)
		device->unmap_host_buffer(*staging.state_index, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_render_state_index)
		device->unmap_host_buffer(*staging.render_state_index, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_render_state)
		device->unmap_host_buffer(*staging.render_state, MEMORY_ACCESS_WRITE_BIT);

	staging.mapped_positions = nullptr;
	staging.mapped_attributes = nullptr;
	staging.mapped_state_index = nullptr;
	staging.mapped_render_state_index = nullptr;
	staging.mapped_render_state = nullptr;

	if (!staging.host_visible && staging.count != 0)
	{
		auto cmd = device->request_command_buffer(CommandBuffer::Type::AsyncTransfer);
		cmd->copy_buffer(*staging.positions_gpu, 0, *staging.positions, 0, staging.count * sizeof(PrimitiveSetupPos));
		cmd->copy_buffer(*staging.attributes_gpu, 0, *staging.attributes, 0, staging.count * sizeof(PrimitiveSetupAttr));
		cmd->copy_buffer(*staging.state_index_gpu, 0, *staging.state_index, 0, staging.count * sizeof(uint8_t));
		cmd->copy_buffer(*staging.render_state_index_gpu, 0, *staging.render_state_index, 0, staging.count * sizeof(uint16_t));
		cmd->copy_buffer(*staging.render_state_gpu, 0, *staging.render_state, 0, state.render_state_count * sizeof(RenderState));
		Semaphore sem;
		device->submit(cmd, nullptr, 1, &sem);
		device->add_wait_semaphore(async_compute ? CommandBuffer::Type::AsyncCompute : CommandBuffer::Type::Generic, sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
	}
}

void RasterizerGPU::Impl::clear_indirect_buffer(CommandBuffer &cmd)
{
	cmd.begin_region("clear-indirect-buffer");
	cmd.set_program("assets://shaders/clear_indirect_buffers.comp");
	cmd.set_specialization_constant_mask(1);
	cmd.set_specialization_constant(0, MAX_NUM_SHADER_STATE_INDICES);
	cmd.set_storage_buffer(0, 0, *raster_work.item_count_per_variant);
	cmd.dispatch(1, 1, 1);
	cmd.end_region();
}

void RasterizerGPU::Impl::binning_low_res_prepass(CommandBuffer &cmd)
{
	cmd.begin_region("binning-low-res-prepass");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer_low_res);
	cmd.set_storage_buffer(0, 1, *staging.positions_gpu);
	cmd.set_uniform_buffer(0, 2, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 3, *staging.render_state_gpu);

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;

	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_BASIC_BIT;
	if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	    (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	    can_support_minimum_subgroup_size(subgroup_size) && subgroup_size <= 64)
	{
		cmd.set_program("assets://shaders/binning_low_res.comp", {{ "SUBGROUP", 1 }});
		cmd.set_specialization_constant_mask(1);
		cmd.set_specialization_constant(0, subgroup_size);

		if (supports_subgroup_size_control())
		{
			cmd.enable_subgroup_size_control(true);
			cmd.set_subgroup_size_log2(true, trailing_zeroes(subgroup_size), trailing_zeroes(subgroup_size));
		}
		cmd.dispatch((staging.count + subgroup_size - 1) / subgroup_size,
		             (width + 4 * TILE_WIDTH - 1) / (4 * TILE_WIDTH),
		             (height + 4 * TILE_HEIGHT - 1) / (4 * TILE_HEIGHT));
		cmd.enable_subgroup_size_control(false);
	}
	else
	{
		// Fallback with shared memory.
		cmd.set_program("assets://shaders/binning_low_res.comp", {{ "SUBGROUP", 0 }});
		cmd.dispatch((staging.count + 31) / 32,
		             (width + 4 * TILE_WIDTH - 1) / (4 * TILE_WIDTH),
		             (height + 4 * TILE_HEIGHT - 1) / (4 * TILE_HEIGHT));
	}
	cmd.end_region();
}

void RasterizerGPU::Impl::binning_full_res(CommandBuffer &cmd, bool ubershader)
{
	cmd.begin_region("binning-full-res");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 1, *staging.positions_gpu);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer_low_res);
	cmd.set_storage_buffer(0, 3, *binning.mask_buffer_coarse[tile_instance_data.index]);

	cmd.set_uniform_buffer(0, 4, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 5, *staging.render_state_gpu);

	if (!ubershader)
	{
		cmd.set_storage_buffer(0, 6, *tile_count.tile_offset[tile_instance_data.index]);
		cmd.set_storage_buffer(0, 7, *raster_work.item_count_per_variant);
		cmd.set_storage_buffer(0, 8, *raster_work.work_list_per_variant);
		cmd.set_storage_buffer(0, 9, *staging.state_index_gpu);
	}

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;

	uint32_t num_masks = (staging.count + 31) / 32;

	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BALLOT_BIT |
	                                        VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

	if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	    (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	    can_support_minimum_subgroup_size(subgroup_size) && subgroup_size <= 64)
	{
		cmd.set_program("assets://shaders/binning.comp", {{ "SUBGROUP", 1 }, { "UBERSHADER", ubershader ? 1 : 0 }});
		cmd.set_specialization_constant_mask(1);
		cmd.set_specialization_constant(0, subgroup_size);

		if (supports_subgroup_size_control())
		{
			cmd.enable_subgroup_size_control(true);
			cmd.set_subgroup_size_log2(true, trailing_zeroes(subgroup_size), trailing_zeroes(subgroup_size));
		}

		cmd.dispatch((num_masks + subgroup_size - 1) / subgroup_size,
		             (width + TILE_WIDTH - 1) / TILE_WIDTH,
		             (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

		cmd.enable_subgroup_size_control(false);
	}
	else
	{
		// Fallback with shared memory.
		cmd.set_program("assets://shaders/binning.comp", {{ "SUBGROUP", 0 }, { "UBERSHADER", ubershader ? 1 : 0 }});
		cmd.dispatch((num_masks + 31) / 32,
		             (width + TILE_WIDTH - 1) / TILE_WIDTH,
		             (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
	}

	cmd.end_region();
}

bool RasterizerGPU::Impl::can_support_minimum_subgroup_size(unsigned size) const
{
	// Vendor specific. AMD and NV have fixed subgroup sizes, no need to check for extension.
	uint32_t vendor_id = device->get_gpu_properties().vendorID;
	if (vendor_id == VENDOR_ID_AMD && size <= 64)
		return true;
	else if (vendor_id == VENDOR_ID_NVIDIA && size <= 32)
		return true;

	if (!supports_subgroup_size_control())
		return false;

	auto &features = device->get_device_features();

	if (size > features.subgroup_size_control_properties.maxSubgroupSize)
		return false;

	return true;
}

bool RasterizerGPU::Impl::supports_subgroup_size_control() const
{
	auto &features = device->get_device_features();

	if ((features.subgroup_size_control_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
		return false;
	if (!features.subgroup_size_control_features.computeFullSubgroups)
		return false;

	return true;
}

void RasterizerGPU::Impl::dispatch_combiner_work(CommandBuffer &cmd)
{
	cmd.begin_region("dispatch-combiner-work");
	cmd.set_storage_buffer(0, 1, *tile_instance_data.color[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 2, *tile_instance_data.depth[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 3, *tile_instance_data.flags[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 4, *staging.positions_gpu);
	cmd.set_storage_buffer(0, 5, *staging.attributes_gpu);
	cmd.set_uniform_buffer(0, 6, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 7, *staging.render_state_gpu);

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;
	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
	                                        VK_SUBGROUP_FEATURE_BALLOT_BIT;

	if (features.compute_shader_derivative_features.computeDerivativeGroupQuads)
	{
		cmd.set_program("assets://shaders/combiner.comp", { {"DERIVATIVE_GROUP_QUAD", 1}, {"SUBGROUP", 0} });
	}
	else if (features.compute_shader_derivative_features.computeDerivativeGroupLinear)
	{
		cmd.set_program("assets://shaders/combiner.comp", { {"DERIVATIVE_GROUP_LINEAR", 1}, {"SUBGROUP", 0} });
	}
	else if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	         (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	         can_support_minimum_subgroup_size(4))
	{
		cmd.set_program("assets://shaders/combiner.comp", { {"SUBGROUP", 1} });
		if (supports_subgroup_size_control())
		{
			cmd.set_subgroup_size_log2(true, 2, 7);
			cmd.enable_subgroup_size_control(true);
		}
	}
	else
	{
		cmd.set_program("assets://shaders/combiner.comp", { {"SUBGROUP", 0} });
	}

	for (unsigned variant = 0; variant < state.state_count; variant++)
	{
		cmd.set_storage_buffer(0, 0, *raster_work.work_list_per_variant,
		                       variant * (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork),
		                       (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork));
		assert(state.image_views[variant]);
		cmd.set_texture(1, 0, *state.image_views[variant], StockSampler::TrilinearWrap);
		cmd.dispatch_indirect(*raster_work.item_count_per_variant, 16 * variant);
	}
	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
}

void RasterizerGPU::Impl::set_fb_info(CommandBuffer &cmd)
{
	auto *fb_info = cmd.allocate_typed_constant_data<FBInfo>(2, 0, 1);
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

void RasterizerGPU::Impl::run_rop_ubershader(CommandBuffer &cmd)
{
	cmd.begin_region("run-rop");
	cmd.set_program("assets://shaders/rop_ubershader.comp");
	cmd.set_storage_buffer(0, 0, *color_buffer);
	cmd.set_storage_buffer(0, 1, *depth_buffer);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 3, *binning.mask_buffer_coarse[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 4, *staging.positions_gpu);
	cmd.set_storage_buffer(0, 5, *staging.attributes_gpu);
	cmd.set_uniform_buffer(0, 6, *staging.state_index_gpu);
	cmd.set_uniform_buffer(0, 7, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 8, *staging.render_state_gpu);

	for (unsigned i = 0; i < num_state_indices; i++)
	{
		cmd.set_texture(1, i, i < state.state_count ? *state.image_views[i] : *state.image_views[0],
		                StockSampler::TrilinearWrap);
	}

	auto &features = device->get_device_features();
	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
	                                        VK_SUBGROUP_FEATURE_BALLOT_BIT;

	if (features.compute_shader_derivative_features.computeDerivativeGroupQuads)
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {{"DERIVATIVE_GROUP_QUAD", 1}, {"SUBGROUP", 0}});
	}
	else if (features.compute_shader_derivative_features.computeDerivativeGroupLinear)
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {{"DERIVATIVE_GROUP_LINEAR", 1}, {"SUBGROUP", 0}});
	}
	else if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	         (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	         can_support_minimum_subgroup_size(4))
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {{"SUBGROUP", 1}});

		if (supports_subgroup_size_control())
		{
			cmd.set_subgroup_size_log2(true, 2, 7);
			cmd.enable_subgroup_size_control(true);
		}
	}
	else
		cmd.set_program("assets://shaders/rop_ubershader.comp", {{"SUBGROUP", 0}});

	cmd.dispatch((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1);
	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
}

void RasterizerGPU::Impl::run_rop(CommandBuffer &cmd)
{
	cmd.begin_region("run-rop");
	cmd.set_program("assets://shaders/rop.comp");
	cmd.set_storage_buffer(0, 0, *color_buffer);
	cmd.set_storage_buffer(0, 1, *depth_buffer);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 3, *binning.mask_buffer_coarse[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 4, *tile_instance_data.color[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 5, *tile_instance_data.depth[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 6, *tile_instance_data.flags[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 7, *tile_count.tile_offset[tile_instance_data.index]);
	cmd.set_uniform_buffer(0, 8, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 9, *staging.render_state_gpu);
	cmd.dispatch((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1);
	cmd.end_region();
}

void RasterizerGPU::Impl::flush_ubershader()
{
	end_staging();
	if (staging.count == 0)
		return;

	auto queue_type = async_compute ? CommandBuffer::Type::AsyncCompute : CommandBuffer::Type::Generic;

	auto cmd = device->request_command_buffer(queue_type);

	set_fb_info(*cmd);

	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	binning_low_res_prepass(*cmd);
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t0, t1, "binning-low-res-prepass");
	device->submit(cmd);

	auto &rop_sem = tile_instance_data.rop_complete[tile_instance_data.index];
	if (rop_sem)
	{
		device->add_wait_semaphore(queue_type, rop_sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
		rop_sem.reset();
	}

	cmd = device->request_command_buffer(queue_type);
	set_fb_info(*cmd);

	t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	binning_full_res(*cmd, true);

	auto t2 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t1, t2, "binning-full-res");

	Semaphore sem;
	device->submit(cmd, nullptr, 1, &sem);
	device->add_wait_semaphore(CommandBuffer::Type::Generic, sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);

	cmd = device->request_command_buffer();
	set_fb_info(*cmd);

	t2 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	run_rop_ubershader(*cmd);

	auto t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t2, t3, "rop-ubershader");

	sem.reset();
	device->submit(cmd, nullptr, 1, &sem);
	tile_instance_data.rop_complete[tile_instance_data.index] = sem;
	reset_staging();

	device->register_time_interval(t0, t3, "iteration");
	tile_instance_data.index ^= 1;
}

void RasterizerGPU::Impl::flush_split()
{
	end_staging();
	if (staging.count == 0)
		return;

	auto queue_type = async_compute ? CommandBuffer::Type::AsyncCompute : CommandBuffer::Type::Generic;

	auto cmd = device->request_command_buffer(queue_type);

	set_fb_info(*cmd);

	// This part can overlap with previous flush.
	// Clear indirect buffer.
	clear_indirect_buffer(*cmd);

	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Binning low-res prepass.
	binning_low_res_prepass(*cmd);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t0, t1, "binning-low-res-prepass");
	device->submit(cmd);

	// Need to wait until an earlier pass of ROP completes.
	auto &rop_sem = tile_instance_data.rop_complete[tile_instance_data.index];
	if (rop_sem)
	{
		device->add_wait_semaphore(queue_type,
			rop_sem,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
		rop_sem.reset();
	}

	cmd = device->request_command_buffer(queue_type);
	set_fb_info(*cmd);

	t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	// Binning at full-resolution.
	binning_full_res(*cmd, false);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
	             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

	auto t2 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t1, t2, "binning-full-res");

	dispatch_combiner_work(*cmd);

	auto t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t2, t3, "dispatch-combiner-work");

	// Hand off shaded result to ROP.
	Semaphore sem;
	device->submit(cmd, nullptr, 1, &sem);
	device->add_wait_semaphore(CommandBuffer::Type::Generic, sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
	cmd = device->request_command_buffer();
	set_fb_info(*cmd);

	t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	// ROP.
	run_rop(*cmd);

	auto t4 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval(t3, t4, "rop");

	device->register_time_interval(t0, t4, "iteration");

	sem.reset();
	device->submit(cmd, nullptr, 1, &sem);
	tile_instance_data.rop_complete[tile_instance_data.index] = sem;

	reset_staging();

	tile_instance_data.index ^= 1;
}

void RasterizerGPU::Impl::init_binning_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint32_t);
	for (auto &mask_buffer : binning.mask_buffer)
		mask_buffer = device->create_buffer(info);

	info.size = MAX_TILES_X_LOW_RES * MAX_TILES_Y_LOW_RES * TILE_BINNING_STRIDE * sizeof(uint32_t);
	binning.mask_buffer_low_res = device->create_buffer(info);

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE_COARSE * sizeof(uint32_t);
	for (auto &mask_buffer : binning.mask_buffer_coarse)
		mask_buffer = device->create_buffer(info);
}

void RasterizerGPU::Impl::init_prefix_sum_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_TILES_X * MAX_TILES_Y * TILE_BINNING_STRIDE * sizeof(uint16_t);
	for (auto &offset : tile_count.tile_offset)
		offset = device->create_buffer(info);
}

void RasterizerGPU::Impl::init_tile_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint32_t);
	for (auto &color : tile_instance_data.color)
		color = device->create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint16_t);
	for (auto &depth : tile_instance_data.depth)
		depth = device->create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * TILE_WIDTH * TILE_HEIGHT * sizeof(uint8_t);
	for (auto &flags : tile_instance_data.flags)
		flags = device->create_buffer(info);
}

void RasterizerGPU::Impl::init_raster_work_buffers()
{
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	// Round MAX_NUM_TILE_INSTANCES up to 0x10000.
	info.size = (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork) * MAX_NUM_SHADER_STATE_INDICES;
	raster_work.work_list_per_variant = device->create_buffer(info);

	info.size = MAX_NUM_SHADER_STATE_INDICES * (4 * sizeof(uint32_t));
	info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
	raster_work.item_count_per_variant = device->create_buffer(info);
}

template <typename T>
static std::vector<T> readback_buffer(Device *device, const Buffer &buffer)
{
	std::vector<T> result(buffer.get_create_info().size / sizeof(T));

	BufferCreateInfo info;
	info.domain = BufferDomain::CachedHost;
	info.size = buffer.get_create_info().size;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	auto readback_buffer = device->create_buffer(info);

	auto cmd = device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
	             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	cmd->copy_buffer(*readback_buffer, buffer);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	Fence fence;
	device->submit(cmd, &fence);
	fence->wait();

	const void *mapped = device->map_host_buffer(*readback_buffer, MEMORY_ACCESS_READ_BIT);
	memcpy(result.data(), mapped, result.size() * sizeof(T));
	device->unmap_host_buffer(*readback_buffer, MEMORY_ACCESS_READ_BIT);

	return result;
}

void RasterizerGPU::Impl::init(Device &device_, bool subgroup_, bool ubershader_, bool async_compute_)
{
	device = &device_;
	subgroup = subgroup_;
	ubershader = ubershader_;
	async_compute = async_compute_;
	num_state_indices = ubershader ? 16 : 64;

	auto &features = device->get_device_features();
	if (!features.storage_8bit_features.storageBuffer8BitAccess)
		throw std::runtime_error("8-bit storage not supported.");
	if (!features.storage_16bit_features.storageBuffer16BitAccess)
		throw std::runtime_error("16-bit storage not supported.");
	if (!features.ubo_std430_features.uniformBufferStandardLayout && !features.scalar_block_features.scalarBlockLayout)
		throw std::runtime_error("UBO std430 storage not supported.");

	init_binning_buffers();
	init_prefix_sum_buffers();
	init_tile_buffers();
	init_raster_work_buffers();
}

RasterizerGPU::RasterizerGPU()
{
	impl.reset(new Impl);
}

RasterizerGPU::~RasterizerGPU()
{
}

void RasterizerGPU::set_texture(const ImageView &view)
{
	impl->state.current_image = &view;
}

void RasterizerGPU::resize(unsigned width, unsigned height)
{
	flush();
	impl->width = width;
	impl->height = height;

	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.size = width * height * sizeof(uint32_t);
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	impl->color_buffer = impl->device->create_buffer(info);

	info.size = (width * height * sizeof(uint16_t) + 3) & ~3u;
	impl->depth_buffer = impl->device->create_buffer(info);

	impl->state.current_render_state.scissor_x = 0;
	impl->state.current_render_state.scissor_y = 0;
	impl->state.current_render_state.scissor_width = width;
	impl->state.current_render_state.scissor_height = height;
}

void RasterizerGPU::clear_depth(uint16_t z)
{
	flush();
	auto cmd = impl->device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_WRITE_BIT);

	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	cmd->fill_buffer(*impl->depth_buffer, (uint32_t(z) << 16) | z);
	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	impl->device->register_time_interval(t0, t1, "clear-depth");
	impl->device->submit(cmd);
}

void RasterizerGPU::clear_color(uint32_t rgba)
{
	flush();
	auto cmd = impl->device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_WRITE_BIT);
	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	cmd->fill_buffer(*impl->color_buffer, rgba);
	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	impl->device->register_time_interval(t0, t1, "clear-color");
	impl->device->submit(cmd);
}

void RasterizerGPU::set_depth_state(DepthTest mode, DepthWrite write)
{
	impl->state.current_render_state.depth_state = uint8_t(mode) | uint8_t(write);
}

void RasterizerGPU::set_rop_state(BlendState state)
{
	impl->state.current_render_state.blend_state = uint8_t(state);
}

void RasterizerGPU::set_scissor(int x, int y, int width, int height)
{
	impl->state.current_render_state.scissor_x = x;
	impl->state.current_render_state.scissor_y = y;
	impl->state.current_render_state.scissor_width = width;
	impl->state.current_render_state.scissor_height = height;
}

void RasterizerGPU::Impl::queue_primitive(const PrimitiveSetup &setup)
{
	unsigned num_conservative_tiles = ubershader ? 0 : compute_num_conservative_tiles(setup);
	bool state_changed = state.state_count != 0 && state.current_image != state.image_views[state.state_count - 1];
	bool render_state_changed = memcmp(&state.current_render_state, &state.last_render_state, sizeof(RenderState)) != 0;

	bool need_flush = false;
	if (staging.count == MAX_PRIMITIVES)
		need_flush = true;
	else if (staging.num_conservative_tile_instances + num_conservative_tiles > MAX_NUM_TILE_INSTANCES)
		need_flush = true;
	else if (state_changed && state.state_count == num_state_indices)
		need_flush = true;
	else if (render_state_changed && state.render_state_count == MAX_NUM_RENDER_STATE_INDICES)
		need_flush = true;

	if (need_flush)
		flush();

	if (staging.count == 0)
		begin_staging();

	unsigned current_state;
	unsigned current_render_state;

	if (state.state_count == 0 || state_changed)
	{
		state.image_views[state.state_count] = state.current_image;
		current_state = state.state_count;
		state.state_count++;
	}
	else
		current_state = state.state_count - 1;

	if (state.render_state_count == 0 || render_state_changed)
	{
		staging.mapped_render_state[state.render_state_count] = state.current_render_state;
		state.last_render_state = state.current_render_state;
		current_render_state = state.render_state_count;
		state.render_state_count++;
	}
	else
		current_render_state = state.render_state_count - 1;

	staging.mapped_positions[staging.count] = setup.pos;
	staging.mapped_attributes[staging.count] = setup.attr;
	staging.mapped_state_index[staging.count] = current_state;
	staging.mapped_render_state_index[staging.count] = current_render_state;

	staging.count++;
	staging.num_conservative_tile_instances += num_conservative_tiles;
}

void RasterizerGPU::rasterize_primitives(const RetroWarp::PrimitiveSetup *setup, size_t count)
{
	for (size_t i = 0; i < count; i++)
		impl->queue_primitive(setup[i]);
}

ImageHandle RasterizerGPU::copy_to_framebuffer()
{
	flush();
	return impl->copy_to_framebuffer();
}

ImageHandle RasterizerGPU::Impl::copy_to_framebuffer()
{
	ImageCreateInfo info = ImageCreateInfo::immutable_2d_image(width, height, VK_FORMAT_R8G8B8A8_SRGB);
	info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	auto image = device->create_image(info);

	auto cmd = device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	cmd->image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
	                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->copy_buffer_to_image(*image, *color_buffer, 0, {}, { width, height, 1 }, 0, 0,
	                          { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
	cmd->image_barrier(*image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	device->submit(cmd);
	return image;
}

bool RasterizerGPU::save_canvas(const char *path)
{
	impl->flush();

	auto cmd = impl->device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_READ_BIT);

	BufferCreateInfo info;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.domain = BufferDomain::CachedHost;
	info.size = impl->width * impl->height * sizeof(uint32_t);
	auto dst_buffer = impl->device->create_buffer(info);
	cmd->copy_buffer(*dst_buffer, *impl->color_buffer);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	Fence fence;
	impl->device->submit(cmd, &fence);

	fence->wait();
	auto *ptr = static_cast<uint32_t *>(impl->device->map_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT | MEMORY_ACCESS_WRITE_BIT));
	for (unsigned i = 0; i < impl->width * impl->height; i++)
		ptr[i] |= 0xff000000u;
	bool res = stbi_write_png(path, impl->width, impl->height, 4, ptr, impl->width * 4);
	impl->device->unmap_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT | MEMORY_ACCESS_WRITE_BIT);
	return res;
}

void RasterizerGPU::init(Device &device, bool subgroup, bool ubershader, bool async_compute)
{
	impl->init(device, subgroup, ubershader, async_compute);
}

void RasterizerGPU::flush()
{
	impl->flush();
}

void RasterizerGPU::Impl::flush()
{
	if (ubershader)
		flush_ubershader();
	else
		flush_split();

	reset_staging();
}

void RasterizerGPU::set_constant_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	impl->state.current_render_state.constant_color[0] = r;
	impl->state.current_render_state.constant_color[1] = g;
	impl->state.current_render_state.constant_color[2] = b;
	impl->state.current_render_state.constant_color[3] = a;
}

void RasterizerGPU::set_alpha_threshold(uint8_t threshold)
{
	impl->state.current_render_state.alpha_threshold = threshold;
}

void RasterizerGPU::set_combiner_mode(CombinerFlags flags)
{
	impl->state.current_render_state.combiner_state = flags;
}

}
