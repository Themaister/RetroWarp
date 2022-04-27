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
constexpr unsigned VRAM_SIZE = 64 * 1024 * 1024;

struct RasterizerGPU::Impl
{
	Device *device;
	BufferHandle vram_buffer;

	struct Framebuffer
	{
		uint32_t offset = 0;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t stride = 0;
	};

	Framebuffer color, depth;

	bool subgroup = false;
	bool ubershader = false;
	bool async_compute = false;

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
		TextureDescriptor tex;
	};
	static_assert(sizeof(RenderState) == 64, "Sizeof render state must be 64.");

	struct
	{
		BufferHandle positions;
		BufferHandle attributes;
		BufferHandle shader_state_index;
		BufferHandle render_state_index;
		BufferHandle render_state;
		BufferHandle positions_gpu;
		BufferHandle attributes_gpu;
		BufferHandle shader_state_index_gpu;
		BufferHandle render_state_index_gpu;
		BufferHandle render_state_gpu;
		PrimitiveSetupPos *mapped_positions = nullptr;
		PrimitiveSetupAttr *mapped_attributes = nullptr;
		uint8_t *mapped_shader_state_index = nullptr;
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
		uint32_t shader_states[MAX_NUM_SHADER_STATE_INDICES] = {};
		unsigned shader_state_count = 0;
		uint32_t current_shader_state = 0;

		RenderState last_render_state;
		RenderState current_render_state;
		unsigned render_state_count = 0;
	} state;

	void init(Device &device, bool subgroup, bool ubershader, bool async_compute, unsigned tile_size);

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
	bool supports_subgroup_size_control(uint32_t minimum_size, uint32_t maximum_size) const;

	int tile_size = 0;
	int tile_size_log2 = 0;
	int max_tiles_x = 0;
	int max_tiles_y = 0;
	int max_tiles_x_low_res = 0;
	int max_tiles_y_low_res = 0;

	uint32_t compute_shader_state() const;
};

struct FBInfo
{
	uvec2 resolution;
	uvec2 resolution_tiles;
	uint32_t primitive_count;
	uint32_t primitive_count_32;
	uint32_t primitive_count_1024;

	uint32_t color_offset;
	uint32_t color_width;
	uint32_t color_height;
	uint32_t color_stride;

	uint32_t depth_offset;
	uint32_t depth_width;
	uint32_t depth_height;
	uint32_t depth_stride;
};

constexpr int MAX_PRIMITIVES = 0x4000;
constexpr int TILE_BINNING_STRIDE = MAX_PRIMITIVES / 32;
constexpr int TILE_BINNING_STRIDE_COARSE = TILE_BINNING_STRIDE / 32;
constexpr int MAX_WIDTH = 2048;
constexpr int MAX_HEIGHT = 2048;
constexpr int TILE_DOWNSAMPLE = 8;
constexpr int TILE_DOWNSAMPLE_LOG2 = 3;
constexpr int MAX_NUM_TILE_INSTANCES = 0xffff;
const int RASTER_ROUNDING = (1 << (SUBPIXELS_LOG2 + 16)) - 1;

struct TileRasterWork
{
	uint32_t tile_x, tile_y;
	uint32_t tile_instance;
	uint32_t primitive;
};

void RasterizerGPU::Impl::reset_staging()
{
	staging = {};
	state.render_state_count = 0;
	state.shader_state_count = 0;
}

uint32_t RasterizerGPU::Impl::compute_shader_state() const
{
	// Ignore shader state for ubershaders.
	if (ubershader)
		return 0;

	uint32_t shader_state = 0;
	shader_state |= state.current_render_state.combiner_state;
	shader_state |= state.current_render_state.alpha_threshold << 8u;
	shader_state |= state.current_render_state.tex.texture_fmt << 16u;
	return shader_state;
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

	int start_tile_x = clipped_bbox.min_x >> tile_size_log2;
	int end_tile_x = clipped_bbox.max_x >> tile_size_log2;
	int start_tile_y = clipped_bbox.min_y >> tile_size_log2;
	int end_tile_y = clipped_bbox.max_y >> tile_size_log2;
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
	staging.shader_state_index_gpu = device->create_buffer(info);
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
	staging.mapped_shader_state_index = static_cast<uint8_t *>(
			device->map_host_buffer(*staging.shader_state_index_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_render_state = static_cast<RenderState *>(
			device->map_host_buffer(*staging.render_state_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));
	staging.mapped_render_state_index = static_cast<uint16_t *>(
			device->map_host_buffer(*staging.render_state_index_gpu,
			                        MEMORY_ACCESS_WRITE_BIT));

	if (staging.mapped_positions && staging.mapped_attributes && staging.mapped_shader_state_index &&
	    staging.mapped_render_state && staging.mapped_render_state_index)
	{
		staging.positions = staging.positions_gpu;
		staging.attributes = staging.attributes_gpu;
		staging.shader_state_index = staging.shader_state_index_gpu;
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
		staging.shader_state_index = device->create_buffer(info);

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
		staging.mapped_shader_state_index = static_cast<uint8_t *>(
				device->map_host_buffer(*staging.shader_state_index,
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
	if (staging.mapped_shader_state_index)
		device->unmap_host_buffer(*staging.shader_state_index, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_render_state_index)
		device->unmap_host_buffer(*staging.render_state_index, MEMORY_ACCESS_WRITE_BIT);
	if (staging.mapped_render_state)
		device->unmap_host_buffer(*staging.render_state, MEMORY_ACCESS_WRITE_BIT);

	staging.mapped_positions = nullptr;
	staging.mapped_attributes = nullptr;
	staging.mapped_shader_state_index = nullptr;
	staging.mapped_render_state_index = nullptr;
	staging.mapped_render_state = nullptr;

	if (!staging.host_visible && staging.count != 0)
	{
		auto cmd = device->request_command_buffer(CommandBuffer::Type::AsyncTransfer);
		cmd->copy_buffer(*staging.positions_gpu, 0, *staging.positions, 0, staging.count * sizeof(PrimitiveSetupPos));
		cmd->copy_buffer(*staging.attributes_gpu, 0, *staging.attributes, 0, staging.count * sizeof(PrimitiveSetupAttr));
		cmd->copy_buffer(*staging.shader_state_index_gpu, 0, *staging.shader_state_index, 0, staging.count * sizeof(uint8_t));
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
	cmd.set_specialization_constant_mask(0);
}

void RasterizerGPU::Impl::binning_low_res_prepass(CommandBuffer &cmd)
{
	uint32_t width = std::max(color.width, depth.width);
	uint32_t height = std::max(color.height, depth.height);

	cmd.begin_region("binning-low-res-prepass");
	cmd.set_storage_buffer(0, 0, *binning.mask_buffer_low_res);
	cmd.set_storage_buffer(0, 1, *staging.positions_gpu);
	cmd.set_uniform_buffer(0, 2, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 3, *staging.render_state_gpu);

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;

#if 0
	if (features.subgroup_size_control_features.subgroupSizeControl &&
	    (features.subgroup_size_control_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
	    features.subgroup_size_control_properties.minSubgroupSize <= 32 &&
	    features.subgroup_size_control_properties.maxSubgroupSize >= 32)
	{
		// Prefer subgroup size of 32 if possible.
		subgroup_size = 32;
	}
#endif

	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_BASIC_BIT;
	if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	    (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	    can_support_minimum_subgroup_size(32) && subgroup_size <= 64)
	{
		cmd.set_program("assets://shaders/binning_low_res.comp", {{ "SUBGROUP", 1 }, { "TILE_SIZE", tile_size }});
		cmd.set_specialization_constant_mask(1);
		cmd.set_specialization_constant(0, subgroup_size);

		if (supports_subgroup_size_control(32, subgroup_size))
		{
			cmd.enable_subgroup_size_control(true);
			cmd.set_subgroup_size_log2(true, 5, trailing_zeroes(subgroup_size));
		}
		cmd.dispatch((staging.count + subgroup_size - 1) / subgroup_size,
		             (width + TILE_DOWNSAMPLE * tile_size - 1) / (TILE_DOWNSAMPLE * tile_size),
		             (height + TILE_DOWNSAMPLE * tile_size - 1) / (TILE_DOWNSAMPLE * tile_size));
		cmd.enable_subgroup_size_control(false);
	}
	else
	{
		// Fallback with shared memory.
		cmd.set_program("assets://shaders/binning_low_res.comp", {{ "SUBGROUP", 0 }, { "TILE_SIZE", tile_size }});
		cmd.dispatch((staging.count + 31) / 32,
		             (width + TILE_DOWNSAMPLE * tile_size - 1) / (TILE_DOWNSAMPLE * tile_size),
		             (height + TILE_DOWNSAMPLE * tile_size - 1) / (TILE_DOWNSAMPLE * tile_size));
	}
	cmd.end_region();
	cmd.set_specialization_constant_mask(0);
}

void RasterizerGPU::Impl::binning_full_res(CommandBuffer &cmd, bool ubershader)
{
	uint32_t width = std::max(color.width, depth.width);
	uint32_t height = std::max(color.height, depth.height);
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
		cmd.set_storage_buffer(0, 9, *staging.shader_state_index_gpu);
	}

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;

#if 0
	if (features.subgroup_size_control_features.subgroupSizeControl &&
	    (features.subgroup_size_control_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
	    features.subgroup_size_control_properties.minSubgroupSize <= 32 &&
	    features.subgroup_size_control_properties.maxSubgroupSize >= 32)
	{
		// Prefer subgroup size of 32 if possible.
		subgroup_size = 32;
	}
#endif

	uint32_t num_masks = (staging.count + 31) / 32;

	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BALLOT_BIT |
	                                        VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_VOTE_BIT |
	                                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

	if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	    (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	    can_support_minimum_subgroup_size(32))
	{
		cmd.set_program("assets://shaders/binning.comp", {{ "SUBGROUP", 1 }, { "UBERSHADER", ubershader ? 1 : 0 }, { "TILE_SIZE", tile_size }});
		cmd.set_specialization_constant_mask(1);
		cmd.set_specialization_constant(0, subgroup_size);

		if (supports_subgroup_size_control(32, subgroup_size))
		{
			cmd.enable_subgroup_size_control(true);
			cmd.set_subgroup_size_log2(true, 5, trailing_zeroes(subgroup_size));
		}

		cmd.dispatch((num_masks + subgroup_size - 1) / subgroup_size,
		             (width + tile_size - 1) / tile_size,
		             (height + tile_size - 1) / tile_size);

		cmd.enable_subgroup_size_control(false);
	}
	else
	{
		// Fallback with shared memory.
		cmd.set_program("assets://shaders/binning.comp", {{ "SUBGROUP", 0 }, { "UBERSHADER", ubershader ? 1 : 0 }, { "TILE_SIZE", tile_size }});
		cmd.dispatch((num_masks + 31) / 32,
		             (width + tile_size - 1) / tile_size,
		             (height + tile_size - 1) / tile_size);
	}

	cmd.end_region();
	cmd.set_specialization_constant_mask(0);
}

bool RasterizerGPU::Impl::can_support_minimum_subgroup_size(unsigned size) const
{
	return supports_subgroup_size_control(size, device->get_device_features().subgroup_properties.subgroupSize);
}

bool RasterizerGPU::Impl::supports_subgroup_size_control(uint32_t minimum_size, uint32_t maximum_size) const
{
	auto &features = device->get_device_features();

	if (!features.subgroup_size_control_features.computeFullSubgroups)
		return false;

	bool use_varying = minimum_size <= features.subgroup_size_control_properties.minSubgroupSize &&
	                   maximum_size >= features.subgroup_size_control_properties.maxSubgroupSize;

	if (!use_varying)
	{
		bool outside_range = minimum_size > features.subgroup_size_control_properties.maxSubgroupSize ||
		                     maximum_size < features.subgroup_size_control_properties.minSubgroupSize;
		if (outside_range)
			return false;

		if ((features.subgroup_size_control_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
			return false;
	}

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
	cmd.set_storage_buffer(0, 8, *vram_buffer);

	auto &features = device->get_device_features();
	uint32_t subgroup_size = features.subgroup_properties.subgroupSize;
	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
	                                        VK_SUBGROUP_FEATURE_BALLOT_BIT;

	if (subgroup && features.compute_shader_derivative_features.computeDerivativeGroupQuads)
	{
		cmd.set_program("assets://shaders/combiner.comp", {
			{"DERIVATIVE_GROUP_QUAD", 1},
			{"SUBGROUP", 0},
			{"TILE_SIZE", tile_size},
			{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}
	else if (subgroup && features.compute_shader_derivative_features.computeDerivativeGroupLinear)
	{
		cmd.set_program("assets://shaders/combiner.comp", {
				{"DERIVATIVE_GROUP_LINEAR", 1},
				{"SUBGROUP", 0},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}
	else if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	         (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	         can_support_minimum_subgroup_size(4))
	{
		cmd.set_program("assets://shaders/combiner.comp", {
				{"SUBGROUP", 1},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});

		if (supports_subgroup_size_control(4, 64))
		{
			cmd.set_subgroup_size_log2(true, 2, 6);
			cmd.enable_subgroup_size_control(true);
		}
	}
	else
	{
		cmd.set_program("assets://shaders/combiner.comp", {
				{"SUBGROUP", 0},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}

	cmd.set_specialization_constant_mask(1);

	for (unsigned variant = 0; variant < state.shader_state_count; variant++)
	{
		cmd.set_specialization_constant(0, state.shader_states[variant]);
		cmd.set_storage_buffer(0, 0, *raster_work.work_list_per_variant,
		                       variant * (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork),
		                       (MAX_NUM_TILE_INSTANCES + 1) * sizeof(TileRasterWork));
		cmd.dispatch_indirect(*raster_work.item_count_per_variant, 16 * variant);
	}

	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
	cmd.set_specialization_constant_mask(0);
}

void RasterizerGPU::Impl::set_fb_info(CommandBuffer &cmd)
{
	uint32_t width = std::max(color.width, depth.width);
	uint32_t height = std::max(color.height, depth.height);

	auto *fb_info = cmd.allocate_typed_constant_data<FBInfo>(2, 0, 1);
	fb_info->resolution.x = width;
	fb_info->resolution.y = height;
	fb_info->resolution_tiles.x = (width + tile_size - 1) / tile_size;
	fb_info->resolution_tiles.y = (height + tile_size - 1) / tile_size;
	fb_info->primitive_count = staging.count;
	uint32_t num_masks = (staging.count + 31) / 32;
	fb_info->primitive_count_32 = num_masks;
	fb_info->primitive_count_1024 = (staging.count + 1023) / 1024;

	fb_info->color_offset = color.offset >> 1u;
	fb_info->color_width = color.width;
	fb_info->color_height = color.height;
	fb_info->color_stride = color.stride >> 1u;
	fb_info->depth_offset = depth.offset >> 1u;
	fb_info->depth_width = depth.width;
	fb_info->depth_height = depth.height;
	fb_info->depth_stride = depth.stride >> 1u;
}

void RasterizerGPU::Impl::run_rop_ubershader(CommandBuffer &cmd)
{
	uint32_t width = std::max(color.width, depth.width);
	uint32_t height = std::max(color.height, depth.height);

	cmd.begin_region("run-rop");
	cmd.set_storage_buffer(0, 0, *vram_buffer);
	cmd.set_storage_buffer(0, 1, *binning.mask_buffer[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer_coarse[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 3, *staging.positions_gpu);
	cmd.set_storage_buffer(0, 4, *staging.attributes_gpu);
	cmd.set_uniform_buffer(0, 5, *staging.shader_state_index_gpu);
	cmd.set_uniform_buffer(0, 6, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 7, *staging.render_state_gpu);

	auto &features = device->get_device_features();
	const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT |
	                                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
	                                        VK_SUBGROUP_FEATURE_BALLOT_BIT;

	if (subgroup && features.compute_shader_derivative_features.computeDerivativeGroupQuads)
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {
			{"DERIVATIVE_GROUP_QUAD", 1},
			{"SUBGROUP", 0},
			{"TILE_SIZE", tile_size},
			{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}
	else if (subgroup && features.compute_shader_derivative_features.computeDerivativeGroupLinear)
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {
				{"DERIVATIVE_GROUP_LINEAR", 1},
				{"SUBGROUP", 0},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}
	else if (subgroup && (features.subgroup_properties.supportedOperations & required) == required &&
	         (features.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	         can_support_minimum_subgroup_size(4))
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {
				{"SUBGROUP", 1},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});

		if (supports_subgroup_size_control(4, 128))
		{
			cmd.set_subgroup_size_log2(true, 2, 7);
			cmd.enable_subgroup_size_control(true);
		}
	}
	else
	{
		cmd.set_program("assets://shaders/rop_ubershader.comp", {
				{"SUBGROUP", 0},
				{"TILE_SIZE", tile_size},
				{"TILE_SIZE_SQUARE", tile_size * tile_size},
		});
	}

	cmd.dispatch((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, 1);
	cmd.end_region();
	cmd.enable_subgroup_size_control(false);
}

void RasterizerGPU::Impl::run_rop(CommandBuffer &cmd)
{
	uint32_t width = std::max(color.width, depth.width);
	uint32_t height = std::max(color.height, depth.height);
	cmd.begin_region("run-rop");
	cmd.set_program("assets://shaders/rop.comp", {
		{"TILE_SIZE", tile_size}
	});

	cmd.set_storage_buffer(0, 0, *vram_buffer);
	cmd.set_storage_buffer(0, 1, *binning.mask_buffer[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 2, *binning.mask_buffer_coarse[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 3, *tile_instance_data.color[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 4, *tile_instance_data.depth[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 5, *tile_instance_data.flags[tile_instance_data.index]);
	cmd.set_storage_buffer(0, 6, *tile_count.tile_offset[tile_instance_data.index]);
	cmd.set_uniform_buffer(0, 7, *staging.render_state_index_gpu);
	cmd.set_uniform_buffer(0, 8, *staging.render_state_gpu);

	cmd.dispatch((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, 1);
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
	device->register_time_interval("GPU", t0, t1, "binning-low-res-prepass");
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
	device->register_time_interval("GPU", t1, t2, "binning-full-res");

	Semaphore sem;
	device->submit(cmd, nullptr, 1, &sem);
	device->add_wait_semaphore(CommandBuffer::Type::Generic, sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);

	cmd = device->request_command_buffer();
	set_fb_info(*cmd);

	t2 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	run_rop_ubershader(*cmd);

	auto t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval("GPU", t2, t3, "rop-ubershader");

	sem.reset();
	device->submit(cmd, nullptr, 1, &sem);
	tile_instance_data.rop_complete[tile_instance_data.index] = sem;
	reset_staging();

	device->register_time_interval("GPU", t0, t3, "iteration");
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
	device->register_time_interval("GPU", t0, t1, "binning-low-res-prepass");
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
	device->register_time_interval("GPU", t1, t2, "binning-full-res");

	dispatch_combiner_work(*cmd);

	auto t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval("GPU", t2, t3, "dispatch-combiner-work");

	// Hand off shaded result to ROP.
	Semaphore sem;
	device->submit(cmd, nullptr, 1, &sem);
	device->add_wait_semaphore(CommandBuffer::Type::Generic, sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
	cmd = device->request_command_buffer();
	set_fb_info(*cmd);

	t3 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	// ROP.
	run_rop(*cmd);

	auto t4 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	device->register_time_interval("GPU", t3, t4, "rop");

	device->register_time_interval("GPU", t0, t4, "iteration");

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

	info.size = max_tiles_x * max_tiles_y * TILE_BINNING_STRIDE * sizeof(uint32_t);
	for (auto &mask_buffer : binning.mask_buffer)
		mask_buffer = device->create_buffer(info);

	info.size = max_tiles_x_low_res * max_tiles_y_low_res * TILE_BINNING_STRIDE * sizeof(uint32_t);
	binning.mask_buffer_low_res = device->create_buffer(info);

	info.size = max_tiles_x * max_tiles_y * TILE_BINNING_STRIDE_COARSE * sizeof(uint32_t);
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

	info.size = max_tiles_x * max_tiles_y * TILE_BINNING_STRIDE * sizeof(uint16_t);
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

	info.size = MAX_NUM_TILE_INSTANCES * tile_size * tile_size * sizeof(uint32_t);
	for (auto &color : tile_instance_data.color)
		color = device->create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * tile_size * tile_size * sizeof(uint16_t);
	for (auto &depth : tile_instance_data.depth)
		depth = device->create_buffer(info);
	info.size = MAX_NUM_TILE_INSTANCES * tile_size * tile_size * sizeof(uint8_t);
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

void RasterizerGPU::Impl::init(Device &device_, bool subgroup_, bool ubershader_, bool async_compute_, unsigned tile_size_)
{
	device = &device_;
	subgroup = subgroup_;
	ubershader = ubershader_;
	async_compute = async_compute_;
	tile_size = tile_size_;

	tile_size_log2 = trailing_zeroes(tile_size);
	max_tiles_x = MAX_WIDTH / tile_size;
	max_tiles_y = MAX_HEIGHT / tile_size;
	max_tiles_x_low_res = MAX_WIDTH / (TILE_DOWNSAMPLE * tile_size);
	max_tiles_y_low_res = MAX_HEIGHT / (TILE_DOWNSAMPLE * tile_size);

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

	BufferCreateInfo vram_info = {};
	vram_info.domain = BufferDomain::Device;
	vram_info.size = VRAM_SIZE;
	vram_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	vram_info.misc = BUFFER_MISC_ZERO_INITIALIZE_BIT;
	vram_buffer = device->create_buffer(vram_info);
}

RasterizerGPU::RasterizerGPU()
{
	impl.reset(new Impl);
}

RasterizerGPU::~RasterizerGPU()
{
}

void RasterizerGPU::set_texture_descriptor(const TextureDescriptor &desc)
{
	impl->state.current_render_state.tex = desc;
}

void RasterizerGPU::set_color_framebuffer(unsigned offset, unsigned width, unsigned height, unsigned stride)
{
	flush();
	impl->color.offset = offset;
	impl->color.width = width;
	impl->color.height = height;
	impl->color.stride = stride;

	impl->state.current_render_state.scissor_x = 0;
	impl->state.current_render_state.scissor_y = 0;
	impl->state.current_render_state.scissor_width = width;
	impl->state.current_render_state.scissor_height = height;
}

void RasterizerGPU::set_depth_framebuffer(unsigned offset, unsigned width, unsigned height, unsigned stride)
{
	flush();
	impl->depth.offset = offset;
	impl->depth.width = width;
	impl->depth.height = height;
	impl->depth.stride = stride;
}

void RasterizerGPU::clear_depth(uint16_t z)
{
	flush();
	auto cmd = impl->device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT);
	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	cmd->set_storage_buffer(0, 0, *impl->vram_buffer);
	cmd->set_program("assets://shaders/clear_framebuffer.comp",
	                 {{ "TILE_SIZE", impl->tile_size }});

	struct Registers
	{
		uint32_t offset;
		uint32_t width;
		uint32_t height;
		uint32_t stride;
		uint32_t value;
	} registers;

	registers.offset = impl->depth.offset >> 1;
	registers.width = impl->depth.width;
	registers.height = impl->depth.height;
	registers.stride = impl->depth.stride >> 1;
	registers.value = z;
	cmd->push_constants(&registers, 0, sizeof(registers));

	cmd->dispatch((registers.width + 15) / 16, (registers.height + 15) / 16, 1);

	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	impl->device->register_time_interval("GPU", t0, t1, "clear-depth");
	impl->device->submit(cmd);
}

void RasterizerGPU::copy_texture_rgba8888_to_vram(uint32_t offset, const uint32_t *src, unsigned width, unsigned height, TextureFormatBits fmt)
{
	flush();

	struct Registers
	{
		uint32_t offset;
		uint32_t blocks_width;
		uint32_t blocks_height;
		uint32_t width;
		uint32_t height;
	} registers;

	switch (fmt)
	{
	case TEXTURE_FMT_ARGB1555:
	case TEXTURE_FMT_LA88:
		registers.blocks_width = (width + 7) / 8;
		break;

	case TEXTURE_FMT_I8:
		registers.blocks_width = (width + 15) / 16;
		break;

	default:
		return;
	}

	BufferCreateInfo info = {};
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	info.domain = BufferDomain::Host;
	info.size = width * height * sizeof(uint32_t);

	auto buffer = impl->device->create_buffer(info, src);
	auto cmd = impl->device->request_command_buffer();

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT);

	cmd->set_storage_buffer(0, 0, *impl->vram_buffer);
	cmd->set_storage_buffer(0, 1, *buffer);
	cmd->set_program("assets://shaders/copy_framebuffer.comp",
	                 {{ "TILE_SIZE", impl->tile_size }, { "FMT", int(fmt) }});

	registers.offset = offset >> 1;
	registers.blocks_height = (height + 7) / 8;

	registers.width = width;
	registers.height = height;
	cmd->push_constants(&registers, 0, sizeof(registers));
	cmd->dispatch(registers.blocks_width, registers.blocks_height, 1);
	impl->device->submit(cmd);
}

void RasterizerGPU::clear_color(uint32_t rgba)
{
	flush();
	auto cmd = impl->device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT);
	auto t0 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	cmd->set_storage_buffer(0, 0, *impl->vram_buffer);
	cmd->set_program("assets://shaders/clear_framebuffer.comp",
	                 {{ "TILE_SIZE", impl->tile_size }});

	struct Registers
	{
		uint32_t offset;
		uint32_t width;
		uint32_t height;
		uint32_t stride;
		uint32_t value;
	} registers;

	registers.offset = impl->color.offset >> 1;
	registers.width = impl->color.width;
	registers.height = impl->color.height;
	registers.stride = impl->color.stride >> 1;
	registers.value = rgba;
	cmd->push_constants(&registers, 0, sizeof(registers));

	cmd->dispatch((registers.width + 15) / 16, (registers.height + 15) / 16, 1);

	auto t1 = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	impl->device->register_time_interval("GPU", t0, t1, "clear-color");
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

	state.current_shader_state = compute_shader_state();
	bool shader_state_changed = state.shader_state_count != 0 &&
	                            state.current_shader_state != state.shader_states[state.shader_state_count - 1];

	bool render_state_changed = memcmp(&state.current_render_state, &state.last_render_state, sizeof(RenderState)) != 0;

	bool need_flush = false;
	if (staging.count == MAX_PRIMITIVES)
		need_flush = true;
	else if (staging.num_conservative_tile_instances + num_conservative_tiles > MAX_NUM_TILE_INSTANCES)
		need_flush = true;
	else if (shader_state_changed && state.shader_state_count == MAX_NUM_SHADER_STATE_INDICES)
		need_flush = true;
	else if (render_state_changed && state.render_state_count == MAX_NUM_RENDER_STATE_INDICES)
		need_flush = true;

	if (need_flush)
		flush();

	if (staging.count == 0)
		begin_staging();

	unsigned current_shader_state;
	unsigned current_render_state;

	if (state.shader_state_count == 0 || shader_state_changed)
	{
		state.shader_states[state.shader_state_count] = state.current_shader_state;
		current_shader_state = state.shader_state_count;
		state.shader_state_count++;
	}
	else
		current_shader_state = state.shader_state_count - 1;

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
	staging.mapped_shader_state_index[staging.count] = current_shader_state;
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
	ImageCreateInfo info = ImageCreateInfo::immutable_2d_image(color.width, color.height, VK_FORMAT_A1R5G5B5_UNORM_PACK16);
	info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	auto image = device->create_image(info);

	auto cmd = device->request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	cmd->image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
	                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->copy_buffer_to_image(*image, *vram_buffer, color.offset, {}, { color.width, color.height, 1 }, color.stride / 2, 0,
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
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	info.domain = BufferDomain::CachedHost;
	info.size = impl->color.width * impl->color.height * sizeof(uint16_t);
	auto dst_buffer = impl->device->create_buffer(info);
	cmd->set_program("assets://shaders/read_framebuffer.comp", {{ "TILE_SIZE", impl->tile_size }});
	cmd->set_storage_buffer(0, 0, *dst_buffer);
	cmd->set_storage_buffer(0, 1, *impl->vram_buffer);
	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	struct Registers
	{
		uint32_t offset;
		uint32_t width;
		uint32_t height;
		uint32_t stride;
	} registers;
	registers.offset = impl->color.offset >> 1;
	registers.width = impl->color.width;
	registers.height = impl->color.height;
	registers.stride = impl->color.stride >> 1;
	cmd->push_constants(&registers, 0, sizeof(registers));

	cmd->dispatch((impl->color.width + 15) / 16, (impl->color.height + 15) / 16, 1);

	Fence fence;
	impl->device->submit(cmd, &fence);

	fence->wait();

	std::vector<u8vec4> readback_result(impl->color.width * impl->color.height);

	const auto unpack = [](uint16_t v) -> u8vec4 {
		unsigned r = (v >> 10) & 31;
		unsigned g = (v >> 5) & 31;
		unsigned b = (v >> 0) & 31;
		unsigned a = v >> 15;
		return u8vec4((r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2), 0xffu);
	};

	auto *ptr = static_cast<uint16_t *>(impl->device->map_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT));
	for (unsigned i = 0; i < impl->color.width * impl->color.height; i++)
		readback_result[i] = unpack(ptr[i]);
	bool res = stbi_write_png(path, impl->color.width, impl->color.height, 4, readback_result.data(), impl->color.width * 4);
	impl->device->unmap_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT);
	return res;
}

void RasterizerGPU::init(Device &device, bool subgroup, bool ubershader, bool async_compute, unsigned tile_size)
{
	impl->init(device, subgroup, ubershader, async_compute, tile_size);
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
