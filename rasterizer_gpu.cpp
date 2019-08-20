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

	BufferHandle binning_mask_buffer;
	BufferHandle binning_mask_buffer_coarse;
};

struct Registers
{
	uvec2 resolution;
	uint32_t primitive_count;
	uint32_t fb_stride;
	int32_t scissor_x, scissor_y, scissor_width, scissor_height;
};

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

	constexpr unsigned MAX_PRIMITIVES = 0x4000;
	constexpr unsigned MAX_WIDTH = 2048;
	constexpr unsigned MAX_HEIGHT = 2048;
	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	info.size = ((MAX_WIDTH + 15) / 16) * ((MAX_HEIGHT + 15) / 16) * (MAX_PRIMITIVES / 8);
	impl->binning_mask_buffer = impl->device.create_buffer(info);
	info.size = (info.size + 31) / 32;
	impl->binning_mask_buffer_coarse = impl->device.create_buffer(info);
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
	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_WRITE_BIT);
	cmd->fill_buffer(*impl->color_buffer, rgba);
	impl->device.submit(cmd);
	impl->device.next_frame_context();
}

void RasterizerGPU::rasterize_primitives(const RetroWarp::PrimitiveSetup *setup, size_t count)
{
	Vulkan::BufferCreateInfo info;
	info.domain = Vulkan::BufferDomain::Host;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	info.size = count * sizeof(PrimitiveSetupPos);
	auto primitive_buffer_pos = impl->device.create_buffer(info);
	info.size = count * sizeof(PrimitiveSetupAttr);
	auto primitive_buffer_attr = impl->device.create_buffer(info);

	auto *positions = static_cast<PrimitiveSetupPos *>(impl->device.map_host_buffer(*primitive_buffer_pos, MEMORY_ACCESS_WRITE_BIT));
	auto *attrs = static_cast<PrimitiveSetupAttr *>(impl->device.map_host_buffer(*primitive_buffer_attr, MEMORY_ACCESS_WRITE_BIT));
	for (size_t i = 0; i < count; i++)
	{
		positions[i] = setup[i].pos;
		attrs[i] = setup[i].attr;
	}
	impl->device.unmap_host_buffer(*primitive_buffer_pos, MEMORY_ACCESS_WRITE_BIT);
	impl->device.unmap_host_buffer(*primitive_buffer_attr, MEMORY_ACCESS_WRITE_BIT);

	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	Registers reg = {};
	reg.fb_stride = impl->width;
	reg.primitive_count = count;
	reg.resolution.x = impl->width;
	reg.resolution.y = impl->height;
	reg.scissor_x = 0;
	reg.scissor_y = 0;
	reg.scissor_width = impl->width;
	reg.scissor_height = impl->height;

	// Binning
	cmd->set_program("assets://shaders/binning.comp");
	cmd->set_storage_buffer(0, 0, *impl->binning_mask_buffer);
	cmd->set_storage_buffer(0, 1, *primitive_buffer_pos);
	cmd->push_constants(&reg, 0, sizeof(reg));
	cmd->dispatch((count + 63) / 64, (impl->width + 15) / 16, (impl->height + 15) / 16);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Merge coarse mask
	cmd->set_program("assets://shaders/build_coarse_mask.comp");
	cmd->set_storage_buffer(0, 0, *impl->binning_mask_buffer);
	cmd->set_storage_buffer(0, 1, *impl->binning_mask_buffer_coarse);
	uint32_t num_masks = (count + 31) / 32;
	cmd->push_constants(&num_masks, 0, sizeof(num_masks));
	cmd->dispatch((num_masks + 63) / 64, (impl->width + 15) / 16, (impl->height + 15) / 16);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	// Rasterization
	cmd->set_program("assets://shaders/rasterize.comp");
	cmd->set_storage_buffer(0, 0, *primitive_buffer_pos);
	cmd->set_storage_buffer(0, 1, *primitive_buffer_attr);
	cmd->set_storage_buffer(0, 2, *impl->color_buffer);
	cmd->set_storage_buffer(0, 3, *impl->depth_buffer);
	cmd->set_storage_buffer(0, 4, *impl->binning_mask_buffer);
	cmd->set_storage_buffer(0, 5, *impl->binning_mask_buffer_coarse);
	cmd->set_texture(1, 0, impl->image->get_view());

	cmd->push_constants(&reg, 0, sizeof(reg));
	cmd->dispatch((impl->width + 15) / 16, (impl->height + 15) / 16, 1);
	impl->device.submit(cmd);
	impl->device.next_frame_context();
}

float RasterizerGPU::get_binning_ratio(size_t count)
{
	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_READ_BIT);

	Vulkan::BufferCreateInfo info;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.domain = Vulkan::BufferDomain::CachedHost;
	info.size = impl->binning_mask_buffer->get_create_info().size;
	auto dst_buffer = impl->device.create_buffer(info);
	info.size = impl->binning_mask_buffer_coarse->get_create_info().size;
	auto dst_buffer_coarse = impl->device.create_buffer(info);
	cmd->copy_buffer(*dst_buffer, *impl->binning_mask_buffer);
	cmd->copy_buffer(*dst_buffer_coarse, *impl->binning_mask_buffer_coarse);
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	Fence fence;
	impl->device.submit(cmd, &fence);
	impl->device.next_frame_context();

	fence->wait();
	auto *ptr = static_cast<uint32_t *>(impl->device.map_host_buffer(*dst_buffer, MEMORY_ACCESS_READ_BIT));
	auto *ptr_coarse = static_cast<uint32_t *>(impl->device.map_host_buffer(*dst_buffer_coarse, MEMORY_ACCESS_READ_BIT));

	unsigned num_tiles_x = (impl->width + 15) / 16;
	unsigned num_tiles_y = (impl->height + 15) / 16;
	constexpr unsigned NUM_TILES_X = 2048 / 16;
	constexpr unsigned TILE_BINNING_STRIDE = 0x4000 / 32;
	constexpr unsigned TILE_BINNING_STRIDE_COARSE = TILE_BINNING_STRIDE / 32;

	unsigned max_count = 0;
	unsigned total_count = 0;
	for (unsigned y = 0; y < num_tiles_y; y++)
	{
		for (unsigned x = 0; x < num_tiles_x; x++)
		{
			auto *p = &ptr[TILE_BINNING_STRIDE * (y * NUM_TILES_X + x)];
			auto *p_coarse = &ptr_coarse[TILE_BINNING_STRIDE_COARSE * (y * NUM_TILES_X + x)];
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
	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT,
	             VK_ACCESS_TRANSFER_READ_BIT);

	Vulkan::BufferCreateInfo info;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.domain = Vulkan::BufferDomain::CachedHost;
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