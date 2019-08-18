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
	info.size = count * sizeof(PrimitiveSetup);
	auto primitive_buffer = impl->device.create_buffer(info, setup);

	auto cmd = impl->device.request_command_buffer();
	cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	cmd->set_storage_buffer(0, 0, *primitive_buffer);
	cmd->set_storage_buffer(0, 1, *impl->color_buffer);
	cmd->set_storage_buffer(0, 2, *impl->depth_buffer);
	cmd->set_texture(1, 0, impl->image->get_view());

	Registers reg = {};
	reg.fb_stride = impl->width;
	reg.primitive_count = count;
	reg.resolution.x = impl->width;
	reg.resolution.y = impl->height;
	reg.scissor_x = 0;
	reg.scissor_y = 0;
	reg.scissor_width = impl->width;
	reg.scissor_height = impl->height;
	cmd->push_constants(&reg, 0, sizeof(reg));

	cmd->set_program("assets://shaders/rasterize.comp");
	cmd->dispatch((impl->width + 7) / 8, (impl->height + 7) / 8, 1);
	impl->device.submit(cmd);
	impl->device.next_frame_context();
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