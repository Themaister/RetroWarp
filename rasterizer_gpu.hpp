#pragma once

#include <stdint.h>
#include <stddef.h>
#include "primitive_setup.hpp"
#include "texture_format.hpp"
#include <memory>
#include "device.hpp"

namespace RetroWarp
{
class RasterizerGPU
{
public:
	RasterizerGPU();
	~RasterizerGPU();

	void init(Vulkan::Device &device);

	void resize(unsigned width, unsigned height);
	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t rgba = 0);
	bool save_canvas(const char *path);

	void rasterize_primitives(const PrimitiveSetup *setup, size_t count);
	void set_texture(const Vulkan::ImageView &view);

	float get_binning_ratio(size_t count);
	Vulkan::ImageHandle copy_to_framebuffer();

	void flush();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}
