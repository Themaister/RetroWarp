#pragma once

#include <stdint.h>
#include <stddef.h>
#include "primitive_setup.hpp"
#include "texture_format.hpp"
#include <memory>

namespace RetroWarp
{
class RasterizerGPU
{
public:
	RasterizerGPU();
	~RasterizerGPU();

	void resize(unsigned width, unsigned height);
	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t rgba = 0);
	bool save_canvas(const char *path);

	void rasterize_primitives(const PrimitiveSetup *setup, size_t count);
	void upload_texture(const Vulkan::TextureFormatLayout &layout);

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}