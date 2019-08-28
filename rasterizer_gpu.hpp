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
	enum { NUM_STATE_INDICES = 32 };
	RasterizerGPU();
	~RasterizerGPU();

	void init(Vulkan::Device &device);

	void resize(unsigned width, unsigned height);
	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t rgba = 0);
	bool save_canvas(const char *path);

	void set_state_index(unsigned state_index);
	void rasterize_primitives(const PrimitiveSetup *setup, size_t count);
	void set_texture(unsigned state_index, const Vulkan::ImageView &view);

	float get_binning_ratio(size_t count);
	Vulkan::ImageHandle copy_to_framebuffer();

	void flush();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}
