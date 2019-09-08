#pragma once

#include <stdint.h>
#include <stddef.h>
#include "primitive_setup.hpp"
#include "texture_format.hpp"
#include <memory>
#include "device.hpp"

namespace RetroWarp
{
enum class DepthTest : uint8_t
{
	Always = 0,
	LE = 1,
	LEQ = 2,
	GE = 3,
	GEQ = 4,
	EQ = 5,
	NEQ = 6,
	Never = 7
};

enum class DepthWrite : uint8_t
{
	Off = 0,
	On = 0x80
};

enum class BlendState : uint8_t
{
	Replace = 0,
	Additive = 1,
	Alpha = 2,
	Subtract = 3
};

class RasterizerGPU
{
public:
	RasterizerGPU();
	~RasterizerGPU();

	void init(Vulkan::Device &device, bool subgroup, bool ubershader, bool async_compute);

	void set_depth_state(DepthTest mode, DepthWrite write);
	void set_rop_state(BlendState state);

	void resize(unsigned width, unsigned height);
	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t rgba = 0);
	bool save_canvas(const char *path);

	void rasterize_primitives(const PrimitiveSetup *setup, size_t count);
	void set_texture(const Vulkan::ImageView &view);

	Vulkan::ImageHandle copy_to_framebuffer();

	void flush();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}
