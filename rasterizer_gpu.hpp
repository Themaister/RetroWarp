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

enum CombinerState
{
	COMBINER_SAMPLE_BIT = 0x80,
	COMBINER_ADD_CONSTANT_BIT = 0x40,
	COMBINER_MODE_TEX_MOD_COLOR = 0,
	COMBINER_MODE_TEX = 1,
	COMBINER_MODE_COLOR = 2,
	COMBINER_MODE_MASK = 0x3f
};
using CombinerFlags = uint8_t;

class RasterizerGPU
{
public:
	RasterizerGPU();
	~RasterizerGPU();

	void init(Vulkan::Device &device, bool subgroup, bool ubershader, bool async_compute, unsigned tile_size);

	void set_depth_state(DepthTest mode, DepthWrite write);
	void set_rop_state(BlendState state);
	void set_scissor(int x, int y, int width, int height);
	void set_alpha_threshold(uint8_t threshold);
	void set_constant_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
	void set_combiner_mode(CombinerFlags flags);

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
