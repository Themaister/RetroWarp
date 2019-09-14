#pragma once

#include <stdint.h>
#include <stddef.h>
#include "primitive_setup.hpp"
#include "texture_format.hpp"
#include <memory>
#include "device.hpp"
#include "math.hpp"

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

enum class TextureFormat : uint8_t
{
	ARGB1555 = 0,
	I8 = 1,
	LA88 = 4
};

struct TextureDescriptor
{
	// 16 bytes.
	muglm::i16vec4 texture_clamp = muglm::i16vec4(-0x8000, -0x8000, 0x7fff, 0x7fff);
	muglm::i16vec2 texture_mask = muglm::i16vec2(255, 255);
	int16_t texture_width = 256;
	int8_t texture_max_lod = 7;
	TextureFormat texture_fmt = TextureFormat::ARGB1555;

	// 32 bytes.
	uint32_t texture_offset[8] = {};
};

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

	void set_color_framebuffer(unsigned offset, unsigned width, unsigned height, unsigned stride);
	void set_depth_framebuffer(unsigned offset, unsigned width, unsigned height, unsigned stride);

	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t rgba = 0);
	bool save_canvas(const char *path);

	void rasterize_primitives(const PrimitiveSetup *setup, size_t count);

	void set_texture_descriptor(const TextureDescriptor &desc);
	void copy_texture_rgba8888_to_vram(uint32_t offset, const uint32_t *src, unsigned width, unsigned height, TextureFormat fmt);

	Vulkan::ImageHandle copy_to_framebuffer();

	void flush();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}
