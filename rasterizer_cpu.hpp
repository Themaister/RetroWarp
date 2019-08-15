#pragma once

#include "primitive_setup.hpp"

namespace RetroWarp
{
struct Texel
{
	uint8_t r, g, b, a;
};

struct Sampler
{
	virtual Texel sample(int u, int v) = 0;
};

struct ROP
{
	virtual void emit_pixel(int x, int y, uint16_t z, const Texel &texel) = 0;
};

class RasterizerCPU
{
public:
	void render_primitive(const PrimitiveSetup &prim);
	void set_scissor(int x, int y, int width, int height);
	void set_sampler(Sampler *sampler);
	void set_rop(ROP *rop);

private:
	Sampler *sampler = nullptr;
	ROP *rop = nullptr;

	struct
	{
		int x = 0;
		int y = 0;
		int width = 1;
		int height = 1;
	} scissor;

	struct FilteredTexel
	{
		uint16_t r, g, b, a;
	};
	static FilteredTexel filter_linear_horiz(const Texel &left, const Texel &right, int weight);
	static Texel filter_linear_vert(const FilteredTexel &top, const FilteredTexel &bottom, int weight);
	static Texel multiply_unorm8(const Texel &left, const Texel &right);
};
}
