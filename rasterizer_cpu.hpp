#pragma once

#include "primitive_setup.hpp"
#include "canvas.hpp"

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

class RasterizerCPU
{
public:
	void resize(unsigned width, unsigned height);
	void render_primitive(const PrimitiveSetup &prim);

	Canvas<uint32_t> &get_canvas();
	const Canvas<uint32_t> &get_canvas() const;

	void set_scissor(int x, int y, int width, int height);

	void fill_alpha_opaque();

	bool save_canvas(const char *path) const;

	void set_sampler(Sampler *sampler);

private:
	Canvas<uint32_t> canvas;
	Sampler *sampler = nullptr;
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
