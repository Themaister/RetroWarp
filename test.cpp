#include "primitive_setup.hpp"
#include "rasterizer_cpu.hpp"
#include "triangle_converter.hpp"
#include "canvas.hpp"
#include "stb_image_write.h"
#include <stdio.h>
#include <vector>
#include <random>
#include <assert.h>

using namespace RetroWarp;

struct CheckerboardSampler : Sampler
{
	Texel sample(int u, int v) override
	{
		u &= 1;
		v &= 1;
		uint8_t res = (u ^ v) * 255;
		return { res, res, res, res };
	}
};

struct CanvasROP : ROP
{
	bool save_canvas(const char *path) const;
	void fill_alpha_opaque();
	void emit_pixel(int x, int y, const Texel &texel) override;
	Canvas<uint32_t> canvas;
};

void CanvasROP::emit_pixel(int x, int y, const Texel &texel)
{
	assert(x >= 0 && y >= 0 && x < int(canvas.get_width()) && y < int(canvas.get_height()));
	auto &v = canvas.get(x, y);
#if 1
	if (v != 0)
	{
		fprintf(stderr, "Double write to (%d, %d)\n", x, y);
		abort();
	}
	v = ~0u;
#else
	v += 0x1f1f1f1f;
#endif
}

bool CanvasROP::save_canvas(const char *path) const
{
	const void *data = canvas.get_data();
	return stbi_write_png(path, canvas.get_width(), canvas.get_height(), 4, data, canvas.get_width() * 4);
}

void CanvasROP::fill_alpha_opaque()
{
	for (unsigned y = 0; y < canvas.get_height(); y++)
	{
		for (unsigned x = 0; x < canvas.get_width(); x++)
		{
			canvas.get(x, y) |= 0xff000000u;
		}
	}
}

int main()
{
	CheckerboardSampler samp;
	CanvasROP rop;
	RasterizerCPU rasterizer;
	rop.canvas.resize(256, 256);
	rasterizer.set_scissor(0, 0, 256, 256);
	rasterizer.set_sampler(&samp);
	rasterizer.set_rop(&rop);

	ViewportTransform vp = { 0.0f, 0.0f, 256.0f, 256.0f, 0.0f, 1.0f };

	InputPrimitive prim = {};
	prim.vertices[0].x = -0.5f;
	prim.vertices[0].y = -2.0f;
	prim.vertices[0].z = 0.0f;
	prim.vertices[0].w = 1.0f;

	prim.vertices[1].x = +0.5f;
	prim.vertices[1].y = -2.0f;
	prim.vertices[1].z = 0.0f;
	prim.vertices[1].w = 1.0f;

	prim.vertices[2].x = 0.0f;
	prim.vertices[2].y = 0.0f;
	prim.vertices[2].z = 0.0f;
	prim.vertices[2].w = 1.0f;

	PrimitiveSetup setup[256];

	unsigned count = setup_clipped_triangles(setup, prim, CullMode::None, vp);
	for (unsigned i = 0; i < count; i++)
		rasterizer.render_primitive(setup[i]);

	rop.fill_alpha_opaque();
	rop.save_canvas("/tmp/test.png");
}
