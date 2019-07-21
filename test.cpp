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

	std::vector<Vertex> vertices(128 * 128);
	std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
	std::mt19937 rnd(42);

	for (int y = 0; y < 128; y++)
	{
		for (int x = 0; x < 128; x++)
		{
			auto &v = vertices[y * 128 + x];
			v.x = 2.0f * float(x) - 64.0f + dist(rnd);
			v.y = 2.0f * float(y) - 64.0f + dist(rnd);
		}
	}

	for (int y = 0; y < 127; y++)
	{
		for (int x = 0; x < 127; x++)
		{
			fprintf(stderr, "=== RENDER QUAD %d, %d ===\n", x, y);

			fprintf(stderr, " (%.3f, %.3f)\n",
			        vertices[128 * (y + 0) + (x + 0)].x,
			        vertices[128 * (y + 0) + (x + 0)].y);

			fprintf(stderr, " (%.3f, %.3f)\n",
			        vertices[128 * (y + 0) + (x + 1)].x,
			        vertices[128 * (y + 0) + (x + 1)].y);

			fprintf(stderr, " (%.3f, %.3f)\n",
			        vertices[128 * (y + 1) + (x + 0)].x,
			        vertices[128 * (y + 1) + (x + 0)].y);

			fprintf(stderr, " (%.3f, %.3f)\n",
			        vertices[128 * (y + 1) + (x + 1)].x,
			        vertices[128 * (y + 1) + (x + 1)].y);

			PrimitiveSetup setup[256];
			InputPrimitive prim;

			prim.vertices[0] = vertices[128 * (y + 0) + (x + 0)];
			prim.vertices[1] = vertices[128 * (y + 0) + (x + 1)];
			prim.vertices[2] = vertices[128 * (y + 1) + (x + 0)];
			unsigned count = setup_clipped_triangles(setup, prim, CullMode::CWOnly);
			for (unsigned i = 0; i < count; i++)
				rasterizer.render_primitive(setup[i]);

			prim.vertices[0] = vertices[128 * (y + 1) + (x + 1)];
			prim.vertices[1] = vertices[128 * (y + 1) + (x + 0)];
			prim.vertices[2] = vertices[128 * (y + 0) + (x + 1)];
			count = setup_clipped_triangles(setup, prim, CullMode::CWOnly);
			for (unsigned i = 0; i < count; i++)
				rasterizer.render_primitive(setup[i]);
			fprintf(stderr, "=== ===\n");
		}
	}

	rop.fill_alpha_opaque();
	rop.save_canvas("/tmp/test.png");
}
