#include "primitive_setup.hpp"
#include "rasterizer_cpu.hpp"
#include "triangle_converter.hpp"

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

int main()
{
	CheckerboardSampler samp;
	RasterizerCPU rasterizer;
	rasterizer.resize(256, 256);
	rasterizer.set_scissor(0, 0, 256, 256);
	rasterizer.set_sampler(&samp);

	PrimitiveSetup setup;
	InputPrimitive prim = {};
	prim.vertices[0].x = 200.3f;
	prim.vertices[0].y = 100.4f;
	prim.vertices[1].x = 104.5f;
	prim.vertices[1].y = 100.6f;
	prim.vertices[2].x = 104.7f;
	prim.vertices[2].y = 201.8f;

	prim.vertices[0].z = 1.0f;
	prim.vertices[0].w = 1.0f;
	prim.vertices[1].z = 1.0f;
	prim.vertices[1].w = 1.0f;
	prim.vertices[2].z = 1.0f;
	prim.vertices[2].w = 1.0f;

	prim.vertices[0].u = 0.0f;
	prim.vertices[0].v = 0.0f;
	prim.vertices[1].u = 16.0f;
	prim.vertices[1].v = 0.0f;
	prim.vertices[2].u = 0.0f;
	prim.vertices[2].v = 16.0f;

	prim.vertices[0].color[0] = 1.0f;
	prim.vertices[1].color[1] = 1.0f;
	prim.vertices[2].color[2] = 1.0f;

	setup_triangle(setup, prim);

	rasterizer.render_primitive(setup);
	rasterizer.fill_alpha_opaque();
	rasterizer.save_canvas("/tmp/test.png");
}