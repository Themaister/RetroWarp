#include "rasterizer_cpu.hpp"
#include "stb_image_write.h"

namespace RetroWarp
{
void RasterizerCPU::resize(unsigned width, unsigned height)
{
	canvas.resize(width, height);
}

Canvas<uint32_t> &RasterizerCPU::get_canvas()
{
	return canvas;
}

const Canvas<uint32_t> &RasterizerCPU::get_canvas() const
{
	return canvas;
}

void RasterizerCPU::set_scissor(int x, int y, int width, int height)
{
	scissor.x = x;
	scissor.y = y;
	scissor.width = width;
	scissor.height = height;
}

static int clamp_unorm8(int v)
{
	if (v < 0)
		return 0;
	else if (v > 255)
		return 255;
	else
		return v;
}

void RasterizerCPU::render_primitive(const PrimitiveSetup &prim)
{
	// X coordinates are represented as 16.3(.13) signed fixed point.
	// Y coordinates are represented as 14.2 signed fixed point.

	// Interpolation of UV, Z, W and Color are all based off the floored integer coordinate.
	int interpolation_base_x = prim.x_a >> 16;
	int interpolation_base_ylo = prim.y_lo >> 2;
	int interpolation_base_ymid = prim.y_mid >> 2;

	int span_begin_y = (prim.y_lo + 3) >> 2;
	int span_end_y = (prim.y_hi - 1) >> 2;

	// Scissor.
	if (span_begin_y < scissor.y)
		span_begin_y = scissor.y;
	if (span_begin_y >= scissor.y + scissor.height)
		span_begin_y = scissor.y + scissor.height - 1;

	if (span_end_y < scissor.y)
		span_end_y = scissor.y;
	if (span_end_y >= scissor.y + scissor.height)
		span_end_y = scissor.y + scissor.height - 1;

	for (int y = span_begin_y; y <= span_end_y; y++)
	{
		// Need to interpolate at high resolution,
		// since dxdy requires a very good resolution to resolve near vertical lines.
		int x_a = prim.x_a + prim.dxdy_a * ((y - interpolation_base_ylo) << 2);
		int x_b = prim.x_b + prim.dxdy_b * ((y - interpolation_base_ylo) << 2);
		int x_c = prim.x_c + prim.dxdy_c * ((y - interpolation_base_ymid) << 2);

		// The secondary span edge is split into two edges.
		bool select_hi = (y << 2) >= prim.y_mid;
		int primary_x = x_a;
		int secondary_x = select_hi ? x_c : x_b;

		// Preserve 3 sub-pixels, X is now 16.3.
		primary_x >>= 13;
		secondary_x >>= 13;
		if (prim.flags & PRIMITIVE_RIGHT_MAJOR_BIT)
			std::swap(primary_x, secondary_x);

		// Compute the span for this scanline.
		int start_x = (primary_x + 7) >> 3;
		int end_x = (secondary_x - 1) >> 3;
		if (start_x >= end_x)
			continue;

		if (start_x < scissor.x)
			start_x = scissor.x;
		if (end_x >= scissor.x + scissor.width)
			end_x = scissor.x + scissor.width - 1;

		// We've passed the rasterization test. Interpolate colors, Z, 1/W.
		int dy = y - interpolation_base_ylo;
		for (int x = start_x; x <= end_x; x++)
		{
			int dx = x - interpolation_base_x;

			int r = int(prim.color[0]) + (int(prim.dcolor_dx[0]) << 1) * dx + (int(prim.dcolor_dy[0]) << 1) * dy;
			int g = int(prim.color[1]) + (int(prim.dcolor_dx[1]) << 1) * dx + (int(prim.dcolor_dy[1]) << 1) * dy;
			int b = int(prim.color[2]) + (int(prim.dcolor_dx[2]) << 1) * dx + (int(prim.dcolor_dy[2]) << 1) * dy;
			int a = int(prim.color[3]) + (int(prim.dcolor_dx[3]) << 1) * dx + (int(prim.dcolor_dy[3]) << 1) * dy;

			r = clamp_unorm8((r + 64) >> 8);
			g = clamp_unorm8((g + 64) >> 8);
			b = clamp_unorm8((b + 64) >> 8);
			a = clamp_unorm8((a + 64) >> 8);

			int z = prim.z + prim.dzdx * dx + prim.dzdy * dy;
			int w = prim.w + prim.dwdx * dx + prim.dwdy * dy;
			int u = prim.u + prim.dudx * dx + prim.dudy * dy;
			int v = prim.v + prim.dvdx * dx + prim.dvdy * dy;

			w = std::max(1, w);
			u = (u << 13) / w;
			v = (v << 13) / w;

			int sub_u = u & 31;
			int sub_v = v & 31;
			u >>= 5;
			v >>= 5;

			auto tex_00 = sampler->sample(u, v);
			auto tex_10 = sampler->sample(u + 1, v);
			auto tex_01 = sampler->sample(u, v + 1);
			auto tex_11 = sampler->sample(u + 1, v + 1);

			auto tex_0 = filter_linear_horiz(tex_00, tex_10, sub_u);
			auto tex_1 = filter_linear_horiz(tex_01, tex_11, sub_u);
			auto tex = filter_linear_vert(tex_0, tex_1, sub_v);

			tex = multiply_unorm8(tex, { uint8_t(r), uint8_t(g), uint8_t(b), uint8_t(a) });

			uint32_t argb = (uint32_t(tex.a) << 24) | (uint32_t(tex.b) << 16) | (uint32_t(tex.g) << 8) | (uint32_t(tex.r) << 0);
			canvas.get(x, y) = argb;
		}
	}
}

RasterizerCPU::FilteredTexel RasterizerCPU::filter_linear_horiz(const Texel &left, const Texel &right, int weight)
{
	int l = 32 - weight;
	int r = weight;
	return {
		uint16_t(left.r * l + right.r * r),
		uint16_t(left.g * l + right.g * r),
		uint16_t(left.b * l + right.b * r),
		uint16_t(left.a * l + right.a * r),
	};
}

Texel RasterizerCPU::filter_linear_vert(const RasterizerCPU::FilteredTexel &top,
                                        const RasterizerCPU::FilteredTexel &bottom, int weight)
{
	int t = 32 - weight;
	int b = weight;
	return {
		uint8_t((top.r * t + bottom.r * b + 512) >> 10),
		uint8_t((top.g * t + bottom.g * b + 512) >> 10),
		uint8_t((top.b * t + bottom.b * b + 512) >> 10),
		uint8_t((top.a * t + bottom.a * b + 512) >> 10),
	};
}

static uint8_t multiply_unorm8_component(uint8_t a, uint8_t b)
{
	int v = a * b;
	v += (v >> 8);
	return uint8_t((v + 0x80) >> 8);
}

Texel RasterizerCPU::multiply_unorm8(const Texel &left, const Texel &right)
{
	return {
		multiply_unorm8_component(left.r, right.r),
		multiply_unorm8_component(left.g, right.g),
		multiply_unorm8_component(left.b, right.b),
		multiply_unorm8_component(left.a, right.a),
	};
}

void RasterizerCPU::fill_alpha_opaque()
{
	for (unsigned y = 0; y < canvas.get_height(); y++)
	{
		for (unsigned x = 0; x < canvas.get_width(); x++)
		{
			canvas.get(x, y) |= 0xff000000u;
		}
	}
}

void RasterizerCPU::set_sampler(Sampler *sampler_)
{
	sampler = sampler_;
}

bool RasterizerCPU::save_canvas(const char *path) const
{
	const void *data = canvas.get_data();
	return stbi_write_png(path, canvas.get_width(), canvas.get_height(), 4, data, canvas.get_width() * 4);
}
}