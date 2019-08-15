#include "rasterizer_cpu.hpp"
#include <utility>
#include <algorithm>
#include <assert.h>
#include <stdio.h>

namespace RetroWarp
{
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

static uint16_t clamp_unorm16(int z)
{
	z = (z + 0x800) >> 12;
	if (z < 0)
		return 0;
	else if (z > 0xffff)
		return 0xffff;
	else
		return uint16_t(z);
}

void RasterizerCPU::render_primitive(const PrimitiveSetup &prim)
{
	fprintf(stderr, "=== START PRIMITIVE ===\n");

	// Interpolation of UV, Z, W and Color are all based off the floored integer coordinate.
	int interpolation_base_x = prim.x_a >> (16 + SUBPIXELS_LOG2);
	int interpolation_base_ylo = prim.y_lo >> SUBPIXELS_LOG2;
	int interpolation_base_ymid = prim.y_mid >> SUBPIXELS_LOG2;

	int span_begin_y = (prim.y_lo + ((1 << SUBPIXELS_LOG2) - 1)) >> SUBPIXELS_LOG2;
	int span_end_y = (prim.y_hi - 1) >> SUBPIXELS_LOG2;

	// Scissor.
	if (span_begin_y < scissor.y)
		span_begin_y = scissor.y;
	if (span_end_y >= scissor.y + scissor.height)
		span_end_y = scissor.y + scissor.height - 1;

	for (int y = span_begin_y; y <= span_end_y; y++)
	{
		// Need to interpolate at high resolution,
		// since dxdy requires a very good resolution to resolve near vertical lines.
		int x_a = prim.x_a + prim.dxdy_a * ((y - interpolation_base_ylo) << SUBPIXELS_LOG2);
		int x_b = prim.x_b + prim.dxdy_b * ((y - interpolation_base_ylo) << SUBPIXELS_LOG2);
		int x_c = prim.x_c + prim.dxdy_c * ((y - interpolation_base_ymid) << SUBPIXELS_LOG2);

		// The secondary span edge is split into two edges.
		bool select_hi = (y << SUBPIXELS_LOG2) >= prim.y_mid;
		int primary_x = x_a;
		int secondary_x = select_hi ? x_c : x_b;

		// Preserve 3 sub-pixels, X is now 13.3.
		primary_x >>= 16;
		secondary_x >>= 16;
		if (prim.flags & PRIMITIVE_RIGHT_MAJOR_BIT)
			std::swap(primary_x, secondary_x);

		// Compute the span for this scanline.
		int start_x = (primary_x + ((1 << SUBPIXELS_LOG2) - 1)) >> SUBPIXELS_LOG2;
		int end_x = (secondary_x - 1) >> SUBPIXELS_LOG2;

		fprintf(stderr, "  Y: %d: [%d, %d]\n", y, start_x, end_x);

		if (start_x < scissor.x)
			start_x = scissor.x;
		if (end_x >= scissor.x + scissor.width)
			end_x = scissor.x + scissor.width - 1;

		// We've passed the rasterization test. Interpolate colors, Z, 1/W.
		int dy = y - interpolation_base_ylo;

		for (int x = start_x; x <= end_x; x++)
		{
			int dx = x - interpolation_base_x;

			int r = int(prim.color[0]) + int(prim.dcolor_dx[0]) * dx + int(prim.dcolor_dy[0]) * dy;
			int g = int(prim.color[1]) + int(prim.dcolor_dx[1]) * dx + int(prim.dcolor_dy[1]) * dy;
			int b = int(prim.color[2]) + int(prim.dcolor_dx[2]) * dx + int(prim.dcolor_dy[2]) * dy;
			int a = int(prim.color[3]) + int(prim.dcolor_dx[3]) * dx + int(prim.dcolor_dy[3]) * dy;

			r = clamp_unorm8((r + 32) >> 6);
			g = clamp_unorm8((g + 32) >> 6);
			b = clamp_unorm8((b + 32) >> 6);
			a = clamp_unorm8((a + 32) >> 6);

			uint16_t z = clamp_unorm16(prim.z + prim.dzdx * dx + prim.dzdy * dy);
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
			rop->emit_pixel(x, y, z, tex);
		}
	}
	fprintf(stderr, "=== END PRIMITIVE ===\n\n");
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
	v = (v + 0x80) >> 8;
	assert(v <= 255 && v >= 0);
	return uint8_t(v);
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


void RasterizerCPU::set_sampler(Sampler *sampler_)
{
	sampler = sampler_;
}

void RasterizerCPU::set_rop(ROP *rop_)
{
	rop = rop_;
}
}
