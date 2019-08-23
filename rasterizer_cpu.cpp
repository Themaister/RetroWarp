#include "rasterizer_cpu.hpp"
#include "approximate_divider.hpp"
#include <utility>
#include <algorithm>
#include <assert.h>
#include <math.h>

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
	//z = (z + 0x80) >> 8;
	if (z < 0)
		return 0;
	else if (z > 0xffff)
		return 0xffff;
	else
		return uint16_t(z);
}

static int wrap_uv(int32_t coord)
{
	return int32_t(uint32_t(coord) << 11) >> 11;
}

void RasterizerCPU::render_primitive(const PrimitiveSetup &prim)
{
	// Interpolation of UV, Z, W and Color are all based off the floored integer coordinate.
	int interpolation_base_x = prim.pos.x_a >> 16;
	int interpolation_base_y = prim.pos.y_lo;

	int span_begin_y = (prim.pos.y_lo + ((1 << SUBPIXELS_LOG2) - 1)) >> SUBPIXELS_LOG2;
	int span_end_y = (prim.pos.y_hi - 1) >> SUBPIXELS_LOG2;

	// Scissor.
	if (span_begin_y < scissor.y)
		span_begin_y = scissor.y;
	if (span_end_y >= scissor.y + scissor.height)
		span_end_y = scissor.y + scissor.height - 1;

	for (int y = span_begin_y; y <= span_end_y; y++)
	{
		int y_sub = y << SUBPIXELS_LOG2;
		// Need to interpolate at high resolution,
		// since dxdy requires a very good resolution to resolve near vertical lines.
		int x_a = prim.pos.x_a + prim.pos.dxdy_a * (y_sub - prim.pos.y_lo);
		int x_b = prim.pos.x_b + prim.pos.dxdy_b * (y_sub - prim.pos.y_lo);
		int x_c = prim.pos.x_c + prim.pos.dxdy_c * (y_sub - prim.pos.y_mid);

		// The secondary span edge is split into two edges.
		bool select_hi = y_sub >= prim.pos.y_mid;
		int primary_x = x_a;
		int secondary_x = select_hi ? x_c : x_b;

		int start_x, end_x;
		constexpr int raster_rounding = (1 << (SUBPIXELS_LOG2 + 16)) - 1;

		if (prim.pos.flags & PRIMITIVE_RIGHT_MAJOR_BIT)
		{
			start_x = (secondary_x + raster_rounding) >> (16 + SUBPIXELS_LOG2);
			end_x = (primary_x - 1) >> (16 + SUBPIXELS_LOG2);
		}
		else
		{
			start_x = (primary_x + raster_rounding) >> (16 + SUBPIXELS_LOG2);
			end_x = (secondary_x - 1) >> (16 + SUBPIXELS_LOG2);
		}

		if (start_x < scissor.x)
			start_x = scissor.x;
		if (end_x >= scissor.x + scissor.width)
			end_x = scissor.x + scissor.width - 1;

		// We've passed the rasterization test. Interpolate colors, Z, 1/W.
		int dy = y_sub - interpolation_base_y;

		for (int x = start_x; x <= end_x; x++)
		{
			int dx = (x << SUBPIXELS_LOG2) - interpolation_base_x;

			//uint16_t z = clamp_unorm16(0xffff * (prim.attr.z + prim.attr.dzdx * dx + prim.attr.dzdy * dy));
			uint16_t z = clamp_unorm16(roundf(0xffff * (prim.attr.z + prim.attr.dzdx * dx + prim.attr.dzdy * dy)));

			float j = prim.attr.djdx * float(dx) + prim.attr.djdy * float(dy);
			float k = prim.attr.dkdx * float(dx) + prim.attr.dkdy * float(dy);
			float i = 1.0f - j - k;

			float r = float(prim.attr.color_a[0]) * i + float(prim.attr.color_b[0]) * j + float(prim.attr.color_c[0]) * k;
			float g = float(prim.attr.color_a[1]) * i + float(prim.attr.color_b[1]) * j + float(prim.attr.color_c[1]) * k;
			float b = float(prim.attr.color_a[2]) * i + float(prim.attr.color_b[2]) * j + float(prim.attr.color_c[2]) * k;
			float a = float(prim.attr.color_a[3]) * i + float(prim.attr.color_b[3]) * j + float(prim.attr.color_c[3]) * k;

			r = clamp_unorm8(int(roundf(r)));
			g = clamp_unorm8(int(roundf(g)));
			b = clamp_unorm8(int(roundf(b)));
			a = clamp_unorm8(int(roundf(a)));

			float u = prim.attr.u_a * i + prim.attr.u_b * j + prim.attr.u_c * k;
			float v = prim.attr.v_a * i + prim.attr.v_b * j + prim.attr.v_c * k;
			float w = prim.attr.w_a * i + prim.attr.w_b * j + prim.attr.w_c * k;
			w = std::max(0.0000001f, w);
			u /= w;
			v /= w;

			int perspective_u = int(roundf(u * 32.0f));
			int perspective_v = int(roundf(v * 32.0f));

			perspective_u -= 16;
			perspective_v -= 16;
			int sub_u = perspective_u & 31;
			int sub_v = perspective_v & 31;
			perspective_u >>= 5;
			perspective_v >>= 5;

			perspective_u += prim.attr.u_offset;
			perspective_v += prim.attr.v_offset;

			auto tex_00 = sampler->sample(perspective_u, perspective_v);
			auto tex_10 = sampler->sample(perspective_u + 1, perspective_v);
			auto tex_01 = sampler->sample(perspective_u, perspective_v + 1);
			auto tex_11 = sampler->sample(perspective_u + 1, perspective_v + 1);

			auto tex_0 = filter_linear_horiz(tex_00, tex_10, sub_u);
			auto tex_1 = filter_linear_horiz(tex_01, tex_11, sub_u);
			auto tex = filter_linear_vert(tex_0, tex_1, sub_v);

			tex = multiply_unorm8(tex, { uint8_t(r), uint8_t(g), uint8_t(b), uint8_t(a) });
			rop->emit_pixel(x, y, z, tex);
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
