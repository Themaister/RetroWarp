#include "rasterizer_cpu.hpp"

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
		int x_a = prim.x_a + prim.dxdy_a * (y - interpolation_base_ylo);
		int x_b = prim.x_b + prim.dxdy_b * (y - interpolation_base_ylo);
		int x_c = prim.x_c + prim.dxdy_c * (y - interpolation_base_ymid);

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

			#if 0
			int z = prim.z + prim.dzdx * dx + prim.dzdy * dy;
			int w = prim.w + prim.dwdx * dx + prim.dwdy * dy;

			int16_t u = prim.uv.u + prim.duv_dx.u * dx + prim.duv_dy.u * dy;
			int16_t v = prim.uv.v + prim.duv_dx.v * dx + prim.duv_dy.v * dy;
			#endif

			uint32_t argb = (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | (uint32_t(r) << 0);
			canvas.get(x, y) = argb;
		}
	}
}
}