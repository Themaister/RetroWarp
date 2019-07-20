#include "triangle_converter.hpp"
#include <utility>
#include <cmath>
#include <algorithm>

namespace RetroWarp
{
enum { SUBPIXELS_LOG2 = 3 };

static int16_t clamp_float_int16(float v)
{
	if (v < float(-0x8000))
		return -0x8000;
	else if (v > 0x7fff)
		return 0x7fff;
	else
		return int16_t(v);
}

static int16_t quantize_xy(float x)
{
	x *= float(1 << SUBPIXELS_LOG2);
	return clamp_float_int16(std::round(x));
}

static void quantize_color(int16_t output[4], const float input[4])
{
	for (int i = 0; i < 4; i++)
	{
		float rounded = std::round(input[i] * 255.0f * 64.0f);
		output[i] = clamp_float_int16(rounded);
	}
}

static int32_t quantize_z(float z)
{
	float rounded = std::round(z * float(1 << 24));
	return int32_t(rounded);
}

static int32_t quantize_w(float w)
{
	float rounded = std::round(w * float(1 << 16));
	return int32_t(rounded);
}

static int16_t quantize_uv(float v)
{
	float rounded = std::round(v * float(1 << 8));
	return clamp_float_int16(rounded);
}

bool setup_triangle(PrimitiveSetup &setup, const InputPrimitive &input, CullMode cull_mode)
{
	setup = {};

	// Assume no clipping is required for now.
	const int32_t xs[] = { quantize_xy(input.vertices[0].x), quantize_xy(input.vertices[1].x), quantize_xy(input.vertices[2].x) };
	const int16_t ys[] = { quantize_xy(input.vertices[0].y), quantize_xy(input.vertices[1].y), quantize_xy(input.vertices[2].y) };

	int index_a = 0;
	int index_b = 1;
	int index_c = 2;

	// Sort primitives by height.
	if (ys[index_b] < ys[index_a])
		std::swap(index_b, index_a);
	if (ys[index_c] < ys[index_b])
		std::swap(index_c, index_b);
	if (ys[index_b] < ys[index_a])
		std::swap(index_b, index_a);

	int16_t y_lo = ys[index_a];
	int16_t y_mid = ys[index_b];
	int16_t y_hi = ys[index_c];

	int16_t x_a = xs[index_a];
	int16_t x_b = xs[index_b];
	int16_t x_c = xs[index_c];

	setup.x_a = x_a << 16;
	setup.x_b = x_a << 16;
	setup.x_c = x_b << 16;

	setup.y_lo = y_lo;
	setup.y_mid = y_mid;
	setup.y_hi = y_hi;

	// Compute slopes.
	// Can only shift by 15 here since subtraction adds another bit of range.
	setup.dxdy_a = ((x_c - x_a) << 15) / (std::max(1 << SUBPIXELS_LOG2, y_hi - y_lo) << 1);
	setup.dxdy_b = ((x_b - x_a) << 15) / (std::max(1 << SUBPIXELS_LOG2, y_mid - y_lo) << 1);
	setup.dxdy_c = ((x_c - x_b) << 15) / (std::max(1 << SUBPIXELS_LOG2, y_hi - y_mid) << 1);

	if (setup.dxdy_b < setup.dxdy_a)
		setup.flags |= PRIMITIVE_RIGHT_MAJOR_BIT;

	quantize_color(setup.color, input.vertices[index_a].color);

	// Compute interpolation derivatives.
	int ab_x = xs[1] - xs[0];
	int ab_y = ys[1] - ys[0];
	int bc_x = xs[2] - xs[1];
	int bc_y = ys[2] - ys[1];
	int ca_x = xs[0] - xs[2];
	int ca_y = ys[0] - ys[2];
	int signed_area = ab_x * bc_y - ab_y * bc_x;

	// Check if triangle is degenerate. Compute derivatives.
	if (signed_area == 0)
		return false;
	else if (cull_mode == CullMode::CCWOnly && signed_area > 0)
		return false;
	else if (cull_mode == CullMode::CWOnly && signed_area < 0)
		return false;

	float inv_signed_area = float(1 << SUBPIXELS_LOG2) / float(signed_area);
	float dcolor_dx[4];
	float dcolor_dy[4];

	for (int c = 0; c < 4; c++)
	{
		dcolor_dx[c] = -(ab_y * input.vertices[2].color[c] +
		                 ca_y * input.vertices[1].color[c] +
		                 bc_y * input.vertices[0].color[c]);

		dcolor_dy[c] = ab_x * input.vertices[2].color[c] +
		               ca_x * input.vertices[1].color[c] +
		               bc_x * input.vertices[0].color[c];

		dcolor_dx[c] *= inv_signed_area;
		dcolor_dy[c] *= inv_signed_area;
	}

	quantize_color(setup.dcolor_dx, dcolor_dx);
	quantize_color(setup.dcolor_dy, dcolor_dy);

	float dzdx = -inv_signed_area * (ab_y * input.vertices[2].z +
	                                 ca_y * input.vertices[1].z +
	                                 bc_y * input.vertices[0].z);
	float dzdy = inv_signed_area * (ab_x * input.vertices[2].z +
	                                ca_x * input.vertices[1].z +
	                                bc_x * input.vertices[0].z);

	float dwdx = -inv_signed_area * (ab_y * input.vertices[2].w +
	                                 ca_y * input.vertices[1].w +
	                                 bc_y * input.vertices[0].w);
	float dwdy = inv_signed_area * (ab_x * input.vertices[2].w +
	                                ca_x * input.vertices[1].w +
	                                bc_x * input.vertices[0].w);

	float dudx = -inv_signed_area * (ab_y * input.vertices[2].u +
	                                 ca_y * input.vertices[1].u +
	                                 bc_y * input.vertices[0].u);
	float dudy = inv_signed_area * (ab_x * input.vertices[2].u +
	                                ca_x * input.vertices[1].u +
	                                bc_x * input.vertices[0].u);

	float dvdx = -inv_signed_area * (ab_y * input.vertices[2].v +
	                                 ca_y * input.vertices[1].v +
	                                 bc_y * input.vertices[0].v);
	float dvdy = inv_signed_area * (ab_x * input.vertices[2].v +
	                                ca_x * input.vertices[1].v +
	                                bc_x * input.vertices[0].v);

	setup.z = quantize_z(input.vertices[index_a].z);
	setup.dzdx = quantize_z(dzdx);
	setup.dzdy = quantize_z(dzdy);

	setup.w = quantize_w(input.vertices[index_a].w);
	setup.dwdx = quantize_w(dwdx);
	setup.dwdy = quantize_w(dwdy);

	setup.u = quantize_uv(input.vertices[index_a].u);
	setup.dudx = quantize_uv(dudx);
	setup.dudy = quantize_uv(dudy);

	setup.v = quantize_uv(input.vertices[index_a].v);
	setup.dvdx = quantize_uv(dvdx);
	setup.dvdy = quantize_uv(dvdy);

	setup.flags |= PRIMITIVE_PERSPECTIVE_CORRECT_BIT;

	// Interpolations are based on the integer coordinate of the top vertex.
	int x_subpel_offset = x_a & ((1 << SUBPIXELS_LOG2) - 1);
	int y_subpel_offset_lo = y_lo & ((1 << SUBPIXELS_LOG2) - 1);
	int y_subpel_offset_mid = y_mid & ((1 << SUBPIXELS_LOG2) - 1);

	// Adjust interpolants for sub-pixel precision.
	setup.x_a -= setup.dxdy_a * y_subpel_offset_lo;
	setup.x_b -= setup.dxdy_b * y_subpel_offset_lo;
	setup.x_c -= setup.dxdy_c * y_subpel_offset_mid;

	for (int c = 0; c < 4; c++)
		setup.color[c] -= (setup.dcolor_dx[c] >> SUBPIXELS_LOG2) * x_subpel_offset + (setup.dcolor_dy[c] >> SUBPIXELS_LOG2) * y_subpel_offset_lo;

	setup.z -= (setup.dzdx >> SUBPIXELS_LOG2) * x_subpel_offset + (setup.dzdy >> SUBPIXELS_LOG2) * y_subpel_offset_lo;
	setup.w -= (setup.dwdx >> SUBPIXELS_LOG2) * x_subpel_offset + (setup.dwdy >> SUBPIXELS_LOG2) * y_subpel_offset_lo;
	setup.u -= (setup.dudx >> SUBPIXELS_LOG2) * x_subpel_offset + (setup.dudy >> SUBPIXELS_LOG2) * y_subpel_offset_lo;
	setup.v -= (setup.dvdx >> SUBPIXELS_LOG2) * x_subpel_offset + (setup.dvdy >> SUBPIXELS_LOG2) * y_subpel_offset_lo;

	return true;
}
}
