#include "triangle_converter.hpp"
#include <utility>
#include <cmath>
#include <algorithm>

namespace RetroWarp
{
static int32_t quantize_x(float x)
{
	x *= float(1 << 19);
	return int32_t(std::round(x));
}

static int16_t quantize_y(float y)
{
	y *= float(1 << 3);
	return int16_t(std::round(y));
}

static int16_t clamp_float_int16(float v)
{
	if (v < float(-0x8000))
		return -0x8000;
	else if (v > 0x7fff)
		return 0x7fff;
	else
		return int16_t(v);
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

void setup_triangle(PrimitiveSetup &setup, const InputPrimitive &input)
{
	setup = {};

	// Assume no clipping is required for now.
	const int32_t xs[] = { quantize_x(input.vertices[0].x), quantize_x(input.vertices[1].x), quantize_x(input.vertices[2].x) };
	const int16_t ys[] = { quantize_y(input.vertices[0].y), quantize_y(input.vertices[1].y), quantize_y(input.vertices[2].y) };
	const int16_t xs_snapped[] = { int16_t(xs[0] >> 16), int16_t(xs[1] >> 16), int16_t(xs[2] >> 16) };

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

	int32_t x_a = xs[index_a];
	int32_t x_b = xs[index_b];
	int32_t x_c = xs[index_c];

	setup.x_a = x_a;
	setup.x_b = x_a;
	setup.x_c = x_b;

	setup.y_lo = y_lo;
	setup.y_mid = y_mid;
	setup.y_hi = y_hi;

	// Compute slopes.
	setup.dxdy_a = (x_c - x_a) / std::max(1, y_hi - y_lo);
	setup.dxdy_b = (x_b - x_a) / std::max(1, y_mid - y_lo);
	setup.dxdy_c = (x_c - x_b) / std::max(1, y_hi - y_mid);
	if (setup.dxdy_b < setup.dxdy_a)
		setup.flags |= PRIMITIVE_RIGHT_MAJOR_BIT;

	quantize_color(setup.color, input.vertices[index_a].color);

	// Compute interpolation derivatives.
	int ab_x = xs_snapped[1] - xs_snapped[0];
	int ab_y = ys[1] - ys[0];
	int bc_x = xs_snapped[2] - xs_snapped[1];
	int bc_y = ys[2] - ys[1];
	int ca_x = xs_snapped[0] - xs_snapped[2];
	int ca_y = ys[0] - ys[2];
	int signed_area = ab_x * bc_y - ab_y * bc_x;

	// Check if triangle is degenerate. Compute derivatives.
	if (signed_area == 0)
		return;

	float inv_signed_area = float(1 << 3) / float(signed_area);
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
	int x_subpel_offset = (x_a >> 16) & 7;
	int y_subpel_offset_lo = y_lo & 7;
	int y_subpel_offset_mid = y_mid & 7;

	// Adjust interpolants for sub-pixel precision.
	setup.x_a -= setup.dxdy_a * y_subpel_offset_lo;
	setup.x_b -= setup.dxdy_b * y_subpel_offset_lo;
	setup.x_c -= setup.dxdy_c * y_subpel_offset_mid;

	for (int c = 0; c < 4; c++)
		setup.color[c] -= (setup.dcolor_dx[c] >> 3) * x_subpel_offset + (setup.dcolor_dy[c] >> 3) * y_subpel_offset_lo;

	setup.z -= (setup.dzdx >> 3) * x_subpel_offset + (setup.dzdy >> 3) * y_subpel_offset_lo;
	setup.w -= (setup.dwdx >> 3) * x_subpel_offset + (setup.dwdy >> 3) * y_subpel_offset_lo;
	setup.u -= (setup.dudx >> 3) * x_subpel_offset + (setup.dudy >> 3) * y_subpel_offset_lo;
	setup.v -= (setup.dvdx >> 3) * x_subpel_offset + (setup.dvdy >> 3) * y_subpel_offset_lo;
}
}
