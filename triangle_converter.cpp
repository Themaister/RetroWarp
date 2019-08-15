#include "triangle_converter.hpp"
#include <utility>
#include <cmath>
#include <algorithm>

namespace RetroWarp
{
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
	float rounded = std::round(z * float(((1 << 16) - 1) << 12));
	return int32_t(rounded);
}

static int32_t quantize_w(float w)
{
	float rounded = std::round(w * float(1 << 16));
	return int32_t(rounded);
}

static int32_t quantize_uv(float v)
{
	float rounded = std::round(v * float(1 << 13));
	//return clamp_float_int16(rounded);
	return int32_t(rounded);
}

static bool setup_triangle(PrimitiveSetup &setup, const InputPrimitive &input, CullMode cull_mode)
{
	setup = {};

	// Assume no clipping is required for now.
	const int16_t xs[] = { quantize_xy(input.vertices[0].x), quantize_xy(input.vertices[1].x), quantize_xy(input.vertices[2].x) };
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
	setup.dxdy_a = ((x_c - x_a) << 15) / std::max(1 << SUBPIXELS_LOG2, (y_hi - y_lo) >> 1);
	setup.dxdy_b = ((x_b - x_a) << 15) / std::max(1 << SUBPIXELS_LOG2, (y_mid - y_lo) >> 1);
	setup.dxdy_c = ((x_c - x_b) << 15) / std::max(1 << SUBPIXELS_LOG2, (y_hi - y_mid) >> 1);

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

static void interpolate_vertex(Vertex &v, const Vertex &a, const Vertex &b, float l)
{
	float left = 1.0f - l;
	float right = l;

	for (int i = 0; i < 4; i++)
	{
		v.clip[i] = a.clip[i] * left + b.clip[i] * right;
		v.color[i] = a.color[i] * left + b.color[i] * right;
	}

	v.u = a.u * left + b.u * right;
	v.v = a.v * left + b.v * right;
}

static unsigned get_clip_code_low(const InputPrimitive &prim, float limit, unsigned comp)
{
	bool clip_a = prim.vertices[0].clip[comp] < limit;
	bool clip_b = prim.vertices[1].clip[comp] < limit;
	bool clip_c = prim.vertices[2].clip[comp] < limit;
	unsigned clip_code = (unsigned(clip_a) << 0) | (unsigned(clip_b) << 1) | (unsigned(clip_c) << 2);
	return clip_code;
}

static unsigned get_clip_code_high(const InputPrimitive &prim, float limit, unsigned comp)
{
	bool clip_a = prim.vertices[0].clip[comp] > limit;
	bool clip_b = prim.vertices[1].clip[comp] > limit;
	bool clip_c = prim.vertices[2].clip[comp] > limit;
	unsigned clip_code = (unsigned(clip_a) << 0) | (unsigned(clip_b) << 1) | (unsigned(clip_c) << 2);
	return clip_code;
}

static void clip_single_output(InputPrimitive &output, const InputPrimitive &input, unsigned component, float target,
                               unsigned a, unsigned b, unsigned c)
{
	float interpolate_a = (target - input.vertices[a].clip[component]) /
	                      (input.vertices[c].clip[component] - input.vertices[a].clip[component]);
	float interpolate_b = (target - input.vertices[b].clip[component]) /
	                      (input.vertices[c].clip[component] - input.vertices[b].clip[component]);

	interpolate_vertex(output.vertices[a], input.vertices[a], input.vertices[c], interpolate_a);
	interpolate_vertex(output.vertices[b], input.vertices[b], input.vertices[c], interpolate_b);

	output.vertices[a].clip[component] = target;
	output.vertices[b].clip[component] = target;
	output.vertices[c] = input.vertices[c];
}

static void clip_dual_output(InputPrimitive *output, const InputPrimitive &input, unsigned component, float target,
                             unsigned a, unsigned b, unsigned c)
{
	float interpolate_ab = (target - input.vertices[a].clip[component]) /
	                       (input.vertices[b].clip[component] - input.vertices[a].clip[component]);
	float interpolate_ac = (target - input.vertices[a].clip[component]) /
	                       (input.vertices[c].clip[component] - input.vertices[a].clip[component]);

	Vertex ab, ac;
	interpolate_vertex(ab, input.vertices[a], input.vertices[b], interpolate_ab);
	interpolate_vertex(ac, input.vertices[a], input.vertices[c], interpolate_ac);

	ab.clip[component] = target;
	ac.clip[component] = target;

	output[0].vertices[0] = ab;
	output[0].vertices[1] = input.vertices[b];
	output[0].vertices[2] = ac;
	output[1].vertices[0] = ac;
	output[1].vertices[1] = input.vertices[b];
	output[1].vertices[2] = input.vertices[c];
}

static unsigned clip_component(InputPrimitive *prims, const InputPrimitive &prim, unsigned component,
                               float target, unsigned code)
{
	switch (code)
	{
	case 0:
		// Nothing to clip.
		prims[0] = prim;
		return 1;

	case 1:
		// Clip A.
		clip_dual_output(prims, prim, component, target, 0, 1, 2);
		return 2;

	case 2:
		// Clip B.
		clip_dual_output(prims, prim, component, target, 1, 2, 0);
		return 2;

	case 3:
		// Interpolate A and B against C.
		clip_single_output(prims[0], prim, component, target, 0, 1, 2);
		return 1;

	case 4:
		// Clip C.
		clip_dual_output(prims, prim, component, target, 2, 0, 1);
		return 2;

	case 5:
		// Interpolate A and C against B.
		clip_single_output(prims[0], prim, component, target, 2, 0, 1);
		return 1;

	case 6:
		// Interpolate B and C against A.
		clip_single_output(prims[0], prim, component, target, 1, 2, 0);
		return 1;

	case 7:
		// All clipped.
		return 0;

	default:
		return 0;
	}
}

static unsigned clip_triangles(InputPrimitive *outputs, const InputPrimitive *inputs, unsigned count, unsigned component, float target)
{
	unsigned output_count = 0;

	for (unsigned i = 0; i < count; i++)
	{
		unsigned clip_code;
		if (target > 0.0f)
			clip_code = get_clip_code_high(inputs[i], target, component);
		else
			clip_code = get_clip_code_low(inputs[i], target, component);

		unsigned clipped_count = clip_component(outputs, inputs[i], component, target, clip_code);
		output_count += clipped_count;
		outputs += clipped_count;
	}

	return output_count;
}

static unsigned setup_clipped_triangles_clipped_w(PrimitiveSetup *setup, InputPrimitive &prim, CullMode mode, const ViewportTransform &vp)
{
	InputPrimitive tmp_a[256];
	InputPrimitive tmp_b[256];

	for (unsigned i = 0; i < 3; i++)
	{
		float iw = 1.0f / prim.vertices[i].w;
		prim.vertices[i].x *= iw;
		prim.vertices[i].y *= iw;
		prim.vertices[i].z *= iw;
		prim.vertices[i].u *= iw;
		prim.vertices[i].v *= iw;
		prim.vertices[i].w = iw;

		// Apply viewport transform for X/Y.
		prim.vertices[i].x = vp.x + (0.5f * prim.vertices[i].x + 0.5f) * vp.width;
		prim.vertices[i].y = vp.y + (0.5f * prim.vertices[i].y + 0.5f) * vp.height;
	}

	// Clip -X on guard bard.
	unsigned count = clip_triangles(tmp_a, &prim, 1, 0, -2048.0f);
	// Clip +X on guard band.
	count = clip_triangles(tmp_b, tmp_a, count, 0, +2048.0f);
	// Clip -Y on guard band.
	count = clip_triangles(tmp_a, tmp_b, count, 1, -2048.0f);
	// Clip +Y on guard band.
	count = clip_triangles(tmp_b, tmp_a, count, 1, +2048.0f);
	// Clip near, before viewport transform.
	count = clip_triangles(tmp_a, tmp_b, count, 2, 0.0f);
	// Clip far, before viewport transform.
	count = clip_triangles(tmp_b, tmp_a, count, 2, +1.0f);

	unsigned output_count = 0;
	for (unsigned i = 0; i < count; i++)
	{
		auto &tmp_prim = tmp_b[i];
		for (unsigned j = 0; j < 3; j++)
		{
			// Apply viewport transform for Z.
			tmp_prim.vertices[j].z = vp.min_depth + tmp_prim.vertices[j].z * (vp.max_depth - vp.min_depth);
		}

		if (setup_triangle(setup[output_count], tmp_b[i], mode))
			output_count++;
	}

	return output_count;
}

unsigned setup_clipped_triangles(PrimitiveSetup *setup, const InputPrimitive &prim, CullMode mode, const ViewportTransform &vp)
{
	// Don't clip against 0, since we have no way to deal with infinities in the rasterizer.
	// W of 1.0 / 1024.0 is super close to eye anyways.
	static const float MIN_W = 1.0f / 1024.0f;

	unsigned clip_code_w = get_clip_code_low(prim, MIN_W, 3);
	InputPrimitive clipped_w[2];
	unsigned clipped_w_count = clip_component(clipped_w, prim, 3, MIN_W, clip_code_w);
	unsigned output_count = 0;

	for (unsigned i = 0; i < clipped_w_count; i++)
	{
		unsigned count = setup_clipped_triangles_clipped_w(setup, clipped_w[i], mode, vp);
		setup += count;
		output_count += count;
	}
	return output_count;
}
}
