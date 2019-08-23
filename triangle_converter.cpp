#include "triangle_converter.hpp"
#include <utility>
#include <algorithm>
#include <cmath>
#include <assert.h>

namespace RetroWarp
{
static int16_t clamp_float_int16(float v)
{
	if (v < float(-0x8000))
		return -0x8000;
	else if (v > float(0x7fff))
		return 0x7fff;
	else
		return int16_t(v);
}

static uint8_t clamp_float_unorm(float v)
{
	if (v < 0.0f)
		return 0;
	else if (v > 255.0f)
		return 255;
	else
		return int16_t(v);
}

static int16_t quantize_xy(float x)
{
	x *= float(1 << SUBPIXELS_LOG2);
	return clamp_float_int16(std::round(x));
}

static void quantize_color(uint8_t output[4], const float input[4])
{
	for (int i = 0; i < 4; i++)
	{
		float rounded = std::round(input[i] * 255.0f);
		output[i] = clamp_float_unorm(rounded);
	}
}

static int32_t quantize_z(float z)
{
	float rounded = std::round(z * float(((1 << 16) - 1) << 8));
	assert(rounded <= float(std::numeric_limits<int32_t>::max()));
	return int32_t(rounded);
}

static int32_t quantize_bary(float z)
{
	float rounded = std::round(z * float(1 << 16));
	assert(rounded <= float(std::numeric_limits<int32_t>::max()));
	return int32_t(rounded);
}

static int32_t quantize_w(float w)
{
	float rounded = std::round(w * float(1 << 16));
	assert(rounded <= float(std::numeric_limits<int32_t>::max()));
	return int32_t(rounded);
}

static int32_t quantize_uv(float v)
{
	float rounded = std::round(v * float(1 << 16));
	assert(rounded <= float(std::numeric_limits<int32_t>::max()));
	return int32_t(rounded);
}

static int32_t round_away_from_zero_divide(int32_t x, int32_t y)
{
	int32_t rounding = y - 1;
	if (x < 0)
		x -= rounding;
	else if (x > 0)
		x += rounding;

	return x / y;
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

	// Sort primitives by height, tie break by sorting on X.
	if (ys[index_b] < ys[index_a])
		std::swap(index_b, index_a);
	else if (ys[index_b] == ys[index_a] && xs[index_b] < xs[index_a])
		std::swap(index_b, index_a);

	if (ys[index_c] < ys[index_b])
		std::swap(index_c, index_b);
	else if (ys[index_c] == ys[index_b] && xs[index_c] < xs[index_b])
		std::swap(index_c, index_b);

	if (ys[index_b] < ys[index_a])
		std::swap(index_b, index_a);
	else if (ys[index_b] == ys[index_a] && xs[index_b] < xs[index_a])
		std::swap(index_b, index_a);

	int16_t y_lo = ys[index_a];
	int16_t y_mid = ys[index_b];
	int16_t y_hi = ys[index_c];

	int16_t x_a = xs[index_a];
	int16_t x_b = xs[index_b];
	int16_t x_c = xs[index_c];

	setup.pos.x_a = x_a << 16;
	setup.pos.x_b = x_a << 16;
	setup.pos.x_c = x_b << 16;

	setup.pos.y_lo = y_lo;
	setup.pos.y_mid = y_mid;
	setup.pos.y_hi = y_hi;

	// Compute slopes.
	setup.pos.dxdy_a = round_away_from_zero_divide((x_c - x_a) << 16, std::max(1, y_hi - y_lo));
	setup.pos.dxdy_b = round_away_from_zero_divide((x_b - x_a) << 16, std::max(1, y_mid - y_lo));
	setup.pos.dxdy_c = round_away_from_zero_divide((x_c - x_b) << 16, std::max(1, y_hi - y_mid));

	if (setup.pos.dxdy_b < setup.pos.dxdy_a)
		setup.pos.flags |= PRIMITIVE_RIGHT_MAJOR_BIT;

	// Compute winding before reorder.
	int ab_x = xs[1] - xs[0];
	int ab_y = ys[1] - ys[0];
	int bc_x = xs[2] - xs[1];
	int bc_y = ys[2] - ys[1];
	int ca_x = xs[0] - xs[2];
	int ca_y = ys[0] - ys[2];
	int signed_area = ab_x * bc_y - ab_y * bc_x;

	// Check if triangle is degenerate.
	if (signed_area == 0)
		return false;
	else if (cull_mode == CullMode::CCWOnly && signed_area > 0)
		return false;
	else if (cull_mode == CullMode::CWOnly && signed_area < 0)
		return false;

	// Recompute based on reordered vertices.
	ab_x = x_b - x_a;
	bc_x = x_c - x_b;
	ca_x = x_a - x_c;
	ab_y = y_mid - y_lo;
	bc_y = y_hi - y_mid;
	ca_y = y_lo - y_hi;
	signed_area = ab_x * bc_y - ab_y * bc_x;

	float inv_signed_area = 1.0f / float(signed_area);

	quantize_color(setup.attr.color_a, input.vertices[index_a].color);
	quantize_color(setup.attr.color_b, input.vertices[index_b].color);
	quantize_color(setup.attr.color_c, input.vertices[index_c].color);
	setup.attr.u_a = input.vertices[index_a].u;
	setup.attr.u_b = input.vertices[index_b].u;
	setup.attr.u_c = input.vertices[index_c].u;
	setup.attr.v_a = input.vertices[index_a].v;
	setup.attr.v_b = input.vertices[index_b].v;
	setup.attr.v_c = input.vertices[index_c].v;

	float dzdx = -inv_signed_area * (ab_y * input.vertices[index_c].z +
	                                 ca_y * input.vertices[index_b].z +
	                                 bc_y * input.vertices[index_a].z);
	float dzdy = inv_signed_area * (ab_x * input.vertices[index_c].z +
	                                ca_x * input.vertices[index_b].z +
	                                bc_x * input.vertices[index_a].z);

	float djdx = -inv_signed_area * ca_y;
	float djdy = inv_signed_area * ca_x;
	float dkdx = -inv_signed_area * ab_y;
	float dkdy = inv_signed_area * ab_x;

	setup.attr.z = input.vertices[index_a].z;
	setup.attr.dzdx = dzdx;
	setup.attr.dzdy = dzdy;

	setup.attr.djdx = djdx;
	setup.attr.djdy = djdy;
	setup.attr.dkdx = dkdx;
	setup.attr.dkdy = dkdy;

	setup.attr.w_a = input.vertices[index_a].w;
	setup.attr.w_b = input.vertices[index_b].w;
	setup.attr.w_c = input.vertices[index_c].w;

	setup.pos.flags |= PRIMITIVE_PERSPECTIVE_CORRECT_BIT;

	setup.attr.u_offset = input.u_offset;
	setup.attr.v_offset = input.v_offset;

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
	output.u_offset = input.u_offset;
	output.v_offset = input.v_offset;
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

	output[0].u_offset = input.u_offset;
	output[1].u_offset = input.u_offset;
	output[0].v_offset = input.v_offset;
	output[1].v_offset = input.v_offset;
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
	// Cull primitives on X/Y early.
	if (prim.vertices[0].x < -prim.vertices[0].w &&
	    prim.vertices[1].x < -prim.vertices[1].w &&
	    prim.vertices[2].x < -prim.vertices[2].w)
	{
		return 0;
	}
	else if (prim.vertices[0].y < -prim.vertices[0].w &&
	         prim.vertices[1].y < -prim.vertices[1].w &&
	         prim.vertices[2].y < -prim.vertices[2].w)
	{
		return 0;
	}
	else if (prim.vertices[0].x > prim.vertices[0].w &&
	         prim.vertices[1].x > prim.vertices[1].w &&
	         prim.vertices[2].x > prim.vertices[2].w)
	{
		return 0;
	}
	else if (prim.vertices[0].y > prim.vertices[0].w &&
	         prim.vertices[1].y > prim.vertices[1].w &&
	         prim.vertices[2].y > prim.vertices[2].w)
	{
		return 0;
	}

	InputPrimitive tmp_a[256];
	InputPrimitive tmp_b[256];

	const float ws[3] = {
		prim.vertices[0].w,
		prim.vertices[1].w,
		prim.vertices[2].w,
	};

	float min_w = std::numeric_limits<float>::max();
	for (auto w : ws)
		min_w = std::min(min_w, w);

#if 1
	// Try to center UV coordinates close to 0 for better division precision.
	float u_offset = floorf((1.0f / 3.0f) * (prim.vertices[0].u + prim.vertices[1].u + prim.vertices[2].u));
	float v_offset = floorf((1.0f / 3.0f) * (prim.vertices[0].v + prim.vertices[1].v + prim.vertices[2].v));
	prim.u_offset = int16_t(u_offset);
	prim.v_offset = int16_t(v_offset);
#else
	prim.u_offset = 0;
	prim.v_offset = 0;
#endif

	for (unsigned i = 0; i < 3; i++)
	{
		float iw = 1.0f / prim.vertices[i].w;
		prim.vertices[i].x *= iw;
		prim.vertices[i].y *= iw;
		prim.vertices[i].z *= iw;

		// Rescale inverse W for improved interpolation accuracy.
		// 1/w is now scaled to be maximum 1.
		iw *= min_w;
		prim.vertices[i].u = (prim.vertices[i].u - u_offset) * iw;
		prim.vertices[i].v = (prim.vertices[i].v - v_offset) * iw;
		prim.vertices[i].w = iw;

		// Apply viewport transform for X/Y.
		prim.vertices[i].x = vp.x + (0.5f * prim.vertices[i].x + 0.5f) * vp.width;
		prim.vertices[i].y = vp.y + (0.5f * prim.vertices[i].y + 0.5f) * vp.height;
	}

	// Clip -X on guard bard.
	unsigned count = clip_triangles(tmp_a, &prim, 1, 0, -2048.0f);
	// Clip +X on guard band.
	count = clip_triangles(tmp_b, tmp_a, count, 0, +2047.0f);
	// Clip -Y on guard band.
	count = clip_triangles(tmp_a, tmp_b, count, 1, -2048.0f);
	// Clip +Y on guard band.
	count = clip_triangles(tmp_b, tmp_a, count, 1, +2047.0f);
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
