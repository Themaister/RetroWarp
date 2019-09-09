#ifndef RASTERIZER_HELPERS_H_
#define RASTERIZER_HELPERS_H_

#include "primitive_setup.h"
#include "fb_info.h"
#include "constants.h"

#ifdef PRIMITIVE_SETUP_POS_BUFFER
layout(std430, set = 0, binding = PRIMITIVE_SETUP_POS_BUFFER) readonly buffer TriangleSetupPos
{
	PrimitiveSetupPos primitives_pos[];
};
#endif

#ifdef PRIMITIVE_SETUP_ATTR_BUFFER
layout(std430, set = 0, binding = PRIMITIVE_SETUP_ATTR_BUFFER) readonly buffer TriangleSetupAttr
{
	PrimitiveSetupAttr primitives_attr[];
};
#endif

#ifdef PRIMITIVE_SETUP_POS_BUFFER
ivec2 get_interpolation_base(uint primitive_index)
{
	return ivec2(primitives_pos[primitive_index].x_a >> 16,
	             int(primitives_pos[primitive_index].y_lo));
}
#endif

#ifdef PRIMITIVE_SETUP_ATTR_BUFFER
vec3 interpolate_barycentrics(uint primitive_index, int x, int y, ivec2 interpolation_base)
{
	float dx = float((x << SUBPIXELS_LOG2) - interpolation_base.x);
	float dy = float((y << SUBPIXELS_LOG2) - interpolation_base.y);
	float j = primitives_attr[primitive_index].djdx * dx + primitives_attr[primitive_index].djdy * dy;
	float k = primitives_attr[primitive_index].dkdx * dx + primitives_attr[primitive_index].dkdy * dy;
	float i = 1.0 - j - k;
	return vec3(i, j, k);
}
#endif

#ifdef PRIMITIVE_SETUP_ATTR_BUFFER
vec2 interpolate_uv(uint primitive_index, vec3 bary)
{
	float u = dot(primitives_attr[primitive_index].u, bary);
	float v = dot(primitives_attr[primitive_index].v, bary);
	float w = dot(primitives_attr[primitive_index].w, bary);
	w = max(w, 0.00001);
	return (vec2(u, v) / w) + vec2(ivec2(primitives_attr[primitive_index].uv_offset));
}
#endif

#ifdef PRIMITIVE_SETUP_ATTR_BUFFER
vec4 interpolate_rgba(uint primitive_index, vec3 bary)
{
	vec4 rgba =
			vec4(uvec4(primitives_attr[primitive_index].color_a)) * bary.x +
			vec4(uvec4(primitives_attr[primitive_index].color_b)) * bary.y +
			vec4(uvec4(primitives_attr[primitive_index].color_c)) * bary.z;

	return clamp(rgba, vec4(0.0), vec4(255.0));
}
#endif

#ifdef PRIMITIVE_SETUP_POS_BUFFER
ivec2 compute_span_y(uint primitive_index)
{
	int span_begin_y = (int(primitives_pos[primitive_index].y_lo) + ((1 << SUBPIXELS_LOG2) - 1)) >> SUBPIXELS_LOG2;
	int span_end_y = (int(primitives_pos[primitive_index].y_hi) - 1) >> SUBPIXELS_LOG2;

	uint render_state_index = uint(render_state_indices[primitive_index]);
	ivec4 scissor = ivec4(render_states[render_state_index].scissor);
	span_begin_y = max(span_begin_y, scissor.y);
	span_end_y = min(span_end_y, scissor.y + scissor.w - 1);

	return ivec2(span_begin_y, span_end_y);
}

ivec2 interpolate_x(uint primitive_index, int y_sub)
{
    int x_a = primitives_pos[primitive_index].x_a +
              primitives_pos[primitive_index].dxdy_a * (y_sub - int(primitives_pos[primitive_index].y_lo));
    int x_b = primitives_pos[primitive_index].x_b +
              primitives_pos[primitive_index].dxdy_b * (y_sub - int(primitives_pos[primitive_index].y_lo));
    int x_c = primitives_pos[primitive_index].x_c +
              primitives_pos[primitive_index].dxdy_c * (y_sub - int(primitives_pos[primitive_index].y_mid));

    bool select_hi = y_sub >= int(primitives_pos[primitive_index].y_mid);
    int primary_x = x_a;
    int secondary_x = select_hi ? x_c : x_b;
    return ivec2(primary_x, secondary_x);
}

int min2x2(ivec2 a, ivec2 b)
{
    ivec2 c = min(a, b);
    return min(c.x, c.y);
}

int max2x2(ivec2 a, ivec2 b)
{
    ivec2 c = max(a, b);
    return max(c.x, c.y);
}

int min2(ivec2 a)
{
    return min(a.x, a.y);
}

int max2(ivec2 a)
{
    return max(a.x, a.y);
}

bool bin_primitive(uint primitive_index, ivec2 start, ivec2 end)
{
	uint render_state_index = uint(render_state_indices[primitive_index]);
	ivec4 scissor = ivec4(render_states[render_state_index].scissor);
	start = max(start, scissor.xy);
	end = min(end, scissor.xy + scissor.zw - 1);

    int start_y = start.y << SUBPIXELS_LOG2;
    int end_y = (end.y - 1) << SUBPIXELS_LOG2;
    // First, we clip start/end against y_lo, y_hi.
    start_y = max(start_y, int(primitives_pos[primitive_index].y_lo));
    end_y = min(end_y, int(primitives_pos[primitive_index].y_hi) - 1);

    // Y is clipped out, exit early.
    if (end_y < start_y)
        return false;

    // Evaluate span ranges at Y = start.y, Y = end.y, and Y = y_mid (if y_mid falls within the span range).
    ivec2 x_lo = interpolate_x(primitive_index, start_y);
    ivec2 x_hi = interpolate_x(primitive_index, end_y);
    int lo_x = min2x2(x_lo, x_hi);
    int hi_x = max2x2(x_lo, x_hi);
    int y_mid = int(primitives_pos[primitive_index].y_mid);
    if (y_mid > start_y && y_mid < end_y)
    {
        ivec2 x_mid = interpolate_x(primitive_index, y_mid);
        lo_x = min(lo_x, min2(x_mid));
        hi_x = max(hi_x, max2(x_mid));
    }

    // Snap min/max to grid.
    int start_x = (lo_x + RASTER_ROUNDING) >> (16 + SUBPIXELS_LOG2);
    int end_x = (hi_x - 1) >> (16 + SUBPIXELS_LOG2);

    // Clip start/end against raster region.
    start_x = max(start_x, start.x);
    end_x = min(end_x, end.x - 1);

    // If start_x <= end_x we will need to rasterize something.
    return start_x <= end_x;
}

ivec2 compute_span_x(uint primitive_index, int y)
{
	ivec2 xs = interpolate_x(primitive_index, y << SUBPIXELS_LOG2);
	if ((int(primitives_pos[primitive_index].flags) & PRIMITIVE_RIGHT_MAJOR_BIT) != 0)
		xs = xs.yx;

	int start_x = (xs.x + RASTER_ROUNDING) >> (16 + SUBPIXELS_LOG2);
	int end_x = (xs.y - 1) >> (16 + SUBPIXELS_LOG2);

	uint render_state_index = uint(render_state_indices[primitive_index]);
	ivec4 scissor = ivec4(render_states[render_state_index].scissor);
	start_x = max(start_x, scissor.x);
	end_x = min(end_x, scissor.x + scissor.z - 1);

	return ivec2(start_x, end_x);
}

bool test_coverage_single(uint primitive_index, int x, int y)
{
	ivec2 span_x = compute_span_x(primitive_index, y);
	ivec2 span_y = compute_span_y(primitive_index);

	bvec4 tests = bvec4(
			x >= span_x.x, x <= span_x.y,
			y >= span_y.x, y <= span_y.y);
	return all(tests);
}
#endif

#ifdef PRIMITIVE_SETUP_ATTR_BUFFER
uint interpolate_z(uint primitive_index, int x, int y, ivec2 interpolation_base)
{
	ivec2 d = (ivec2(x, y) << SUBPIXELS_LOG2) - interpolation_base;

	float fz = primitives_attr[primitive_index].z +
	           primitives_attr[primitive_index].dzdx * d.x +
	           primitives_attr[primitive_index].dzdy * d.y;
	uint z = uint(clamp(round(float(0xffff) * fz), 0.0, float(0xffff)));
	return z;
}
#endif

#endif