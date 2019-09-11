#ifndef ROP_H_
#define ROP_H_

#include "render_state.h"
#include "pixel_conv.h"

uvec4 current_color;
uint current_z;
bool dirty_color = false;
bool dirty_depth = false;

#define ROP_Z_ALWAYS 0u
#define ROP_Z_LE 1u
#define ROP_Z_LEQ 2u
#define ROP_Z_GE 3u
#define ROP_Z_GEQ 4u
#define ROP_Z_EQ 5u
#define ROP_Z_NEQ 6u
#define ROP_Z_NEVER 7u

#define ROP_BLEND_REPLACE 0u
#define ROP_BLEND_ADDITIVE 1u
#define ROP_BLEND_ALPHA 2u
#define ROP_BLEND_SUBTRACT 3u

bool get_rop_dirty_color()
{
	return dirty_color;
}

bool get_rop_dirty_depth()
{
	return dirty_depth;
}

void set_initial_rop_color(uint color)
{
	current_color = unpack_argb1555(color);
}

void set_initial_rop_depth(uint z)
{
	current_z = z;
}

uint get_rop_state_variant(int primitive_index)
{
	return uint(render_state_indices[primitive_index]);
}

uvec3 lerp_unorm8(uvec3 a, uvec3 b, uint l)
{
	uvec3 res = a * (255u - l) + b * l;
	res += res >> 8u;
	return (res + 0x80u) >> 8u;
}

uvec4 blend_unorm(uvec4 src, uvec4 dst)
{
	uvec3 rgb = lerp_unorm8(dst.rgb, src.rgb, src.a);
	return uvec4(rgb, src.a);
}

const uint DITHER_LUT[16] = uint[](
	0, 4, 1, 5,
	6, 2, 7, 3,
	1, 5, 0, 4,
	7, 3, 6, 2);

uvec4 quantize_argb1555_dither(uvec4 color, int x, int y)
{
	int wrap_x = x & 3;
	int wrap_y = y & 3;
	int wrap_index = wrap_x + wrap_y * 4;
	return quantize_argb1555(uvec4(min(color.rgb + DITHER_LUT[wrap_index], uvec3(255)), color.a));
}

void rop_blend(uvec4 color, uint variant, int x, int y)
{
	uint blend_state = uint(render_states[variant].blend_state);
	switch (blend_state)
	{
	case ROP_BLEND_REPLACE:
		current_color = quantize_argb1555_dither(color, x, y);
		break;

	case ROP_BLEND_ADDITIVE:
		current_color = clamp(current_color + quantize_argb1555_dither(color, x, y), uvec4(0), uvec4(31, 31, 31, 1));
		break;

	case ROP_BLEND_SUBTRACT:
		current_color = uvec4(clamp(ivec4(current_color) - ivec4(quantize_argb1555_dither(color, x, y)), ivec4(0), ivec4(31, 31, 31, 1)));
		break;

	case ROP_BLEND_ALPHA:
		current_color = quantize_argb1555_dither(blend_unorm(color, expand_argb1555(current_color)), x, y);
		break;
	}

	dirty_color = true;
}

uint get_current_color()
{
	return pack_argb1555(current_color);
}

uint get_current_depth()
{
	return current_z;
}

bool rop_depth_test(uint z, uint variant)
{
	bool ret;
	uint depth_state = uint(render_states[variant].depth_state);
	uint z_test = depth_state & 7u;
	bool z_write = (depth_state & 0x80u) != 0;
	switch (z_test)
	{
	case ROP_Z_ALWAYS:
		ret = true;
		break;

	case ROP_Z_EQ:
		ret = current_z == z;
		break;

	case ROP_Z_NEQ:
		ret = current_z != z;
		break;

	case ROP_Z_LE:
		ret = z < current_z;
		break;

	case ROP_Z_LEQ:
		ret = z <= current_z;
		break;

	case ROP_Z_GE:
		ret = z > current_z;
		break;

	case ROP_Z_GEQ:
		ret = z >= current_z;
		break;

	default:
		ret = false;
		break;
	}

	if (ret && z_write)
	{
		current_z = z;
		dirty_depth = true;
	}

	return ret;
}

#endif