#ifndef COMBINER_H_
#define COMBINER_H_

#define COMBINER_SAMPLE_BIT (0x80u)
#define COMBINER_ADD_CONSTANT_BIT (0x40u)
#define COMBINER_MODE_TEX_MOD_COLOR 0u
#define COMBINER_MODE_TEX 1u
#define COMBINER_MODE_COLOR 2u
#define COMBINER_MODE_MASK 0x3fu

uvec4 mul_unorm8(uvec4 a, uvec4 b)
{
	uvec4 res = a * b;
	res += res >> 8u;
	return (res + 0x80u) >> 8u;
}

uvec4 combine_result(uvec4 tex, uvec4 color, uvec4 constant_color, uint opts)
{
	uvec4 res;
	switch (opts & COMBINER_MODE_MASK)
	{
	case COMBINER_MODE_TEX_MOD_COLOR:
		res = mul_unorm8(tex, color);
		break;

	case COMBINER_MODE_TEX:
		res = tex;
		break;

	case COMBINER_MODE_COLOR:
		res = color;
		break;

	default:
		res = uvec4(0);
		break;
	}

	if ((opts & COMBINER_ADD_CONSTANT_BIT) != 0u)
		res = clamp(res + constant_color, uvec4(0), uvec4(255));

	return res;
}

#endif