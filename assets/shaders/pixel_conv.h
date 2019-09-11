#ifndef PIXEL_CONV_H_
#define PIXEL_CONV_H_

uvec4 expand_argb1555(uvec4 color)
{
	return uvec4((color.rgb << 3u) | (color.rgb >> 2u), color.a * 0xffu);
}

uvec4 unpack_argb1555(uint color)
{
	uint r = (color >> 10u) & 31u;
	uint g = (color >> 5u) & 31u;
	uint b = (color >> 0u) & 31u;
	uint a = (color >> 15u) & 1u;
	return uvec4(r, g, b, a);
}

uint pack_argb1555(uvec4 color)
{
	return (color.r << 10u) | (color.g << 5u) | (color.b << 0u) | (color.a << 15u);
}

uvec4 quantize_argb1555(uvec4 color)
{
	return color >> uvec4(3u, 3u, 3u, 7u);
}

#endif