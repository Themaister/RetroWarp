#ifndef DITHER_H_
#define DITHER_H_

// Classic 4x4 bayer dither.

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

#endif