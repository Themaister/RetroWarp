#ifndef PRIMITIVE_SETUP_H_
#define PRIMITIVE_SETUP_H_

struct PrimitiveSetupPos
{
	int x_a, x_b, x_c;
	int dxdy_a, dxdy_b, dxdy_c;
	int16_t y_lo, y_mid, y_hi, flags;
};

struct PrimitiveSetupAttr
{
	ivec4 uvzw;
	ivec4 duvzw_dx;
	ivec4 duvzw_dy;

	ivec4 color;
	ivec4 dcolor_dx;
	ivec4 dcolor_dy;

	i16vec2 uv_offset;
};

#endif