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
	vec3 u;
	u8vec4 color_a;
	vec3 v;
	u8vec4 color_b;
	vec3 w;
	u8vec4 color_c;

	float z, dzdx, dzdy;
	float djdx, dkdx;
	float djdy, dkdy;

	i16vec2 uv_offset;
};

#endif