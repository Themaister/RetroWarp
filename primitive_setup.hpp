#pragma once

#include <stdint.h>

namespace RetroWarp
{
enum PrimitiveFlagBits
{
	PRIMITIVE_RIGHT_MAJOR_BIT = 1 << 0,
	PRIMITIVE_PERSPECTIVE_CORRECT_BIT = 1 << 1,
	PRIMITIVE_FLAG_MAX_ENUM = 0x7fff
};

using PrimitiveFlags = uint16_t;

struct PrimitiveSetup
{
	int32_t x_a, x_b, x_c;
	int16_t dxdy_a, dxdy_b, dxdy_c;
	int16_t y_lo, y_mid, y_hi;
	int16_t dwdy;
	uint16_t flags;

	int32_t w, dwdx;
	int32_t z, dzdx;
	int16_t dzdy;

	uint16_t color[4];
	int16_t dcolor_dx[4];
	int16_t dcolor_dy[4];
	struct UV
	{
		int16_t u;
		int16_t v;
	};
	UV uv;
	UV duv_dx;
	UV duv_dy;
};
}