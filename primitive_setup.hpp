#pragma once

#include <stdint.h>

namespace RetroWarp
{
enum { SUBPIXELS_LOG2 = 3 };

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

	int32_t u, dudx, dudy;
	int32_t v, dvdx, dvdy;
	int16_t u_offset;
	int16_t v_offset;

	int32_t dxdy_a, dxdy_b, dxdy_c;

	int32_t w, dwdx, dwdy;
	int32_t z, dzdx, dzdy;

	int32_t color[4];
	int32_t dcolor_dx[4];
	int32_t dcolor_dy[4];

	int16_t y_lo, y_mid, y_hi;
	uint16_t flags;
};
}
