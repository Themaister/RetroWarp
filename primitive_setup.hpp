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
	int32_t u, v, z, w;
	int32_t dudx, dvdx, dzdx, dwdx;
	int32_t dudy, dvdy, dzdy, dwdy;

	int32_t color[4];
	int32_t dcolor_dx[4];
	int32_t dcolor_dy[4];

	int32_t x_a, x_b, x_c;
	int32_t dxdy_a, dxdy_b, dxdy_c;

	int16_t u_offset;
	int16_t v_offset;

	int16_t y_lo, y_mid, y_hi;
	uint16_t flags;
	uint32_t padding[3];
};

static_assert((sizeof(PrimitiveSetup) & 15) == 0, "PrimitiveSetup is not aligned to 16 bytes.");
}
