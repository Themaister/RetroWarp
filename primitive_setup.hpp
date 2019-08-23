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

struct PrimitiveSetupPos
{
	int32_t x_a, x_b, x_c;
	int32_t dxdy_a, dxdy_b, dxdy_c;
	int16_t y_lo, y_mid, y_hi;
	uint16_t flags;
};

struct PrimitiveSetupAttr
{
	float u_a, u_b, u_c;
	uint8_t color_a[4];
	float v_a, v_b, v_c;
	uint8_t color_b[4];
	float w_a, w_b, w_c;
	uint8_t color_c[4];

	float z, dzdx, dzdy;
	float djdx, dkdx;
	float djdy, dkdy;

	int16_t u_offset;
	int16_t v_offset;
};

struct PrimitiveSetup
{
	PrimitiveSetupPos pos;
	PrimitiveSetupAttr attr;
};

static_assert((sizeof(PrimitiveSetup) & 15) == 0, "PrimitiveSetup is not aligned to 16 bytes.");
}
