#include "approximate_divider.hpp"

#ifdef __GNUC__
#define leading_zeroes(x) __builtin_clz(x)
#elif defined(_MSC_VER)
static inline uint32_t leading_zeroes(uint32_t x)
{
	unsigned long result;
	if (_BitScanReverse(&result, x))
		return 31 - result;
	else
		return 32;
}
#else
#error "Implement me."
#endif

enum { INVERSE_BITS = 4 };
static int32_t inverse_table[1 << INVERSE_BITS + 1];

void setup_fixed_divider()
{
	for (unsigned i = 0; i <= 1 << INVERSE_BITS; i++)
		inverse_table[i] = int32_t(double(-0x400000) * 1.0 / (0.5 + (0.5 / (1 << INVERSE_BITS)) * double(i)));
}

int32_t fixed_divider(int32_t x, uint32_t y, unsigned extra_bits)
{
	unsigned leading = leading_zeroes(y);
	y <<= leading;
	y >>= (31 - INVERSE_BITS - 8);

	int rcp_frac = y & 0xff;
	y >>= 8;
	y &= (1 << INVERSE_BITS) - 1;

	int64_t rcp = inverse_table[y] * (0x100 - rcp_frac) + inverse_table[y + 1] * rcp_frac;
	int32_t res = -int32_t((int64_t(x) * rcp) >> (30 - extra_bits));

	unsigned msb_index = 32 - leading;
	res = (res + (1 << (msb_index - 1))) >> msb_index;
	return res;
}