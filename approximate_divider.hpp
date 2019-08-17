#pragma once

#include <stdint.h>

void setup_fixed_divider();
int32_t fixed_divider(int32_t x, uint32_t y, unsigned add_bits);
