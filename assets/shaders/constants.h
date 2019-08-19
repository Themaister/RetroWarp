#ifndef CONSTANTS_H_
#define CONSTANTS_H_

const int SUBPIXELS_LOG2 = 3;
const int PRIMITIVE_RIGHT_MAJOR_BIT = (1 << 0);

const int MAX_PRIMITIVES = 0x4000;
const int TILE_BINNING_STRIDE = MAX_PRIMITIVES / 32;
const int MAX_WIDTH = 2048;
const int TILE_WIDTH = 16;
const int TILE_HEIGHT = 16;
const int MAX_TILES_X = MAX_WIDTH / TILE_WIDTH;
const int RASTER_ROUNDING = (1 << (SUBPIXELS_LOG2 + 16)) - 1;

#endif