#ifndef FB_INFO_H_
#define FB_INFO_H_

layout(set = 2, binding = 0, std140) uniform FBInfo
{
	ivec4 scissor;
	uvec2 resolution;
	uvec2 resolution_tiles;
	int fb_stride;
	int primitive_count;
	int primitive_count_32;
	int primitive_count_1024;
} fb_info;

#endif