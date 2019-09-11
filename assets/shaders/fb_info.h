#ifndef FB_INFO_H_
#define FB_INFO_H_

layout(set = 2, binding = 0, std140) uniform FBInfo
{
	uvec2 resolution;
	uvec2 resolution_tiles;
	int primitive_count;
	int primitive_count_32;
	int primitive_count_1024;

	int color_offset;
	int color_width;
	int color_height;
	int color_stride;
	int depth_offset;
	int depth_width;
	int depth_height;
	int depth_stride;
} fb_info;

#endif