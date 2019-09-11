#ifndef TEXTURE_H_
#define TEXTURE_H_

#include "render_state.h"
#include "pixel_conv.h"

uvec4 filter_horiz(uvec4 a, uvec4 b, int l)
{
	return a * (32u - uint(l)) + b * uint(l);
}

uvec4 filter_vert(uvec4 a, uvec4 b, int l)
{
	uvec4 ret = a * (32u - uint(l)) + b * uint(l);
	return (ret + 512u) >> 10u;
}

uvec4 filter_bilinear(uint sample0, uint sample1, uint sample2, uint sample3, ivec2 l)
{
	uvec4 tex_top = filter_horiz(expand_argb1555(unpack_argb1555(sample0)), expand_argb1555(unpack_argb1555(sample1)), l.x);
	uvec4 tex_bottom = filter_horiz(expand_argb1555(unpack_argb1555(sample2)), expand_argb1555(unpack_argb1555(sample3)), l.x);
	uvec4 tex = filter_vert(tex_top, tex_bottom, l.y);
	return tex;
}

uvec4 filter_trilinear(uvec4 a, uvec4 b, int l)
{
	uvec4 res = a * uint(256 - l) + b * uint(l);
	return (res + 0x80u) >> 8u;
}

uvec4 sample_texture_lod(uint variant_index, ivec2 base_uv, int lod)
{
	int tex_width = int(render_states[variant_index].texture_width);
	int mip_width = max(tex_width >> lod, 1);

	ivec4 tex_clamp = ivec4(render_states[variant_index].texture_clamp) >> lod;
	ivec2 tex_mask = ivec2(render_states[variant_index].texture_mask) >> lod;

	ivec2 uv = base_uv;
	uv >>= lod;
	uv -= 16;
	ivec2 wrap_uv = uv & 31;
	uv >>= 5;

	int offset = render_states[variant_index].texture_offset[lod] >> 1;

	ivec2 uv0 = clamp(uv, tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv1 = clamp(uv + ivec2(1, 0), tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv2 = clamp(uv + ivec2(0, 1), tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv3 = clamp(uv + ivec2(1), tex_clamp.xy, tex_clamp.zw) & tex_mask;

	int offset0 = (offset + uv0.x + uv0.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset1 = (offset + uv1.x + uv1.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset2 = (offset + uv2.x + uv2.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset3 = (offset + uv3.x + uv3.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	uint sample0 = uint(vram_data[offset0]);
	uint sample1 = uint(vram_data[offset1]);
	uint sample2 = uint(vram_data[offset2]);
	uint sample3 = uint(vram_data[offset3]);

	uvec4 tex = filter_bilinear(sample0, sample1, sample2, sample3, wrap_uv);
	return tex;
}

uvec4 sample_texture(uint variant_index, vec2 f_uv, float f_lod)
{
	int texture_max_lod = int(render_states[variant_index].texture_max_lod);
	int lod = int(round(256.0 * max(f_lod, 0.0)));
	int lod_frac = lod & 0xff;
	int a_lod = clamp(lod >> 8, 0, texture_max_lod);
	int b_lod = clamp((lod >> 8) + 1, 0, texture_max_lod);

	ivec2 base_uv = ivec2(round(f_uv * 32.0));
	uvec4 sample_l0 = sample_texture_lod(variant_index, base_uv, a_lod);
	uvec4 sample_l1 = sample_texture_lod(variant_index, base_uv, b_lod);
	return filter_trilinear(sample_l0, sample_l1, lod_frac);
}

#endif