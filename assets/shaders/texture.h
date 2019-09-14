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

uvec4 filter_bilinear(uvec4 sample0, uvec4 sample1, uvec4 sample2, uvec4 sample3, ivec2 l)
{
	uvec4 tex_top = filter_horiz(sample0, sample1, l.x);
	uvec4 tex_bottom = filter_horiz(sample2, sample3, l.x);
	uvec4 tex = filter_vert(tex_top, tex_bottom, l.y);
	return tex;
}

uvec4 filter_trilinear(uvec4 a, uvec4 b, int l)
{
	uvec4 res = a * uint(256 - l) + b * uint(l);
	return (res + 0x80u) >> 8u;
}

const uint TEXTURE_FMT_ARGB1555 = 0;
const uint TEXTURE_FMT_I8 = 1;
const uint TEXTURE_FMT_LA88 = 4;

int round_up_bits(int u, int subsample)
{
	return (u + ((1 << subsample) - 1)) >> subsample;
}

int round_down_bits(int u, int subsample)
{
	return u >> subsample;
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

	uint fmt = uint(render_states[variant_index].texture_fmt);
	int subsample = int(fmt & 3u);
	mip_width = round_up_bits(mip_width, subsample);

	ivec2 uv0 = clamp(uv, tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv1 = clamp(uv + ivec2(1, 0), tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv2 = clamp(uv + ivec2(0, 1), tex_clamp.xy, tex_clamp.zw) & tex_mask;
	ivec2 uv3 = clamp(uv + ivec2(1), tex_clamp.xy, tex_clamp.zw) & tex_mask;

	int offset = render_states[variant_index].texture_offset[lod] >> 1;

	int offset0 = (offset + round_down_bits(uv0.x, subsample) + uv0.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset1 = (offset + round_down_bits(uv1.x, subsample) + uv1.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset2 = (offset + round_down_bits(uv2.x, subsample) + uv2.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	int offset3 = (offset + round_down_bits(uv3.x, subsample) + uv3.y * mip_width) & ((VRAM_SIZE >> 1) - 1);
	uint raw_sample0 = uint(vram_data[offset0]);
	uint raw_sample1 = uint(vram_data[offset1]);
	uint raw_sample2 = uint(vram_data[offset2]);
	uint raw_sample3 = uint(vram_data[offset3]);

	uvec4 sample0, sample1, sample2, sample3;

	switch (fmt)
	{
	case TEXTURE_FMT_ARGB1555:
		sample0 = expand_argb1555(unpack_argb1555(raw_sample0));
		sample1 = expand_argb1555(unpack_argb1555(raw_sample1));
		sample2 = expand_argb1555(unpack_argb1555(raw_sample2));
		sample3 = expand_argb1555(unpack_argb1555(raw_sample3));
		break;

	case TEXTURE_FMT_LA88:
		sample0 = uvec4(uvec3(raw_sample0 & 0xffu), raw_sample0 >> 8u);
		sample1 = uvec4(uvec3(raw_sample1 & 0xffu), raw_sample1 >> 8u);
		sample2 = uvec4(uvec3(raw_sample2 & 0xffu), raw_sample2 >> 8u);
		sample3 = uvec4(uvec3(raw_sample3 & 0xffu), raw_sample3 >> 8u);
		break;

	case TEXTURE_FMT_I8:
		sample0 = uvec4((raw_sample0 >> (8u * (uv0.x & 1u))) & 0xffu);
		sample1 = uvec4((raw_sample1 >> (8u * (uv1.x & 1u))) & 0xffu);
		sample2 = uvec4((raw_sample2 >> (8u * (uv2.x & 1u))) & 0xffu);
		sample3 = uvec4((raw_sample3 >> (8u * (uv3.x & 1u))) & 0xffu);
		break;
	}

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
	uvec4 sample_l1 = uvec4(0);
	if (lod_frac != 0)
	{
		sample_l1 = sample_texture_lod(variant_index, base_uv, b_lod);
		sample_l0 = filter_trilinear(sample_l0, sample_l1, lod_frac);
	}
	return sample_l0;
}

#endif