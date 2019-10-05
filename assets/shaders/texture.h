#ifndef TEXTURE_H_
#define TEXTURE_H_

#include "render_state.h"
#include "pixel_conv.h"

// Trilinear mipmapped textures all done "by hand".
// Textures are referenced with "descriptors" which we parse by hand as well.
// Fairly trivial implementation.

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
const uint TEXTURE_FMT_FILTER_LINEAR_BIT = 0x80u;
const uint TEXTURE_FMT_FILTER_MIP_LINEAR_BIT = 0x40u;

int round_up_bits(int u, int subsample)
{
	return (u + ((1 << subsample) - 1)) >> subsample;
}

int round_down_bits(int u, int subsample)
{
	return u >> subsample;
}

int compute_offset(int x, int y, int blocks_x, int subsample)
{
	x = round_down_bits(x, subsample);
	int block_x = x >> 3;
	int block_y = y >> 3;
	int block = (block_y * blocks_x + block_x) * 64;
	int offset = block + (y & 7) * 8 + (x & 7);
	return offset;
}

uvec4 sample_texture_lod(uint variant_index, ivec2 base_uv, int lod, uint fmt)
{
	int tex_width = int(render_states[variant_index].texture_width);
	int mip_width = max(tex_width >> lod, 1);

	ivec4 tex_clamp = ivec4(render_states[variant_index].texture_clamp) >> lod;
	ivec2 tex_mask = ivec2(render_states[variant_index].texture_mask) >> lod;

	bool linear_filter = (fmt & TEXTURE_FMT_FILTER_LINEAR_BIT) != 0u;

	ivec2 uv = base_uv;
	uv >>= lod;
	if (linear_filter)
		uv -= 16;
	ivec2 wrap_uv = uv & 31;
	uv >>= 5;

	int subsample = int(fmt & 3u);
	mip_width = round_up_bits(mip_width, subsample);
	int blocks_x = (mip_width + 7) >> 3;

	ivec2 uv0 = clamp(uv, tex_clamp.xy, tex_clamp.zw) & tex_mask;
	int offset = render_states[variant_index].texture_offset[lod] >> 1;
	int offset0 = (offset + compute_offset(uv0.x, uv0.y, blocks_x, subsample)) & ((VRAM_SIZE >> 1) - 1);
	uint raw_sample0 = uint(vram_data[offset0]);

	uint raw_sample1, raw_sample2, raw_sample3;
	ivec2 uv1, uv2, uv3;
	if (linear_filter)
	{
		uv1 = clamp(uv + ivec2(1, 0), tex_clamp.xy, tex_clamp.zw) & tex_mask;
		uv2 = clamp(uv + ivec2(0, 1), tex_clamp.xy, tex_clamp.zw) & tex_mask;
		uv3 = clamp(uv + ivec2(1), tex_clamp.xy, tex_clamp.zw) & tex_mask;
		int offset1 = (offset + compute_offset(uv1.x, uv1.y, blocks_x, subsample)) & ((VRAM_SIZE >> 1) - 1);
		int offset2 = (offset + compute_offset(uv2.x, uv2.y, blocks_x, subsample)) & ((VRAM_SIZE >> 1) - 1);
		int offset3 = (offset + compute_offset(uv3.x, uv3.y, blocks_x, subsample)) & ((VRAM_SIZE >> 1) - 1);
		raw_sample1 = uint(vram_data[offset1]);
		raw_sample2 = uint(vram_data[offset2]);
		raw_sample3 = uint(vram_data[offset3]);
	}

	uvec4 sample0, sample1, sample2, sample3;

	switch (fmt & 0x3fu)
	{
	case TEXTURE_FMT_ARGB1555:
		sample0 = expand_argb1555(unpack_argb1555(raw_sample0));
		if (linear_filter)
		{
			sample1 = expand_argb1555(unpack_argb1555(raw_sample1));
			sample2 = expand_argb1555(unpack_argb1555(raw_sample2));
			sample3 = expand_argb1555(unpack_argb1555(raw_sample3));
		}
		break;

	case TEXTURE_FMT_LA88:
		sample0 = uvec4(uvec3(raw_sample0 & 0xffu), raw_sample0 >> 8u);
		if (linear_filter)
		{
			sample1 = uvec4(uvec3(raw_sample1 & 0xffu), raw_sample1 >> 8u);
			sample2 = uvec4(uvec3(raw_sample2 & 0xffu), raw_sample2 >> 8u);
			sample3 = uvec4(uvec3(raw_sample3 & 0xffu), raw_sample3 >> 8u);
		}
		break;

	case TEXTURE_FMT_I8:
		sample0 = uvec4((raw_sample0 >> (8u * (uv0.x & 1u))) & 0xffu);
		if (linear_filter)
		{
			sample1 = uvec4((raw_sample1 >> (8u * (uv1.x & 1u))) & 0xffu);
			sample2 = uvec4((raw_sample2 >> (8u * (uv2.x & 1u))) & 0xffu);
			sample3 = uvec4((raw_sample3 >> (8u * (uv3.x & 1u))) & 0xffu);
		}
		break;
	}

	if (linear_filter)
		return filter_bilinear(sample0, sample1, sample2, sample3, wrap_uv);
	else
		return sample0;
}

uvec4 sample_texture(uint variant_index, vec2 f_uv, float f_lod)
{
#if UBERSHADER
	uint fmt = uint(render_states[variant_index].texture_fmt);
	bool trilinear = (fmt & TEXTURE_FMT_FILTER_MIP_LINEAR_BIT) != 0u;
#else
	const uint fmt = (SHADER_VARIANT_MASK >> 16u) & 0xffu;
	const bool trilinear = (fmt & TEXTURE_FMT_FILTER_MIP_LINEAR_BIT) != 0u;
#endif
	if (!trilinear)
		f_lod += 0.5;

	int texture_max_lod = int(render_states[variant_index].texture_max_lod);
	int lod = int(round(256.0 * max(f_lod, 0.0)));
	int lod_frac = trilinear ? (lod & 0xff) : 0;

	int a_lod = clamp(lod >> 8, 0, texture_max_lod);

	ivec2 base_uv = ivec2(round(f_uv * 32.0));
	uvec4 sample_l0 = sample_texture_lod(variant_index, base_uv, a_lod, fmt);

#if !UBERSHADER
	if (trilinear)
#endif
	{
		if (lod_frac != 0)
		{
			int b_lod = clamp(a_lod + 1, 0, texture_max_lod);
			if (a_lod != b_lod)
			{
				uvec4 sample_l1 = sample_texture_lod(variant_index, base_uv, b_lod, fmt);
				sample_l0 = filter_trilinear(sample_l0, sample_l1, lod_frac);
			}
		}
	}
	return sample_l0;
}

#endif