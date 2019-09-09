#ifndef RENDER_STATE_H_
#define RENDER_STATE_H_

#include "constants.h"

layout(std430, set = 0, binding = RENDER_STATE_INDEX_BUFFER) uniform ROPStateIndex
{
	uint16_t render_state_indices[MAX_PRIMITIVES];
};

struct RenderState
{
	i16vec4 scissor;
	u8vec4 constant_color;
	uint8_t depth_state;
	uint8_t blend_state;
	uint8_t combiner_state;
	uint8_t alpha_threshold;
};

layout(std430, set = 0, binding = RENDER_STATE_BUFFER) uniform RenderStates
{
	RenderState render_states[MAX_RENDER_STATES];
};

#endif