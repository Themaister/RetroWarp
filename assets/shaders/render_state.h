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
	uint8_t depth_state;
	uint8_t blend_state;
};

layout(std430, set = 0, binding = RENDER_STATE_BUFFER) uniform RenderStates
{
	RenderState render_states[MAX_RENDER_STATES];
};

#endif