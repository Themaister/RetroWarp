#pragma once

#include "primitive_setup.hpp"

namespace RetroWarp
{
struct Vertex
{
	union
	{
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
		float clip[4];
	};

	float u;
	float v;
	float color[4];
};

struct InputPrimitive
{
	Vertex vertices[3];
	int16_t u_offset;
	int16_t v_offset;
};

enum class CullMode
{
	None,
	CCWOnly,
	CWOnly
};

struct ViewportTransform
{
	float x;
	float y;
	float width;
	float height;
	float min_depth;
	float max_depth;
};

unsigned setup_clipped_triangles(PrimitiveSetup prim[8], const InputPrimitive &input, CullMode mode, const ViewportTransform &vp);
}
