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
};

enum class CullMode
{
	None,
	CCWOnly,
	CWOnly
};

unsigned setup_clipped_triangles(PrimitiveSetup prim[8], const InputPrimitive &input, CullMode mode);
}
