#pragma once

#include "primitive_setup.hpp"

namespace RetroWarp
{
struct Vertex
{
	float x;
	float y;
	float z;
	float w;
	float uv[2];
	float color[4];
};

struct InputPrimitive
{
	Vertex vertices[3];
};

void setup_triangle(PrimitiveSetup &prim, const InputPrimitive &input);
}