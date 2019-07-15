#pragma once

#include "primitive_setup.hpp"
#include "canvas.hpp"

namespace RetroWarp
{
class RasterizerCPU
{
public:
	void resize(unsigned width, unsigned height);
	void render_primitive(const PrimitiveSetup &prim);

	Canvas<uint32_t> &get_canvas();
	const Canvas<uint32_t> &get_canvas() const;

	void set_scissor(int x, int y, int width, int height);

private:
	Canvas<uint32_t> canvas;
	struct
	{
		int x = 0;
		int y = 0;
		int width = 1;
		int height = 1;
	} scissor;
};
}