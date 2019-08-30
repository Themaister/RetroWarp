#include "primitive_setup.hpp"
#include "rasterizer_cpu.hpp"
#include "triangle_converter.hpp"
#include "canvas.hpp"
#include "stb_image_write.h"
#include <stdio.h>
#include <vector>
#include <random>
#include <assert.h>

#include "global_managers.hpp"
#include "math.hpp"
#include "texture_files.hpp"
#include "gltf.hpp"
#include "camera.hpp"
#include "approximate_divider.hpp"
#include "rasterizer_gpu.hpp"
#include "os_filesystem.hpp"
#include "scene_loader.hpp"
#include "mesh_util.hpp"
#include "application.hpp"

using namespace RetroWarp;
using namespace Granite;

class StreamReader
{
public:
	StreamReader(const uint8_t *blob, size_t size);
	bool eof() const;
	bool parse_header();
	bool parse_resolution(uint32_t &width, uint32_t &height);
	bool parse_num_textures(uint32_t &count);

	enum Op { TEX, PRIM };
	bool parse_op(Op &op);
	bool parse_uint(uint32_t &value);
	bool parse_primitive(PrimitiveSetup &setup);

private:
	const uint8_t *blob;
	size_t offset;
	size_t size;
};

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		LOGE("Usage: dump-bench <path>\n");
		return EXIT_FAILURE;
	}

	Global::init();
	auto dump_file = Global::filesystem()->open(argv[1], FileMode::ReadOnly);
	if (!dump_file)
	{
		LOGE("Failed to open %s\n", argv[1]);
		return EXIT_FAILURE;
	}

	auto *mapped = static_cast<const uint8_t *>(dump_file->map());
	if (!mapped)
	{
		LOGE("Failed to map buffer.\n");
		return EXIT_FAILURE;
	}

	StreamReader reader(mapped, dump_file->get_size());

	if (!Vulkan::Context::init_loader(nullptr))
	{
		LOGE("Failed to init loader.\n");
		return EXIT_FAILURE;
	}

	Vulkan::Context ctx;
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0))
	{
		LOGE("Failed to create instance.\n");
		return EXIT_FAILURE;
	}

	Vulkan::Device device;
	device.set_context(ctx);

	RasterizerGPU rasterizer;
	rasterizer.init(device);

}
