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
#include "cli_parser.hpp"
#include "texture_utils.hpp"

constexpr int TEXTURE_BASE_LEVEL = 1;

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

	enum Op { TEX, PRIM, AlphaThreshold, BlendState, CombinerMode, ConstantColor, DepthTest, DepthWrite };
	bool parse_op(Op &op);
	bool parse_uint(uint32_t &value);
	bool parse_primitive(PrimitiveSetup &setup);

private:
	const uint8_t *blob;
	size_t offset = 0;
	size_t size;
};

StreamReader::StreamReader(const uint8_t *blob_, size_t size_)
	: blob(blob_), size(size_)
{
}

bool StreamReader::eof() const
{
	return offset == size;
}

bool StreamReader::parse_header()
{
	if (offset + 16 > size)
		return false;
	if (memcmp(blob + offset, "RETROWARP DUMP01", 16))
		return false;

	offset += 16;
	return true;
}

bool StreamReader::parse_resolution(uint32_t &width, uint32_t &height)
{
	if (offset + 2 * sizeof(uint32_t) > size)
		return false;

	memcpy(&width, blob + offset, sizeof(uint32_t));
	offset += sizeof(uint32_t);
	memcpy(&height, blob + offset, sizeof(uint32_t));
	offset += sizeof(uint32_t);
	return true;
}

bool StreamReader::parse_num_textures(uint32_t &count)
{
	return parse_uint(count);
}

bool StreamReader::parse_op(Op &op)
{
	if (offset + 4 > size)
		return false;

	if (memcmp(blob + offset, "TEX ", 4) == 0)
	{
		op = Op::TEX;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "PRIM", 4) == 0)
	{
		op = Op::PRIM;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "ATRS", 4) == 0)
	{
		op = Op::AlphaThreshold;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "BSTA", 4) == 0)
	{
		op = Op::BlendState;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "CMOD", 4) == 0)
	{
		op = Op::CombinerMode;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "CCOL", 4) == 0)
	{
		op = Op::ConstantColor;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "DTST", 4) == 0)
	{
		op = Op::DepthTest;
		offset += 4;
		return true;
	}
	else if (memcmp(blob + offset, "DWRT", 4) == 0)
	{
		op = Op::DepthWrite;
		offset += 4;
		return true;
	}
	else
		return false;
}

bool StreamReader::parse_uint(uint32_t &value)
{
	if (offset + sizeof(uint32_t) > size)
		return false;
	memcpy(&value, blob + offset, sizeof(uint32_t));
	offset += sizeof(uint32_t);
	return true;
}

bool StreamReader::parse_primitive(PrimitiveSetup &setup)
{
	if (offset + sizeof(setup) > size)
		return false;
	memcpy(&setup, blob + offset, sizeof(setup));
	offset += sizeof(setup);
	return true;
}

int main(int argc, char **argv)
{
	bool ubershader = false;
	bool subgroup = true;
	bool async_compute = false;
	std::string path;
	unsigned tile_size = 16;

	Util::CLICallbacks cbs;
	cbs.add("--ubershader", [&](Util::CLIParser &) { ubershader = true; });
	cbs.add("--nosubgroup", [&](Util::CLIParser &) { subgroup = false; });
	cbs.add("--async-compute", [&](Util::CLIParser &) { async_compute = true; });
	cbs.add("--tile-size", [&](Util::CLIParser &parser) { tile_size = parser.next_uint(); });
	cbs.default_handler = [&](const char *arg) { path = arg; };
	Util::CLIParser parser(std::move(cbs), argc - 1, argv + 1);

	if (!parser.parse() || path.empty())
	{
		LOGE("Failed to parse.\n");
		return EXIT_FAILURE;
	}

	if (tile_size & (tile_size - 1))
	{
		LOGE("Tile size must be POT.\n");
		return EXIT_FAILURE;
	}

	Global::init();
	Global::filesystem()->register_protocol("assets", std::make_unique<OSFilesystem>(ASSET_DIRECTORY));

	auto dump_file = Global::filesystem()->open(path, FileMode::ReadOnly);
	if (!dump_file)
	{
		LOGE("Failed to open %s\n", path.c_str());
		return EXIT_FAILURE;
	}

	auto *mapped = static_cast<const uint8_t *>(dump_file->map());
	if (!mapped)
	{
		LOGE("Failed to map buffer.\n");
		return EXIT_FAILURE;
	}

	StreamReader reader(mapped, dump_file->get_size());
	if (!reader.parse_header())
	{
		LOGE("Failed to parse header.\n");
		return EXIT_FAILURE;
	}

	uint32_t width, height;
	if (!reader.parse_resolution(width, height))
	{
		LOGE("Failed to parse resolution.\n");
		return EXIT_FAILURE;
	}

	uint32_t num_textures;
	if (!reader.parse_num_textures(num_textures))
	{
		LOGE("Failed to parse num textures.\n");
		return EXIT_FAILURE;
	}

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
	rasterizer.init(device, subgroup, ubershader, async_compute, tile_size);

	uint32_t addr = 0;
	rasterizer.set_color_framebuffer(addr, width, height, width * 2);
	addr += width * height * 2;
	rasterizer.set_depth_framebuffer(addr, width, height, width * 2);
	addr += width * height * 2;

	std::vector<TextureDescriptor> texture_descriptors;
	for (unsigned i = 0; i < num_textures; i++)
	{
		auto tex_path = std::string(argv[1]) + ".tex." + std::to_string(i);
		auto tex_file = load_texture_from_file(tex_path, ColorSpace::Linear);
		if (tex_file.empty())
		{
			LOGE("Failed to load texture.\n");
			return EXIT_FAILURE;
		}
		tex_file = SceneFormats::generate_mipmaps(tex_file.get_layout(), 0);
		auto &layout = tex_file.get_layout();
		unsigned levels = std::min(layout.get_levels() - TEXTURE_BASE_LEVEL, 8u);

		TextureDescriptor descriptor;

		descriptor.texture_clamp = i16vec4(-0x8000, -0x8000, 0x7fff, 0x7fff);
		descriptor.texture_mask = i16vec2(layout.get_width(TEXTURE_BASE_LEVEL) - 1,
		                                  layout.get_height(TEXTURE_BASE_LEVEL) - 1);
		descriptor.texture_max_lod = levels - 1;
		descriptor.texture_width = layout.get_width(TEXTURE_BASE_LEVEL);

		for (unsigned level = 0; level < levels; level++)
		{
			unsigned mip_width = layout.get_width(level + TEXTURE_BASE_LEVEL);
			unsigned mip_height = layout.get_height(level + TEXTURE_BASE_LEVEL);
			descriptor.texture_offset[level] = addr;
			rasterizer.copy_texture_rgba8888_to_vram(addr,
			                                         static_cast<const uint32_t *>(layout.data(0, level + TEXTURE_BASE_LEVEL)),
			                                         mip_width, mip_height, TextureFormat::ARGB1555);
			addr += mip_width * mip_height * 2;
		}

		texture_descriptors.push_back(descriptor);
	}

	struct Cache
	{
		unsigned state_index;
		uint8_t alpha_threshold;
		BlendState blend_state;
		PrimitiveSetup setup;
		CombinerFlags combiner_state;
		DepthTest depth_test;
		DepthWrite depth_write;
		uint8_t constant_color[4];
	};

	std::vector<Cache> commands;

	Cache current = {};
	current.blend_state = BlendState::Replace;
	current.combiner_state = COMBINER_MODE_TEX_MOD_COLOR | COMBINER_SAMPLE_BIT;
	current.depth_test = DepthTest::LE;
	current.depth_write = DepthWrite::On;

	while (!reader.eof())
	{
		StreamReader::Op op;
		if (!reader.parse_op(op))
		{
			LOGE("Failed to parse op.\n");
			return EXIT_FAILURE;
		}

		if (op == StreamReader::Op::TEX)
		{
			if (!reader.parse_uint(current.state_index))
			{
				LOGE("Failed to parse uint.\n");
				return EXIT_FAILURE;
			}
		}
		else if (op == StreamReader::Op::PRIM)
		{
			if (!reader.parse_primitive(current.setup))
			{
				LOGE("Failed to parse primitive.\n");
				return EXIT_FAILURE;
			}
			commands.push_back(current);
		}
		else if (op == StreamReader::Op::AlphaThreshold)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse alpha threshold.\n");
				return EXIT_FAILURE;
			}
			current.alpha_threshold = word;
		}
		else if (op == StreamReader::Op::BlendState)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse blend state.\n");
				return EXIT_FAILURE;
			}
			current.blend_state = BlendState(word);
		}
		else if (op == StreamReader::Op::CombinerMode)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse blend state.\n");
				return EXIT_FAILURE;
			}
			current.combiner_state = word;
		}
		else if (op == StreamReader::Op::ConstantColor)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse constant color.\n");
				return EXIT_FAILURE;
			}
			current.constant_color[0] = (word >> 0) & 0xff;
			current.constant_color[1] = (word >> 8) & 0xff;
			current.constant_color[2] = (word >> 16) & 0xff;
			current.constant_color[3] = (word >> 24) & 0xff;
		}
		else if (op == StreamReader::Op::DepthTest)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse depth test.\n");
				return EXIT_FAILURE;
			}
			current.depth_test = DepthTest(word);
		}
		else if (op == StreamReader::Op::DepthWrite)
		{
			uint32_t word;
			if (!reader.parse_uint(word))
			{
				LOGE("Failed to parse depth write.\n");
				return EXIT_FAILURE;
			}
			current.depth_write = DepthWrite(word);
		}
	}

	auto start_run = Util::get_current_time_nsecs();
	for (unsigned i = 0; i < 1000; i++)
	{
		device.next_frame_context();
		rasterizer.clear_depth();
		rasterizer.clear_color();
		for (auto &command : commands)
		{
			rasterizer.set_texture_descriptor(texture_descriptors[command.state_index]);
			rasterizer.set_combiner_mode(command.combiner_state);
			rasterizer.set_constant_color(command.constant_color[0], command.constant_color[1], command.constant_color[2], command.constant_color[3]);
			rasterizer.set_depth_state(command.depth_test, command.depth_write);
			rasterizer.set_alpha_threshold(command.alpha_threshold);
			rasterizer.set_rop_state(command.blend_state);
			rasterizer.set_depth_state(command.depth_test, command.depth_write);
			rasterizer.rasterize_primitives(&command.setup, 1);
		}
		rasterizer.flush();
	}
	device.wait_idle();
	auto end_run = Util::get_current_time_nsecs();
	LOGI("Total time: %.3f s\n", (end_run - start_run) * 1e-9);

	rasterizer.save_canvas("canvas.png");
}
