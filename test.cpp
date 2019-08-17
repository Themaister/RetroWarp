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

using namespace RetroWarp;
using namespace Granite;

struct TextureSampler : Sampler
{
	Texel sample(int u, int v) override
	{
		if (u < 0)
			u = 0;
		if (v < 0)
			v = 0;
		if (unsigned(u) >= tex.get_layout().get_width())
			u = tex.get_layout().get_width() - 1;
		if (unsigned(v) >= tex.get_layout().get_height())
			v = tex.get_layout().get_height() - 1;

		auto *res = tex.get_layout().data_2d<u8vec4>(u, v);
		return { res->x, res->y, res->z, res->w };
	}

	SceneFormats::MemoryMappedTexture tex;
};

struct CanvasROP : ROP
{
	bool save_canvas(const char *path) const;
	void fill_alpha_opaque();
	void emit_pixel(int x, int y, uint16_t z, const Texel &texel) override;
	Canvas<uint32_t> canvas;
	Canvas<uint16_t> depth_canvas;

	void clear_depth(uint16_t z = 0xffff);
};

void CanvasROP::clear_depth(uint16_t z)
{
	for (unsigned y = 0; y < depth_canvas.get_height(); y++)
		for (unsigned x = 0; x < depth_canvas.get_width(); x++)
			depth_canvas.get(x, y) = z;
}

void CanvasROP::emit_pixel(int x, int y, uint16_t z, const Texel &texel)
{
	assert(x >= 0 && y >= 0 && x < int(canvas.get_width()) && y < int(canvas.get_height()));
	auto &v = canvas.get(x, y);
	auto &d = depth_canvas.get(x, y);

	if (z < d)
	{
		d = z;
		v = (uint32_t(texel.r) << 0) | (uint32_t(texel.g) << 8) | (uint32_t(texel.b) << 16) | (uint32_t(texel.a) << 24);
	}
}

bool CanvasROP::save_canvas(const char *path) const
{
	const void *data = canvas.get_data();
	return stbi_write_png(path, canvas.get_width(), canvas.get_height(), 4, data, canvas.get_width() * 4);
}

void CanvasROP::fill_alpha_opaque()
{
	for (unsigned y = 0; y < canvas.get_height(); y++)
		for (unsigned x = 0; x < canvas.get_width(); x++)
			canvas.get(x, y) |= 0xff000000u;
}

static void test_divider()
{
	for (int32_t i = -0x10000; i <= 0x10000; i++)
	{
		int32_t res = fixed_divider(i, 1, 0);
		if (res != i)
			abort();
	}

	int32_t res = fixed_divider(4097 * 256, 1, 9);
	assert(res == (4097 * 256) << 9);
	res = fixed_divider(-4098 * 256, 1, 9);
	assert(res == (-4098 * 256) << 9);
}

int main(int argc, char **argv)
{
	setup_fixed_divider();
	test_divider();
	if (argc != 2)
		return EXIT_FAILURE;

	Global::init(Global::MANAGER_FEATURE_FILESYSTEM_BIT);

	GLTF::Parser parser(argv[1]);
	const SceneFormats::Mesh *mesh = nullptr;
	for (auto &node : parser.get_nodes())
	{
		if (!node.meshes.empty())
		{
			mesh = &parser.get_meshes()[node.meshes.front()];
			break;
		}
	}

	if (!mesh)
	{
		LOGE("No meshes.\n");
		return EXIT_FAILURE;
	}

	std::vector<Vertex> vertices;
	std::vector<InputPrimitive> input_primitives;

	unsigned num_vertices = mesh->positions.size() / mesh->position_stride;
	vertices.resize(num_vertices);
	auto pos_format = mesh->attribute_layout[Util::ecast(MeshAttribute::Position)].format;
	auto normal_format = mesh->attribute_layout[Util::ecast(MeshAttribute::Normal)].format;
	auto normal_offset = mesh->attribute_layout[Util::ecast(MeshAttribute::Normal)].offset;

	for (unsigned i = 0; i < num_vertices; i++)
	{
		if (pos_format == VK_FORMAT_R32G32B32_SFLOAT)
		{
			memcpy(vertices[i].clip, mesh->positions.data() + i * mesh->position_stride, 3 * sizeof(float));
			vertices[i].w = 1.0f;
		}
		else if (pos_format == VK_FORMAT_R32G32B32A32_SFLOAT)
			memcpy(vertices[i].clip, mesh->positions.data() + i * mesh->position_stride, 4 * sizeof(float));
		else
		{
			LOGE("Unknown position format.\n");
			return EXIT_FAILURE;
		}

		if (normal_format == VK_FORMAT_R32G32B32_SFLOAT)
		{
			vec3 n;
			memcpy(n.data, mesh->attributes.data() + i * mesh->attribute_stride + normal_offset, 3 * sizeof(float));
			float ndotl = clamp(dot(n, vec3(0.2f, 0.3f, 0.5f)) + 0.1f, 0.0f, 1.0f);
			for (auto &c : vertices[i].color)
				c = ndotl;
		}
		else
		{
			for (auto &c : vertices[i].color)
				c = 1.0f;
		}
	}

	auto &mat = parser.get_materials()[mesh->material_index];
	TextureSampler sampler;
	sampler.tex = load_texture_from_file(mat.base_color.path);

	if (mesh->attribute_layout[Util::ecast(MeshAttribute::UV)].format == VK_FORMAT_R32G32_SFLOAT)
	{
		auto offset = mesh->attribute_layout[Util::ecast(MeshAttribute::UV)].offset;
		for (unsigned i = 0; i < num_vertices; i++)
		{
			memcpy(&vertices[i].u, mesh->attributes.data() + i * mesh->attribute_stride + offset, sizeof(float));
			memcpy(&vertices[i].v, mesh->attributes.data() + i * mesh->attribute_stride + offset + sizeof(float), sizeof(float));
			vertices[i].u = vertices[i].u * float(sampler.tex.get_layout().get_width());
			vertices[i].v = vertices[i].v * float(sampler.tex.get_layout().get_height());
		}
	}

	Camera cam;
	cam.look_at(vec3(0.0f, 0.0f, 4.0f), vec3(0.0f));
	cam.set_fovy(0.3f * pi<float>());
	cam.set_depth_range(0.1f, 100.0f);
	cam.set_aspect(1280.0f / 720.0f);
	mat4 mvp = cam.get_projection() * cam.get_view();

	for (unsigned i = 0; i < num_vertices; i++)
	{
		vec4 v;
		memcpy(v.data, vertices[i].clip, sizeof(vec4));
		v = mvp * v;
		memcpy(vertices[i].clip, v.data, sizeof(vec4));
	}

	if (mesh->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
	{
		LOGE("Unsupported topology.\n");
		return EXIT_FAILURE;
	}

	if (!mesh->indices.empty())
	{
		if (mesh->index_type == VK_INDEX_TYPE_UINT16)
		{
			auto *indices = reinterpret_cast<const uint16_t *>(mesh->indices.data());
			for (unsigned i = 0; i < mesh->count; i += 3)
			{
				InputPrimitive prim;
				prim.vertices[0] = vertices[indices[i + 0]];
				prim.vertices[1] = vertices[indices[i + 1]];
				prim.vertices[2] = vertices[indices[i + 2]];
				input_primitives.push_back(prim);
			}
		}
		else if (mesh->index_type == VK_INDEX_TYPE_UINT32)
		{
			auto *indices = reinterpret_cast<const uint32_t *>(mesh->indices.data());
			for (unsigned i = 0; i < mesh->count; i += 3)
			{
				InputPrimitive prim;
				prim.vertices[0] = vertices[indices[i + 0]];
				prim.vertices[1] = vertices[indices[i + 1]];
				prim.vertices[2] = vertices[indices[i + 2]];
				input_primitives.push_back(prim);
			}
		}
	}
	else
	{
		for (unsigned i = 0; i < mesh->count; i += 3)
		{
			InputPrimitive prim;
			prim.vertices[0] = vertices[i + 0];
			prim.vertices[1] = vertices[i + 1];
			prim.vertices[2] = vertices[i + 2];
			input_primitives.push_back(prim);
		}
	}

	CanvasROP rop;
	RasterizerCPU rasterizer;
	rop.canvas.resize(1280, 720);
	rop.depth_canvas.resize(1280, 720);
	rasterizer.set_scissor(0, 0, 1280, 720);
	rasterizer.set_sampler(&sampler);
	rasterizer.set_rop(&rop);
	rop.clear_depth();

	ViewportTransform vp = { 0.0f, 0.0f, 1280.0f, 720.0f, 0.0f, 1.0f };
	PrimitiveSetup setup[256];

	for (auto &prim : input_primitives)
	{
		//unsigned prim_index = unsigned(&prim - input_primitives.data());
		unsigned count = setup_clipped_triangles(setup, prim, CullMode::CCWOnly, vp);
		for (unsigned i = 0; i < count; i++)
			rasterizer.render_primitive(setup[i]);
	}

	rop.fill_alpha_opaque();
	rop.save_canvas("/tmp/test.png");
}
