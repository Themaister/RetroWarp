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

struct TextureSampler : Sampler
{
	Texel sample(int u, int v) override
	{
		if (u < 0)
			u = 0;
		if (v < 0)
			v = 0;
		if (unsigned(u) >= layout->get_width())
			u = layout->get_width() - 1;
		if (unsigned(v) >= layout->get_height())
			v = layout->get_height() - 1;

		auto *res = layout->data_2d<u8vec4>(u, v);
		return { res->x, res->y, res->z, res->w };
	}

	const Vulkan::TextureFormatLayout *layout = nullptr;
};

struct CanvasROP : ROP
{
	bool save_canvas(const char *path) const;
	void fill_alpha_opaque();
	void emit_pixel(int x, int y, uint16_t z, const Texel &texel) override;
	Canvas<uint32_t> canvas;
	Canvas<uint16_t> depth_canvas;

	void clear_depth(uint16_t z = 0xffff);
	void clear_color(uint32_t c = 0);
};

void CanvasROP::clear_depth(uint16_t z)
{
	for (unsigned y = 0; y < depth_canvas.get_height(); y++)
		for (unsigned x = 0; x < depth_canvas.get_width(); x++)
			depth_canvas.get(x, y) = z;
}

void CanvasROP::clear_color(uint32_t c)
{
	for (unsigned y = 0; y < canvas.get_height(); y++)
		for (unsigned x = 0; x < canvas.get_width(); x++)
			canvas.get(x, y) = c;
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

struct SoftwareRenderableComponent : ComponentBase
{
	GRANITE_COMPONENT_TYPE_DECL(SoftwareRenderableComponent)
	std::vector<Vertex> vertices;
	std::vector<Vertex> transformed_vertices;
	std::vector<uvec3> indices;
	SceneFormats::MemoryMappedTexture color_texture;
};

static void create_software_renderable(Entity *entity, RenderableComponent *renderable)
{
	auto *imported_mesh = dynamic_cast<ImportedMesh *>(renderable->renderable.get());
	if (!imported_mesh)
		return;

	auto &mesh = imported_mesh->get_mesh();

	if (mesh.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
	{
		LOGE("Unsupported topology.\n");
		return;
	}

	auto *sw = entity->allocate_component<SoftwareRenderableComponent>();

	unsigned num_vertices = mesh.positions.size() / mesh.position_stride;
	sw->vertices.resize(num_vertices);

	auto pos_format = mesh.attribute_layout[Util::ecast(MeshAttribute::Position)].format;
	auto normal_format = mesh.attribute_layout[Util::ecast(MeshAttribute::Normal)].format;
	auto normal_offset = mesh.attribute_layout[Util::ecast(MeshAttribute::Normal)].offset;

	for (unsigned i = 0; i < num_vertices; i++)
	{
		if (pos_format == VK_FORMAT_R32G32B32_SFLOAT)
		{
			memcpy(sw->vertices[i].clip, mesh.positions.data() + i * mesh.position_stride, 3 * sizeof(float));
			sw->vertices[i].w = 1.0f;
		}
		else if (pos_format == VK_FORMAT_R32G32B32A32_SFLOAT)
			memcpy(sw->vertices[i].clip, mesh.positions.data() + i * mesh.position_stride, 4 * sizeof(float));
		else
		{
			LOGE("Unknown position format.\n");
			entity->free_component<SoftwareRenderableComponent>();
			return;
		}

		if (normal_format == VK_FORMAT_R32G32B32_SFLOAT)
		{
			vec3 n;
			memcpy(n.data, mesh.attributes.data() + i * mesh.attribute_stride + normal_offset, 3 * sizeof(float));
			n = normalize(n);
			memcpy(sw->vertices[i].color, n.data, 3 * sizeof(float));
		}
		else
		{
			for (auto &c : sw->vertices[i].color)
				c = 1.0f;
		}
	}

	auto &mat = imported_mesh->get_material_info();
	sw->color_texture = load_texture_from_file(mat.base_color.path);

	if (mesh.attribute_layout[Util::ecast(MeshAttribute::UV)].format == VK_FORMAT_R32G32_SFLOAT)
	{
		auto offset = mesh.attribute_layout[Util::ecast(MeshAttribute::UV)].offset;
		for (unsigned i = 0; i < num_vertices; i++)
		{
			memcpy(&sw->vertices[i].u, mesh.attributes.data() + i * mesh.attribute_stride + offset, sizeof(float));
			memcpy(&sw->vertices[i].v, mesh.attributes.data() + i * mesh.attribute_stride + offset + sizeof(float), sizeof(float));
			sw->vertices[i].u = sw->vertices[i].u * float(sw->color_texture.get_layout().get_width());
			sw->vertices[i].v = sw->vertices[i].v * float(sw->color_texture.get_layout().get_height());
		}
	}

	sw->indices.reserve(mesh.count / 3);
	if (!mesh.indices.empty())
	{
		if (mesh.index_type == VK_INDEX_TYPE_UINT16)
		{
			auto *indices = reinterpret_cast<const uint16_t *>(mesh.indices.data());
			for (unsigned i = 0; i < mesh.count; i += 3)
				sw->indices.push_back(uvec3(indices[i + 0], indices[i + 1], indices[i + 2]));
		}
		else if (mesh.index_type == VK_INDEX_TYPE_UINT32)
		{
			auto *indices = reinterpret_cast<const uint32_t *>(mesh.indices.data());
			for (unsigned i = 0; i < mesh.count; i += 3)
				sw->indices.push_back(uvec3(indices[i + 0], indices[i + 1], indices[i + 2]));
		}
		else
		{
			LOGE("Unknown index type.\n");
			entity->free_component<SoftwareRenderableComponent>();
			return;
		}
	}
	else
	{
		for (unsigned i = 0; i < mesh.count; i += 3)
			sw->indices.push_back(uvec3(i, i + 1, i + 2));
	}

	sw->transformed_vertices = sw->vertices;
}

struct SWRenderApplication : Application
{
	explicit SWRenderApplication(const char *path);
	void render_frame(double, double) override;

	SceneLoader loader;
	CanvasROP rop;
	RasterizerCPU rasterizer;
	FPSCamera cam;
};

SWRenderApplication::SWRenderApplication(const char *path)
{
	loader.load_scene(path);

	auto &scene = loader.get_scene();
	auto *renderables_holder = scene.get_entity_pool().get_component_group_holder<RenderableComponent, OpaqueComponent, RenderInfoComponent>();
	auto &renderables = renderables_holder->get_groups();
	auto &renderable_entities = renderables_holder->get_entities();

	for (size_t i = 0; i < renderables.size(); i++)
		create_software_renderable(renderable_entities[i], get_component<RenderableComponent>(renderables[i]));

	cam.set_fovy(0.5f * pi<float>());
	cam.set_depth_range(0.1f, 100.0f);
	cam.set_aspect(640.0f / 360.0f);
	cam.look_at(vec3(0.0f, 0.0f, 3.0f), vec3(0.0f));

	rop.canvas.resize(640, 360);
	rop.depth_canvas.resize(640, 360);
	rasterizer.set_scissor(0, 0, 640, 360);
	rasterizer.set_rop(&rop);
}

static void transform_vertex(Vertex &out_vertex, const Vertex &in_vertex, const mat4 &mvp, const mat3 &normal_matrix)
{
	vec3 n = vec3(in_vertex.color[0], in_vertex.color[1], in_vertex.color[2]);
	n = normalize(normal_matrix * n);
	float ndotl = dot(n, vec3(0.6f, 0.8f, 0.4f)) + 0.2f;
	ndotl = clamp(ndotl, 0.0f, 1.0f);

	vec4 pos = vec4(in_vertex.x, in_vertex.y, in_vertex.z, 1.0f);
	vec4 clip = mvp * pos;
	out_vertex.color[0] = ndotl;
	out_vertex.color[1] = ndotl;
	out_vertex.color[2] = ndotl;
	out_vertex.color[3] = 1.0f;
	memcpy(out_vertex.clip, clip.data, 4 * sizeof(float));
}

void SWRenderApplication::render_frame(double, double)
{
	auto &device = get_wsi().get_device();
	auto &scene = loader.get_scene();
	scene.update_cached_transforms();

	rop.clear_color();
	rop.clear_depth();

	mat4 vp = cam.get_projection() * cam.get_view();
	ViewportTransform viewport_transform = { -0.5f, -0.5f, 640.0f, 360.0f, 0.0f, 1.0f };
	InputPrimitive input = {};
	PrimitiveSetup setups[256];
	TextureSampler sampler;
	rasterizer.set_sampler(&sampler);

	auto &renderables = scene.get_entity_pool().get_component_group<SoftwareRenderableComponent, RenderInfoComponent>();
	for (auto &renderable : renderables)
	{
		auto &m = get_component<RenderInfoComponent>(renderable)->transform->world_transform;
		mat4 mvp = vp * m;
		mat3 n = mat3(m);
		auto *sw = get_component<SoftwareRenderableComponent>(renderable);
		sampler.layout = &sw->color_texture.get_layout();

		size_t vertex_count = sw->vertices.size();
		for (size_t i = 0; i < vertex_count; i++)
			transform_vertex(sw->transformed_vertices[i], sw->vertices[i], mvp, n);

		for (auto &primitive : sw->indices)
		{
			input.vertices[0] = sw->transformed_vertices[primitive.x];
			input.vertices[1] = sw->transformed_vertices[primitive.y];
			input.vertices[2] = sw->transformed_vertices[primitive.z];
			unsigned count = setup_clipped_triangles(setups, input, CullMode::CCWOnly, viewport_transform);
			for (unsigned i = 0; i < count; i++)
				rasterizer.render_primitive(setups[i]);
		}
	}

	rop.fill_alpha_opaque();

	auto info = Vulkan::ImageCreateInfo::immutable_2d_image(rop.canvas.get_width(), rop.canvas.get_height(), VK_FORMAT_R8G8B8A8_SRGB);
	Vulkan::ImageInitialData initial = {};
	initial.data = rop.canvas.get_data();
	auto image = device.create_image(info, &initial);

	auto cmd = device.request_command_buffer();
	cmd->begin_render_pass(device.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly));
	cmd->set_texture(0, 0, image->get_view(), Vulkan::StockSampler::LinearClamp);
	Vulkan::CommandBufferUtil::draw_fullscreen_quad(*cmd, "builtin://shaders/quad.vert", "builtin://shaders/blit.frag");
	cmd->end_render_pass();
	device.submit(cmd);
}

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	setup_fixed_divider();
	if (argc != 2)
		return nullptr;

	Global::filesystem()->register_protocol("assets", std::make_unique<OSFilesystem>(ASSET_DIRECTORY));
	return new SWRenderApplication(argv[1]);
}
}

#if 0
int main(int argc, char **argv)
{

	Global::init();

	SceneLoader loader;
	loader.load_scene(argv[1]);

	auto &scene = loader.get_scene();
	auto *renderables_holder = scene.get_entity_pool().get_component_group_holder<RenderableComponent, RenderInfoComponent>();
	auto &renderables = renderables_holder->get_groups();
	auto &renderable_entities = renderables_holder->get_entities();

	for (size_t i = 0; i < renderables.size(); i++)
		create_software_renderable(renderable_entities[i], get_component<RenderableComponent>(renderables[i]));

	ViewportTransform vp = { 0.0f, 0.0f, 1920.0f, 1080.0f, 0.0f, 1.0f };
	PrimitiveSetup setup[256];

	for (auto &prim : input_primitives)
	{
		//unsigned prim_index = unsigned(&prim - input_primitives.data());
		unsigned count = setup_clipped_triangles(setup, prim, CullMode::CCWOnly, vp);
		for (unsigned i = 0; i < count; i++)
		{
			rasterizer.render_primitive(setup[i]);
		}
	}

	rop.save_canvas("/tmp/test.png");


#if 0
	RasterizerGPU gpu;
	gpu.resize(1920, 1080);
	gpu.upload_texture(sampler.tex.get_layout());
	gpu.clear_color(0);
	gpu.clear_depth();
	gpu.rasterize_primitives(setups.data(), setups.size());
	gpu.save_canvas("/tmp/test_gpu.png");
	float ratio = gpu.get_binning_ratio(setups.size());
	LOGI("Binning ratio: %f %%\n", 100.0f * ratio);
#endif
}
#endif
