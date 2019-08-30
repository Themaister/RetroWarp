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
	unsigned state_index;
};

static std::unordered_map<std::string, unsigned> state_index_map;

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

	auto itr = state_index_map.find(mat.base_color.path);
	if (itr == end(state_index_map))
	{
		unsigned index = state_index_map.size();
		state_index_map[mat.base_color.path] = index;
		sw->state_index = index;
	}
	else
	{
		unsigned index = itr->second;
		sw->state_index = index;
	}

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

struct SWRenderApplication : Application, EventHandler
{
	explicit SWRenderApplication(const char *path);
	void render_frame(double, double) override;

	SceneLoader loader;
	CanvasROP rop;
	RasterizerCPU rasterizer;
	FPSCamera cam;
	RasterizerGPU rasterizer_gpu;

	void on_device_created(const Vulkan::DeviceCreatedEvent &e);
	void on_device_destroyed(const Vulkan::DeviceCreatedEvent &);
	bool on_key_pressed(const KeyboardEvent &e);
	bool queue_dump_frame = false;

	FILE *dump_file = nullptr;
	void begin_dump_frame();
	void end_dump_frame();
	void dump_textures(const std::vector<SceneFormats::MemoryMappedTexture *> &textures);
	void dump_set_texture(unsigned index);
	void dump_primitives(const PrimitiveSetup *setup, unsigned count);
};

constexpr unsigned WIDTH = 640;
constexpr unsigned HEIGHT = 360;

void SWRenderApplication::on_device_created(const Vulkan::DeviceCreatedEvent& e)
{
	rasterizer_gpu.init(e.get_device());
	rasterizer_gpu.resize(WIDTH, HEIGHT);
}

void SWRenderApplication::on_device_destroyed(const Vulkan::DeviceCreatedEvent &)
{
}

void SWRenderApplication::begin_dump_frame()
{
	dump_file = fopen("/tmp/retrowarp.dump", "wb");
	if (!dump_file)
	{
		LOGE("Failed to dump.\n");
		exit(EXIT_FAILURE);
	}

	fwrite("RETROWARP DUMP01", 1, 16, dump_file);
}

void SWRenderApplication::dump_textures(const std::vector<SceneFormats::MemoryMappedTexture *> &textures)
{
	uint32_t word = textures.size();
	fwrite(&word, 1, sizeof(word), dump_file);
	for (unsigned i = 0; i < textures.size(); i++)
		textures[i]->copy_to_path(std::string("/tmp/retrowarp.dump.tex.") + std::to_string(i));
}

void SWRenderApplication::dump_set_texture(unsigned index)
{
	if (!dump_file)
		return;
	uint32_t word = index;
	fwrite("TEX ", 1, 4, dump_file);
	fwrite(&word, 1, sizeof(word), dump_file);
}

void SWRenderApplication::dump_primitives(const PrimitiveSetup *setup, unsigned count)
{
	for (unsigned i = 0; i < count; i++)
	{
		fwrite("PRIM", 1, 4, dump_file);
		fwrite(&setup[i], 1, sizeof(PrimitiveSetup), dump_file);
	}
}

void SWRenderApplication::end_dump_frame()
{
	if (dump_file)
		fclose(dump_file);
	dump_file = nullptr;
}

SWRenderApplication::SWRenderApplication(const char *path)
{
	loader.load_scene(path);

	auto &scene = loader.get_scene();
	auto *renderables_holder = scene.get_entity_pool().get_component_group_holder<RenderableComponent, OpaqueComponent, RenderInfoComponent>();
	auto &renderables = renderables_holder->get_groups();
	auto &renderable_entities = renderables_holder->get_entities();

	for (size_t i = 0; i < renderables.size(); i++)
		create_software_renderable(renderable_entities[i], get_component<RenderableComponent>(renderables[i]));

	cam.set_fovy(0.4f * pi<float>());
	cam.set_depth_range(0.1f, 100.0f);
	cam.set_aspect(640.0f / 360.0f);
	cam.look_at(vec3(0.0f, 0.0f, 3.0f), vec3(0.0f));

	rop.canvas.resize(WIDTH, HEIGHT);
	rop.depth_canvas.resize(WIDTH, HEIGHT);
	rasterizer.set_scissor(0, 0, WIDTH, HEIGHT);
	rasterizer.set_rop(&rop);

	EVENT_MANAGER_REGISTER_LATCH(SWRenderApplication, on_device_created, on_device_destroyed, Vulkan::DeviceCreatedEvent);
	EVENT_MANAGER_REGISTER(SWRenderApplication, on_key_pressed, KeyboardEvent);
}

bool SWRenderApplication::on_key_pressed(const KeyboardEvent &e)
{
	if (e.get_key_state() == KeyState::Pressed && e.get_key() == Key::C)
		queue_dump_frame = true;
	return true;
}

static void transform_vertex(Vertex &out_vertex, const Vertex &in_vertex, const mat4 &mvp, const mat3 &normal_matrix)
{
	vec3 n = vec3(in_vertex.color[0], in_vertex.color[1], in_vertex.color[2]);
	n = normalize(normal_matrix * n);
	float ndotl = clamp(dot(n, vec3(0.6f, 0.8f, 0.4f)), 0.0f, 1.0f) * 0.9f + 0.1f;

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
	rasterizer_gpu.clear_color(0);
	rasterizer_gpu.clear_depth();

	mat4 vp = cam.get_projection() * cam.get_view();
	ViewportTransform viewport_transform = { -0.5f, -0.5f, float(WIDTH), float(HEIGHT), 0.0f, 1.0f };
	InputPrimitive input = {};
	PrimitiveSetup setups[256];
	TextureSampler sampler;
	rasterizer.set_sampler(&sampler);

	auto &renderables = scene.get_entity_pool().get_component_group<RenderableComponent, SoftwareRenderableComponent, RenderInfoComponent>();

	sort(begin(renderables), end(renderables), [&](const auto &a, const auto &b) {
		return get_component<SoftwareRenderableComponent>(a)->state_index < get_component<SoftwareRenderableComponent>(b)->state_index;
	});

	if (queue_dump_frame)
	{
		begin_dump_frame();
		unsigned max_state_index = 0;
		for (auto &renderable : renderables)
			max_state_index = std::max(max_state_index, get_component<SoftwareRenderableComponent>(renderable)->state_index);
		std::vector<SceneFormats::MemoryMappedTexture *> source_paths(max_state_index + 1);
		for (auto &renderable : renderables)
		{
			auto *sw = get_component<SoftwareRenderableComponent>(renderable);
			source_paths[sw->state_index] = &sw->color_texture;
		}
		dump_textures(source_paths);
	}

	unsigned current_state[RasterizerGPU::NUM_STATE_INDICES];
	for (unsigned i = 0; i < RasterizerGPU::NUM_STATE_INDICES; i++)
		current_state[i] = ~0u;

	for (auto &renderable : renderables)
	{
		auto &m = get_component<RenderInfoComponent>(renderable)->transform->world_transform;
		mat4 mvp = vp * m;
		mat3 n = mat3(m);
		auto *sw = get_component<SoftwareRenderableComponent>(renderable);

		if (current_state[sw->state_index] != ~0u &&
		    current_state[sw->state_index & (RasterizerGPU::NUM_STATE_INDICES - 1)] != sw->state_index)
		{
			rasterizer_gpu.flush();
		}

		if (queue_dump_frame)
			dump_set_texture(sw->state_index);

		current_state[sw->state_index & (RasterizerGPU::NUM_STATE_INDICES - 1)] = sw->state_index;

		sampler.layout = &sw->color_texture.get_layout();

		auto *render = get_component<RenderableComponent>(renderable);
		auto *static_mesh = dynamic_cast<ImportedMesh *>(render->renderable.get());
		if (!static_mesh)
			continue;

		auto *gpu_texture = static_mesh->material->textures[Util::ecast(Material::Textures::BaseColor)];
		rasterizer_gpu.set_state_index(sw->state_index & (RasterizerGPU::NUM_STATE_INDICES - 1));
		rasterizer_gpu.set_texture(sw->state_index & (RasterizerGPU::NUM_STATE_INDICES - 1), gpu_texture->get_image()->get_view());

		size_t vertex_count = sw->vertices.size();
		for (size_t i = 0; i < vertex_count; i++)
			transform_vertex(sw->transformed_vertices[i], sw->vertices[i], mvp, n);

		for (auto &primitive : sw->indices)
		{
			input.vertices[0] = sw->transformed_vertices[primitive.x];
			input.vertices[1] = sw->transformed_vertices[primitive.y];
			input.vertices[2] = sw->transformed_vertices[primitive.z];
			unsigned count = setup_clipped_triangles(setups, input, CullMode::CCWOnly, viewport_transform);
			rasterizer_gpu.rasterize_primitives(setups, count);
			if (queue_dump_frame)
				dump_primitives(setups, count);
		}
	}

	auto image_gpu = rasterizer_gpu.copy_to_framebuffer();

	auto cmd = device.request_command_buffer();
	cmd->begin_render_pass(device.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly));
	cmd->set_texture(0, 0, image_gpu->get_view(), Vulkan::StockSampler::LinearClamp);
	Vulkan::CommandBufferUtil::draw_fullscreen_quad(*cmd, "builtin://shaders/quad.vert", "builtin://shaders/blit.frag");
	cmd->end_render_pass();
	device.submit(cmd);

	if (queue_dump_frame)
		end_dump_frame();
	queue_dump_frame = false;
}

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	//setup_fixed_divider();
	if (argc != 2)
		return nullptr;

	Global::filesystem()->register_protocol("assets", std::make_unique<OSFilesystem>(ASSET_DIRECTORY));
	return new SWRenderApplication(argv[1]);
}
}
