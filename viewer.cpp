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

using namespace RetroWarp;
using namespace Granite;

constexpr int TEXTURE_BASE_LEVEL = 1;

#if 0
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
#endif

struct SoftwareRenderableComponent : ComponentBase
{
	GRANITE_COMPONENT_TYPE_DECL(SoftwareRenderableComponent)
	std::vector<Vertex> vertices;
	std::vector<Vertex> transformed_vertices;
	std::vector<uvec3> indices;
	SceneFormats::MemoryMappedTexture color_texture;
	unsigned state_index;
};


struct SWRenderApplication : Application, EventHandler
{
	explicit SWRenderApplication(const std::string &path, bool subgroup, bool ubershader, bool async_compute,
	                             unsigned width, unsigned height, unsigned tile_size);
	void render_frame(double, double) override;

	SceneLoader loader;
	//CanvasROP rop;
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
	void dump_alpha_threshold(uint8_t threshold);
	void dump_rop_state(BlendState blend_state);

	struct Cached
	{
		unsigned index;
		const Vulkan::ImageView *view;
		PrimitiveSetup setup;
		DrawPipeline pipeline;
	};
	std::vector<Cached> setup_cache;
	bool update_setup_cache = true;
	bool subgroup;
	bool ubershader;
	bool async_compute;
	unsigned fb_width;
	unsigned fb_height;
	unsigned tile_size;

	std::unordered_map<std::string, unsigned> state_index_map;
	std::vector<const Vulkan::TextureFormatLayout *> state_index_layout;
	std::vector<TextureDescriptor> texture_descriptors;
	void create_software_renderable(Entity *entity, RenderableComponent *renderable);
};

void SWRenderApplication::create_software_renderable(Entity *entity, RenderableComponent *renderable)
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
		state_index_layout.push_back(&sw->color_texture.get_layout());
	}
	else
	{
		unsigned index = itr->second;
		sw->state_index = index;
	}

	if (mesh.attribute_layout[Util::ecast(MeshAttribute::UV)].format == VK_FORMAT_R32G32_SFLOAT)
	{
		uint32_t width = sw->color_texture.get_layout().get_width();
		uint32_t height = sw->color_texture.get_layout().get_height();
		width = std::max((width >> TEXTURE_BASE_LEVEL), 1u);
		height = std::max((height >> TEXTURE_BASE_LEVEL), 1u);

		auto offset = mesh.attribute_layout[Util::ecast(MeshAttribute::UV)].offset;
		for (unsigned i = 0; i < num_vertices; i++)
		{
			memcpy(&sw->vertices[i].u, mesh.attributes.data() + i * mesh.attribute_stride + offset, sizeof(float));
			memcpy(&sw->vertices[i].v, mesh.attributes.data() + i * mesh.attribute_stride + offset + sizeof(float), sizeof(float));
			sw->vertices[i].u = sw->vertices[i].u * float(width);
			sw->vertices[i].v = sw->vertices[i].v * float(height);
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

void SWRenderApplication::on_device_created(const Vulkan::DeviceCreatedEvent& e)
{
	rasterizer_gpu.init(e.get_device(), subgroup, ubershader, async_compute, tile_size);
	rasterizer_gpu.set_rop_state(BlendState::Replace);
	rasterizer_gpu.set_depth_state(DepthTest::LE, DepthWrite::On);
	rasterizer_gpu.set_combiner_mode(COMBINER_MODE_TEX_MOD_COLOR | COMBINER_SAMPLE_BIT);

	uint32_t addr = 0;
	rasterizer_gpu.set_color_framebuffer(addr, fb_width, fb_height, fb_width * 2);
	addr += fb_width * fb_height * 2;
	rasterizer_gpu.set_depth_framebuffer(addr, fb_width, fb_height, fb_width * 2);
	addr += fb_width * fb_height * 2;

	unsigned num_textures = state_index_layout.size();
	for (unsigned i = 0; i < num_textures; i++)
	{
		auto texture = SceneFormats::generate_mipmaps(*state_index_layout[i], 0);
		auto &layout = texture.get_layout();
		unsigned levels = std::min(layout.get_levels() - TEXTURE_BASE_LEVEL, 8u);

		TextureFormatBits fmt = TEXTURE_FMT_ARGB1555;
		TextureDescriptor descriptor;
		descriptor.texture_fmt = fmt | TEXTURE_FMT_FILTER_MIP_LINEAR_BIT | TEXTURE_FMT_FILTER_LINEAR_BIT;
		descriptor.texture_clamp = i16vec4(-0x8000, -0x8000, 0x7fff, 0x7fff);
		descriptor.texture_mask = u16vec2(layout.get_width(TEXTURE_BASE_LEVEL) - 1,
		                                  layout.get_height(TEXTURE_BASE_LEVEL) - 1);
		descriptor.texture_max_lod = levels - 1;
		descriptor.texture_width = layout.get_width(TEXTURE_BASE_LEVEL);

		addr = (addr + 63) & ~63;

		for (unsigned level = 0; level < levels; level++)
		{
			unsigned mip_width = layout.get_width(level + TEXTURE_BASE_LEVEL);
			unsigned mip_height = layout.get_height(level + TEXTURE_BASE_LEVEL);
			descriptor.texture_offset[level] = addr;

			uint32_t blocks_width = (mip_width + 7) / 8;
			uint32_t blocks_height = (mip_height + 7) / 8;

			rasterizer_gpu.copy_texture_rgba8888_to_vram(addr,
			                                             static_cast<const uint32_t *>(layout.data(0, level + TEXTURE_BASE_LEVEL)),
			                                             mip_width, mip_height, fmt);
			addr += blocks_width * blocks_height * 64 * sizeof(uint16_t);
		}

		texture_descriptors.push_back(descriptor);
	}

	LOGI("Allocated %u bytes.\n", addr);
}

void SWRenderApplication::on_device_destroyed(const Vulkan::DeviceCreatedEvent &)
{
}

void SWRenderApplication::begin_dump_frame()
{
	dump_file = fopen("retrowarp.dump", "wb");
	if (!dump_file)
	{
		LOGE("Failed to dump.\n");
		exit(EXIT_FAILURE);
	}

	fwrite("RETROWARP DUMP01", 1, 16, dump_file);

	uint32_t word = fb_width;
	fwrite(&word, 1, sizeof(word), dump_file);
	word = fb_height;
	fwrite(&word, 1, sizeof(word), dump_file);
}

void SWRenderApplication::dump_textures(const std::vector<SceneFormats::MemoryMappedTexture *> &textures)
{
	if (!dump_file)
		return;
	uint32_t word = textures.size();
	fwrite(&word, 1, sizeof(word), dump_file);
	for (unsigned i = 0; i < textures.size(); i++)
		textures[i]->copy_to_path(std::string("retrowarp.dump.tex.") + std::to_string(i));
}

void SWRenderApplication::dump_set_texture(unsigned index)
{
	if (!dump_file)
		return;
	uint32_t word = index;
	fwrite("TEX ", 1, 4, dump_file);
	fwrite(&word, 1, sizeof(word), dump_file);
}

void SWRenderApplication::dump_alpha_threshold(uint8_t threshold)
{
	fwrite("ATRS", 1, 4, dump_file);
	uint32_t word = threshold;
	fwrite(&word, 1, sizeof(word), dump_file);
}

void SWRenderApplication::dump_rop_state(BlendState blend_state)
{
	fwrite("BSTA", 1, 4, dump_file);
	uint32_t word = uint32_t(blend_state);
	fwrite(&word, 1, sizeof(word), dump_file);
}

void SWRenderApplication::dump_primitives(const PrimitiveSetup *setup, unsigned count)
{
	if (!dump_file)
		return;
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

SWRenderApplication::SWRenderApplication(const std::string &path, bool subgroup_, bool ubershader_, bool async_compute_,
                                         unsigned width_, unsigned height_, unsigned tile_size_)
		: subgroup(subgroup_), ubershader(ubershader_), async_compute(async_compute_),
		  fb_width(width_), fb_height(height_), tile_size(tile_size_)
{
	loader.load_scene(path);
	get_wsi().set_backbuffer_srgb(false);

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

#if 0
	rop.canvas.resize(width, height);
	rop.depth_canvas.resize(width, height);
	rasterizer.set_scissor(0, 0, width, height);
	rasterizer.set_rop(&rop);
#endif

	EVENT_MANAGER_REGISTER_LATCH(SWRenderApplication, on_device_created, on_device_destroyed, Vulkan::DeviceCreatedEvent);
	EVENT_MANAGER_REGISTER(SWRenderApplication, on_key_pressed, KeyboardEvent);
}

bool SWRenderApplication::on_key_pressed(const KeyboardEvent &e)
{
	if (e.get_key_state() == KeyState::Pressed && e.get_key() == Key::C)
		queue_dump_frame = true;
	else if (e.get_key_state() == KeyState::Pressed && e.get_key() == Key::U)
		update_setup_cache = !update_setup_cache;
	else if (e.get_key_state() == KeyState::Pressed && e.get_key() == Key::Space)
		get_wsi().set_present_mode(get_wsi().get_present_mode() == Vulkan::PresentMode::SyncToVBlank ? Vulkan::PresentMode::Unlocked : Vulkan::PresentMode::SyncToVBlank);
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

void SWRenderApplication::render_frame(double frame_time, double)
{
	auto &device = get_wsi().get_device();
	auto &scene = loader.get_scene();
	scene.update_cached_transforms();

#if 0
	rop.clear_color();
	rop.clear_depth();
#endif
	rasterizer_gpu.clear_color();
	rasterizer_gpu.clear_depth();

	mat4 vp = cam.get_projection() * cam.get_view();
	ViewportTransform viewport_transform = { -0.5f, -0.5f, float(fb_width), float(fb_height), 0.0f, 1.0f };
	InputPrimitive input = {};
	PrimitiveSetup setups[256];
#if 0
	TextureSampler sampler;
	rasterizer.set_sampler(&sampler);
#endif

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

	if (update_setup_cache)
	{
		setup_cache.clear();
		for (auto &renderable : renderables)
		{
			auto &m = get_component<RenderInfoComponent>(renderable)->transform->world_transform;
			mat4 mvp = vp * m;
			mat3 n = mat3(m);
			auto *sw = get_component<SoftwareRenderableComponent>(renderable);

			auto *render = get_component<RenderableComponent>(renderable);
			auto *static_mesh = dynamic_cast<ImportedMesh *>(render->renderable.get());
			if (!static_mesh)
				continue;

			auto two_sided = static_mesh->material->two_sided;
			//bool two_sided = false;
			auto pipeline = static_mesh->material->pipeline;

			size_t vertex_count = sw->vertices.size();
			for (size_t i = 0; i < vertex_count; i++)
				transform_vertex(sw->transformed_vertices[i], sw->vertices[i], mvp, n);

			for (auto &primitive : sw->indices)
			{
				input.vertices[0] = sw->transformed_vertices[primitive.x];
				input.vertices[1] = sw->transformed_vertices[primitive.y];
				input.vertices[2] = sw->transformed_vertices[primitive.z];

				unsigned count = setup_clipped_triangles(setups, input,
				                                         two_sided ? CullMode::None : CullMode::CCWOnly,
				                                         viewport_transform);

				for (unsigned i = 0; i < count; i++)
				{
					setup_cache.push_back({
							                      sw->state_index,
							                      &static_mesh->material->textures[Util::ecast(
									                      Material::Textures::BaseColor)]->get_image()->get_view(),
							                      setups[i],
							                      pipeline,
					                      });
				}
			}
		}
	}
	else
		LOGI("Cached %u primitive setups!\n", unsigned(setup_cache.size()));

#if 1
	for (auto &setup : setup_cache)
	{
		if (queue_dump_frame)
			dump_set_texture(setup.index);

		auto pipeline = setup.pipeline;
		switch (pipeline)
		{
		case DrawPipeline::Opaque:
			rasterizer_gpu.set_alpha_threshold(0);
			rasterizer_gpu.set_rop_state(BlendState::Replace);
			if (queue_dump_frame)
			{
				dump_alpha_threshold(0);
				dump_rop_state(BlendState::Replace);
			}
			break;

		case DrawPipeline::AlphaTest:
			rasterizer_gpu.set_alpha_threshold(128);
			rasterizer_gpu.set_rop_state(BlendState::Replace);
			if (queue_dump_frame)
			{
				dump_alpha_threshold(128);
				dump_rop_state(BlendState::Replace);
			}
			break;

		case DrawPipeline::AlphaBlend:
			rasterizer_gpu.set_alpha_threshold(0);
			rasterizer_gpu.set_rop_state(BlendState::Alpha);
			if (queue_dump_frame)
			{
				dump_alpha_threshold(0);
				dump_rop_state(BlendState::Alpha);
			}
			break;
		}

		rasterizer_gpu.set_texture_descriptor(texture_descriptors[setup.index]);
		rasterizer_gpu.rasterize_primitives(&setup.setup, 1);
		if (queue_dump_frame)
			dump_primitives(&setup.setup, 1);
	}
#else
	for (unsigned i = 0; i < 40; i++)
	{
		InputPrimitive prim = {};
		prim.vertices[0].x = -0.5f + i / 200.0f;
		prim.vertices[0].y = -0.5f + i / 200.0f;
		prim.vertices[1].x = -0.5f + i / 200.0f;
		prim.vertices[1].y = +0.5f + i / 200.0f;
		prim.vertices[2].x = +0.5f + i / 200.0f;
		prim.vertices[2].y = -0.5f + i / 200.0f;
		prim.vertices[0].w = 1.0f;
		prim.vertices[1].w = 1.0f;
		prim.vertices[2].w = 1.0f;
		prim.vertices[0].z = i / 64.0f;
		prim.vertices[1].z = i / 64.0f;
		prim.vertices[2].z = i / 64.0f;
		if (i & 1)
		{
			prim.vertices[0].color[0] = 1.0f;
			prim.vertices[1].color[1] = 1.0f;
			prim.vertices[2].color[2] = 1.0f;
		}
		else
		{
			prim.vertices[0].color[2] = 1.0f;
			prim.vertices[1].color[1] = 1.0f;
			prim.vertices[2].color[0] = 1.0f;
		}
		rasterizer_gpu.set_texture(*setup_cache.front().view);
		rasterizer_gpu.set_combiner_mode(COMBINER_MODE_COLOR);
		unsigned count = setup_clipped_triangles(setups, prim, CullMode::None, viewport_transform);
		rasterizer_gpu.rasterize_primitives(setups, count);
	}
#endif

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

	LOGI("Frame time: %.3f ms\n", 1000.0 * frame_time);
}

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	bool ubershader = false;
	bool subgroup = true;
	bool async_compute = false;
	std::string path;
	unsigned width = 640;
	unsigned height = 360;
	unsigned tile_size = 8;

	Util::CLICallbacks cbs;
	cbs.add("--ubershader", [&](Util::CLIParser &) { ubershader = true; });
	cbs.add("--nosubgroup", [&](Util::CLIParser &) { subgroup = false; });
	cbs.add("--async-compute", [&](Util::CLIParser &) { async_compute = true; });
	cbs.add("--width", [&](Util::CLIParser &parser) { width = parser.next_uint(); });
	cbs.add("--height", [&](Util::CLIParser &parser) { height = parser.next_uint(); });
	cbs.add("--tile-size", [&](Util::CLIParser &parser) { tile_size = parser.next_uint(); });
	cbs.default_handler = [&](const char *arg) { path = arg; };
	Util::CLIParser parser(std::move(cbs), argc - 1, argv + 1);

	if (!parser.parse() || path.empty())
	{
		LOGE("Failed to parse.\n");
		return nullptr;
	}

	if (tile_size & (tile_size - 1))
	{
		LOGE("Tile size must be POT.\n");
		return nullptr;
	}

	Global::filesystem()->register_protocol("assets", std::make_unique<OSFilesystem>(ASSET_DIRECTORY));
	return new SWRenderApplication(path, subgroup, ubershader, async_compute, width, height, tile_size);
}
}
