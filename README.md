# RetroWarp

This is a straight forward Vulkan compute shader implementation of a fictional GPU.

## Update submodules

This implementation makes use of my Granite engine.

```
git submodule update --init --recursive
```

## Build

Standard CMake build. Should build on GCC, Clang and MSVC 2015+.

## `viewer`

A simple test program which renders a glTF 2.0 file in real-time.
Don't expect good compatibility as all the meshes have to be translated into a `SWRenderableComponent`, which implements the bare minimum.
Some test models I've used are Sponza, Suzanne or Lantern from KhronosGroup/glTF-Sample-Models.
Most likely, the application will be CPU bound as all vertex processing is done with unoptimized CPU code unless the "freeze" feature is used.

### Controls

- WASD: Move camera around
- Hold right-click: Rotate camera
- U: Freeze the frame, no vertex processing on CPU is done, which is useful for testing GPU bound scenario.
- C: Dumps the current frame to `retrowarp.dump` along with textures. This can be replayed and benchmarked in `dump-bench`.
- Space: Toggle vsync.

### Options

- `--width`: Control resolution. Maximum is 2048.
- `--height`: Control resolution. Maximum is 2048.
- `--tile-size`: Tile size, use 8 or 16.
- `--ubershader`: Use ubershader rather than split shader architecture.
- `--nosubgroup`: Disable all subgroup support.
- `--async-compute`: Enable async compute support.

## `dump-bench`

Takes a `.dump` file created by the `viewer` application and replays that frame over and over.
At the end, performance metrics are reported in time / iteration (i.e. frame).

### Options

- `--tile-size`: Tile size, use 8 or 16.
- `--ubershader`: Use ubershader rather than split shader architecture.
- `--nosubgroup`: Disable all subgroup support.
- `--async-compute`: Enable async compute support.
- `--iterations`: Number of iterations.

Resolution is specified in the dump as it contains post-triangle setup data and cannot be rescaled.

## Implementation

`rasterizer_gpu.hpp` and `rasterizer_gpu.cpp` implement the Vulkan side of things.
Shaders are contained in `assets/shaders`.
