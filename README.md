# CUDA Ray Tracing Scenes

This repository contains three standalone CUDA rendering scenes that share the same general goal: make it easy to compare CPU and GPU rendering while keeping the codebase small enough to study.

The current layout keeps the original rendering ideas intact, but the user experience is more structured:

- one launcher to choose the scene
- clear scene names
- consistent English prompts
- timestamped PNG outputs in `images/` to avoid overwriting previous renders

## Scenes

| Scene                    | Description                                                                   | Supported modes                                       |
| ------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| `simple_raytracer`       | An advanced ray tracer with reflections, direct lighting, and a richer scene. | CPU, GPU, or both                                     |
| `hybrid_ray_path_tracer` | A path tracer with GPU rendering and optional CPU/OpenMP reference modes.     | Path tracing on GPU plus optional CPU reference modes |
| `procedural_path_tracer` | A GPU-only procedural path tracer with the heaviest workload in the project.  | GPU only                                              |

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- GNU Make
- Bash shell support through WSL or another Unix-like environment
- `stb_image_write.h` is already included in `scenes/`

### Notes for Windows users

The project is wired primarily for WSL. `make` uses POSIX shell commands and `make run` opens a Bash launcher script. If you prefer native Windows, you can still build with MinGW/MSYS2 and run the binaries directly, but WSL is the recommended path.

## Build

Build every scene with:

```sh
make
```

Useful targets:

```sh
make help
make run
make clean
```

`make run` builds the project and opens the WSL scene launcher.

### Output binaries

The build creates these executables in `bin/`:

- `bin/simple_raytracer`
- `bin/hybrid_ray_path_tracer`
- `bin/procedural_path_tracer`

On Windows, the generated files may appear with a `.exe` suffix.

## Run

The recommended workflow is:

```sh
make run
```

The launcher shows the three scenes with short descriptions. After you choose one, the selected scene asks its own questions in the terminal.

If you prefer direct execution, you can run the binaries yourself:

```sh
./bin/simple_raytracer
./bin/hybrid_ray_path_tracer
./bin/procedural_path_tracer
```

The programs save PNG files in the `images/` directory. The directory is created automatically if it does not exist.

Files produced during the same execution share the same timestamp suffix, which keeps CPU/GPU comparison renders grouped together.

## Prompt Style

All scene prompts are English-only and follow the same rules:

- press Enter to accept the default value
- use whole numbers where integers are requested
- use the example values shown in the prompt if you want a safe first test
- the launcher chooses the scene only; the selected scene handles its own rendering options

## Scene Details

### 1. Simple Ray Tracer

This scene is the best starting point if you want a quick CPU vs GPU comparison.

Prompt flow:

1. Choose the render mode:
   - `1` = CPU
   - `2` = GPU
   - `3` = compare both
2. The framebuffer is fixed at `3840x2880` in the current version.

Recommended use:

- choose GPU first for a faster test
- choose both only when you want to compare timing and output directly

Output files:

- `images/simple_raytracer_cpu_YYYYMMDD_HHMMSS.png`
- `images/simple_raytracer_gpu_YYYYMMDD_HHMMSS.png`

### 2. Hybrid Ray/Path Tracer

This is the most flexible scene in the repository. It contains two rendering modes inside one binary.

Choose this when you want the heavier, more realistic path tracer.

Prompt flow:

1. Enter resolution as `width height`
2. Choose samples per pixel
3. Choose the maximum bounce depth
4. Choose the CPU reference mode:
   - `0` = none
   - `1` = single-thread CPU
   - `2` = OpenMP CPU
   - `3` = both CPU references
5. If you select OpenMP, choose the thread count or press Enter for automatic detection

Recommended first run:

- resolution: `1280 720`
- samples: `64`
- max bounces: `6`
- CPU reference: `0`

Output files:

- `images/hybrid_pt_gpu_YYYYMMDD_HHMMSS.png`
- `images/hybrid_pt_cpu_ref_YYYYMMDD_HHMMSS.png`
- `images/hybrid_pt_cpu_omp_YYYYMMDD_HHMMSS.png`

### 3. Procedural Path Tracer

This is the heaviest scene in the repository and is GPU-only.

Prompt flow:

1. Enter resolution as `width height`
2. Choose samples per pixel
3. Choose max bounces
4. Choose fog density

Recommended first run:

- resolution: `1280 720`
- samples: `64`
- max bounces: `6`
- fog density: `0.02`

Output file:

- `images/procedural_path_tracer_gpu_YYYYMMDD_HHMMSS.png`

## Performance Tips

- Start with `1280x720` and a small sample count to confirm composition.
- Increase samples only after the camera and scene look correct.
- Increase bounce depth only when the scene benefits from additional indirect light.
- The procedural path tracer is intentionally expensive; use it as a quality scene, not a quick preview.
- If you want a faster build, keep `-O2` in the Makefile. If you want more speed and can tolerate longer compilation, you can experiment with `-O3` or `-use_fast_math`.

## Troubleshooting

- `nvcc` not found: install the CUDA Toolkit and make sure the CUDA binaries are on your `PATH`.
- `pwsh` not found: install PowerShell 7 or run the scene binaries directly instead of using `make run`.
- `stb_image_write.h` missing: the header should already be present in `scenes/`; restore it if it was removed.
- Build fails on `rm` or shell syntax: use a shell environment that supports the Makefile commands, such as WSL or Git Bash.
- Render is too slow: lower the resolution, reduce samples, or use the simple ray tracer before trying the heavy path tracer.

## Credits

- Cristian dos Santos Siquiera - https://github.com/CristianSSiqueira
- Pedro Rockenbach Frosi - https://github.com/frosipedro

The repository also uses `stb_image_write.h` from Sean Barrett's stb library:

- https://github.com/nothings/stb
