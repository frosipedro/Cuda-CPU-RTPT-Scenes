# Build all scene binaries into bin/ for the WSL workflow.
SCENE_DIR := scenes
BIN_DIR := bin
NVCC := nvcc
NVFLAGS := -O2

SOURCES := \
	$(SCENE_DIR)/simple_raytracer.cu \
	$(SCENE_DIR)/simple_path_tracer.cu \
	$(SCENE_DIR)/procedural_path_tracer.cu

BINARIES := $(patsubst $(SCENE_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

.PHONY: all help run clean remove

all: $(BINARIES)

help:
	@echo "Available targets:"
	@echo "  make        Build all scenes"
	@echo "  make run    Build and open the WSL scene launcher"
	@echo "  make clean  Remove the bin/ directory"
	@echo ""
	@echo "Scenes:"
	@echo "  simple_raytracer         - Advanced RT with CPU, GPU, or both"
	@echo "  simple_path_tracer       - Path tracer with CPU/OpenMP options"
	@echo "  procedural_path_tracer   - GPU-only heavy path tracer"

$(BIN_DIR):
	@if [ ! -d "$(BIN_DIR)" ]; then mkdir -p "$(BIN_DIR)"; fi

$(BIN_DIR)/simple_path_tracer: $(SCENE_DIR)/simple_path_tracer.cu | $(BIN_DIR)
	$(NVCC) $< $(NVFLAGS) -Xcompiler -fopenmp -o $@

$(BIN_DIR)/%: $(SCENE_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $< $(NVFLAGS) -o $@

run: all
	bash ./run.sh

clean: rm

rm:
	@if [ -d "$(BIN_DIR)" ]; then rm -rf "$(BIN_DIR)"; fi