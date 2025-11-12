# Compila todos os .cu em scene/ para bin/
SCENE_DIR := scenes
BIN_DIR := bin
NVCC := nvcc
NVFLAGS := -O2

SOURCES := $(wildcard $(SCENE_DIR)/*.cu)
BINARIES := $(patsubst $(SCENE_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

.PHONY: all clean remove

all: $(BINARIES)

$(BIN_DIR):
	@if [ ! -d "$(BIN_DIR)" ]; then mkdir -p "$(BIN_DIR)"; fi

$(BIN_DIR)/rt_and_pt: $(SCENE_DIR)/rt_and_pt.cu | $(BIN_DIR)
	$(NVCC) $< $(NVFLAGS) -Xcompiler -fopenmp -o $@

$(BIN_DIR)/%: $(SCENE_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $< $(NVFLAGS) -o $@

clean: rm

rm:
	@if [ -d "$(BIN_DIR)" ]; then rm -rf "$(BIN_DIR)"; fi