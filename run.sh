#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

resolve_binary() {
    local base_name="$1"

    if [[ -x "bin/$base_name" ]]; then
        printf '%s\n' "bin/$base_name"
        return 0
    fi

    if [[ -x "bin/$base_name.exe" ]]; then
        printf '%s\n' "bin/$base_name.exe"
        return 0
    fi

    return 1
}

show_menu() {
    cat <<'EOF'
CUDA Ray/Path Tracer Launcher (WSL)

1. Simple Ray Tracer
    CPU, GPU, or both. Best for a fast but more polished RT scene.
2. Simple Path Tracer
    Path tracing with CPU and OpenMP reference options.
3. Procedural Path Tracer
   GPU-only heavy path tracer with more complex geometry.
0. Exit

EOF
}

while true; do
    show_menu

    read -r -p "Choose a scene [1-3] (press Enter for 1): " choice || true
    choice="${choice:-1}"

    case "$choice" in
        0)
            echo "Exiting."
            exit 0
            ;;
        1)
            binary="$(resolve_binary simple_raytracer)"
            label="Simple Ray Tracer"
            ;;
        2)
            binary="$(resolve_binary simple_path_tracer)"
            label="Simple Path Tracer"
            ;;
        3)
            binary="$(resolve_binary procedural_path_tracer)"
            label="Procedural Path Tracer"
            ;;
        *)
            echo "Invalid choice. Please select 0, 1, 2, or 3."
            echo
            continue
            ;;
    esac

    if [[ -z "${binary:-}" ]]; then
        echo "Binary not found for $label."
        echo "Build the project first with make or make run."
        exit 1
    fi

    echo "Launching $label from $binary"
    echo
    "$repo_root/$binary"
    exit 0
done
