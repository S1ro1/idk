#!/usr/bin/env bash
# Install dependencies required for disaggregated prefill/decode serving.
# Run this on EVERY node that will host a P (prefill) or D (decode) instance.
#
# Usage:
#   bash scripts/install_disagg_deps.sh [--nixl-version X.Y.Z]
#
# Assumptions:
#   - The project venv is at <repo>/.venv
#   - deep_ep was already built via scripts/install_python_libraries.sh
#   - NVSHMEM is at <repo>/ep_kernels_workspace/nvshmem/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- defaults ---------------------------------------------------------------
NIXL_VERSION="${NIXL_VERSION:-0.10.0}"
VENV_BIN="$REPO_ROOT/.venv/bin"
PYTHON="$VENV_BIN/python3"
# uv lives outside the venv (user-level install); fall back to pip3
UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || echo "")}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nixl-version) NIXL_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- helpers ----------------------------------------------------------------
pip_install() {
    local pkg="$1"
    local import_name="${2:-${pkg%%[=<>!]*}}"
    if "$PYTHON" -c "import ${import_name//-/_}" 2>/dev/null; then
        echo "[ok] ${import_name} already installed"
    else
        echo "[..] Installing $pkg ..."
        if [ -n "$UV_BIN" ]; then
            "$UV_BIN" pip install --python "$PYTHON" "$pkg"
        else
            "$PYTHON" -m pip install "$pkg"
        fi
    fi
}

echo "================================================================"
echo " Installing disaggregated prefill/decode dependencies"
echo " NIXL_VERSION=$NIXL_VERSION"
echo "================================================================"

# 1. NIXL — KV cache transfer via UCX/InfiniBand
#    Source: https://github.com/ai-dynamo/nixl
#    vLLM reads nixl._api.nixl_agent (NixlWrapper) at runtime.
pip_install "nixl==${NIXL_VERSION}" "nixl"

# 2. aiohttp — async HTTP client/server for the disagg proxy
pip_install "aiohttp" "aiohttp"

# 3. deep_ep — DeepEP A2A kernels (must be pre-built)
if ! "$VENV_BIN/python3" -c "import deep_ep" 2>/dev/null; then
    echo ""
    echo "ERROR: deep_ep is not installed."
    echo "  Run:  bash scripts/install_python_libraries.sh"
    echo "  Or:   uv pip install ep_kernels_workspace/dist/deep-ep-*.whl"
    exit 1
fi
echo "[ok] deep_ep"

# 4. Verify NVSHMEM shared library (required at runtime for DeepEP low-latency kernel)
NVSHMEM_LIB_DIR="$REPO_ROOT/ep_kernels_workspace/nvshmem/lib"
if [ -f "$NVSHMEM_LIB_DIR/libnvshmem_host.so.3" ]; then
    echo "[ok] NVSHMEM lib at $NVSHMEM_LIB_DIR"
else
    echo ""
    echo "WARNING: NVSHMEM lib not found at $NVSHMEM_LIB_DIR"
    echo "  The deepep_low_latency backend will fail without it."
    echo "  Run:  bash scripts/install_python_libraries.sh"
fi

# 5. Confirm NIXL import works end-to-end
echo ""
echo "=== Verifying imports ==="
"$PYTHON" -c "
import nixl
import nixl._api
print('[ok] nixl', nixl.__version__ if hasattr(nixl, '__version__') else '(version n/a)')
import deep_ep; print('[ok] deep_ep')
import aiohttp; print('[ok] aiohttp', aiohttp.__version__)
"

echo ""
echo "================================================================"
echo " All disagg dependencies ready."
echo " LD_LIBRARY_PATH must include: $NVSHMEM_LIB_DIR"
echo "================================================================"
