#!/usr/bin/env bash
# Build and install pywhispercpp with the CUDA backend, then patch the
# auditwheel-bundled libcuda so it falls back to the system driver.
#
# Background: the upstream pywhispercpp wheel build runs auditwheel, which
# vendors a copy of libcuda.so into pywhispercpp.libs/. That copy cannot
# talk to the kernel driver and aborts with CUDA_ERROR_NOT_INITIALIZED at
# cuDeviceGet. The system libcuda.so.1 must be used instead.
#
# Usage:
#   ./scripts/install-pywhispercpp-cuda.sh [venv_path]
#
# Defaults to ./venv. Requires: nvidia driver + CUDA toolkit installed
# (nvcc on PATH or CUDA_HOME set), and either `uv` or the venv's `pip`.

set -euo pipefail

VENV_DIR="${1:-venv}"
PYWHISPERCPP_REF="${PYWHISPERCPP_REF:-git+https://github.com/absadiki/pywhispercpp}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "error: no python at $VENV_DIR/bin/python — pass the venv path as arg 1" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
unset PYTHONPATH

# Locate CUDA toolkit
if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [[ -d /usr/local/cuda ]]; then
        CUDA_HOME=/usr/local/cuda
    else
        echo "error: CUDA toolkit not found. Install it or set CUDA_HOME." >&2
        exit 1
    fi
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
echo "[1/4] Using CUDA_HOME=$CUDA_HOME"

# Locate system libcuda.so.1 (must come from the kernel driver, never a wheel)
SYSTEM_LIBCUDA="$(ldconfig -p | awk '/libcuda\.so\.1/ {print $NF; exit}')"
if [[ -z "$SYSTEM_LIBCUDA" || ! -e "$SYSTEM_LIBCUDA" ]]; then
    echo "error: system libcuda.so.1 not found — install the NVIDIA driver." >&2
    exit 1
fi
echo "[2/4] System libcuda: $SYSTEM_LIBCUDA"

# Pick installer
if command -v uv >/dev/null; then
    PIP_INSTALL=(uv pip install --force-reinstall --no-cache --no-deps
                 --no-build-isolation-package pywhispercpp)
else
    PIP_INSTALL=("$VENV_DIR/bin/python" -m pip install --force-reinstall
                 --no-cache-dir --no-deps)
fi

echo "[3/4] Building pywhispercpp with GGML_CUDA=1 (this can take several minutes)..."
GGML_CUDA=1 CMAKE_ARGS="-DGGML_CUDA=ON" "${PIP_INSTALL[@]}" "$PYWHISPERCPP_REF"

# Patch the bundled libcuda
LIBS_DIR="$VENV_DIR/lib/python$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pywhispercpp.libs"
BUNDLED_LIBCUDA="$(find "$LIBS_DIR" -maxdepth 1 -name 'libcuda-*.so.*' -print -quit || true)"
if [[ -n "$BUNDLED_LIBCUDA" ]]; then
    echo "[4/4] Replacing bundled $(basename "$BUNDLED_LIBCUDA") with symlink to system libcuda"
    rm -f "$BUNDLED_LIBCUDA"
    ln -s "$SYSTEM_LIBCUDA" "$BUNDLED_LIBCUDA"
else
    echo "[4/4] No bundled libcuda found — nothing to patch"
fi

echo
echo "Verifying..."
"$VENV_DIR/bin/python" - <<'PY'
from pywhispercpp.model import Model  # noqa: F401
import ctypes, glob, os
libs = glob.glob(os.path.join(os.path.dirname(__import__("pywhispercpp").__file__), "..", "pywhispercpp.libs", "libggml-cuda-*.so"))
print("ggml-cuda lib:", libs[0] if libs else "MISSING")
print("OK: pywhispercpp imports cleanly with CUDA backend")
PY
