#!/usr/bin/env bash
# Install this SGLang fork natively (no Docker) for an RDNA2 GPU, into a venv.
# Mirrors docker/rocm-rdna2.Dockerfile; keep the pins below in step with it.
#
#   GPU_ARCH=gfx1031 scripts/rdna2/install_native.sh [venv-dir]
#
# Needs: Python 3.12, the amdgpu kernel driver, and membership in the video and
# render groups. No system ROCm install is needed; the SDK comes from wheels.
set -euo pipefail

GPU_ARCH="${GPU_ARCH:-gfx1031}"
VENV="${1:-.venv-rdna2}"
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://stable.repo.amd.com/rocm/whl-next/}"
ROCM_SDK_VERSION="10.1.0"
TORCH_VERSION="2.12.0"
TORCHVISION_VERSION="0.27.0"
TRITON_VERSION="3.8.0+git669b31ac"
BUILD_SGL_KERNEL="${BUILD_SGL_KERNEL:-1}"
MAX_JOBS="${MAX_JOBS:-8}"

case "${GPU_ARCH}" in
  gfx103*) ;;
  *) echo "GPU_ARCH=${GPU_ARCH} is not an RDNA2 (gfx103x) target" >&2; exit 1 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3.12}"

"${PY}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools setuptools_scm wheel

python -m pip install --index-url "${ROCM_INDEX_URL}" \
    "rocm-sdk-core==${ROCM_SDK_VERSION}" \
    "rocm-sdk-libraries==${ROCM_SDK_VERSION}" \
    "rocm-sdk-devel==${ROCM_SDK_VERSION}" \
    "rocm-sdk-device-${GPU_ARCH}==${ROCM_SDK_VERSION}" \
    "torch==${TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
    "amd-torch-device-${GPU_ARCH}==${TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}" \
    "triton==${TRITON_VERSION}.rocm${ROCM_SDK_VERSION}"
rocm-sdk init

SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export ROCM_HOME="${SP}/_rocm_sdk_devel"
export ROCM_PATH="${ROCM_HOME}"
export PATH="${ROCM_HOME}/llvm/bin:${ROCM_HOME}/bin:${PATH}"
export PYTORCH_ROCM_ARCH="${GPU_ARCH}"

# Same duplicate-libamd_smi collapse as the Dockerfile.
CORE="$(ls "${SP}"/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1 || true)"
DEVEL="${SP}/_rocm_sdk_devel/lib/libamd_smi.so"
if [ -n "${CORE}" ] && [ -e "${DEVEL}" ] && [ ! -L "${DEVEL}" ]; then
  ln -sf "${CORE}" "${DEVEL}"
fi

CONSTRAINTS="${VENV}/torch-rocm-constraints.txt"
python -m pip freeze | grep -iE '^(torch|torchvision|triton)==' > "${CONSTRAINTS}"
export PIP_CONSTRAINT="${CONSTRAINTS}"

python "${REPO}/scripts/rdna2/flatten_requirements.py" "${REPO}/python/pyproject_other.toml" \
    > "${VENV}/rdna2-requirements.txt"
python -m pip install -r "${VENV}/rdna2-requirements.txt"

# No build isolation: python/pyproject.toml's build-system pins the CUDA torch,
# which an isolated build env would download. setuptools-rust needs cargo.
command -v cargo >/dev/null || {
  echo "cargo not found; install Rust (https://rustup.rs) and re-run" >&2
  exit 1
}
python -m pip install "setuptools-rust>=1.11"
python -m pip install --no-deps --no-build-isolation -e "${REPO}/python"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
if [ "${BUILD_SGL_KERNEL}" = "1" ]; then
  cp -r "${REPO}/python/sglang/kernels/aot" "${BUILD_DIR}/aot"
  cd "${BUILD_DIR}/aot"
  mv pyproject_rocm.toml pyproject.toml
  sh "${REPO}/docker/patches/sgl-kernel-rdna2.sh" setup_rocm.py
  AMDGPU_TARGET="${GPU_ARCH}" MAX_JOBS="${MAX_JOBS}" python setup_rocm.py install
  cd "${REPO}"
fi

ACTIVATE_EXTRA="${VENV}/bin/activate.d-rdna2.sh"
cat > "${ACTIVATE_EXTRA}" <<EOF
export ROCM_HOME="${ROCM_HOME}"
export ROCM_PATH="${ROCM_HOME}"
export PATH="${ROCM_HOME}/llvm/bin:${ROCM_HOME}/bin:\${PATH}"
export PYTORCH_ROCM_ARCH="${GPU_ARCH}"
export SGLANG_USE_AITER=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
EOF
grep -qF "${ACTIVATE_EXTRA}" "${VENV}/bin/activate" \
  || echo "source \"${ACTIVATE_EXTRA}\"" >> "${VENV}/bin/activate"

echo
echo "Done. Next:"
echo "  source ${VENV}/bin/activate"
echo "  python ${REPO}/scripts/rdna2/check_env.py"
