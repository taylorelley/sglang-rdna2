# SGLang for AMD RDNA2 Radeon GPUs (gfx103x: RX 6700 XT / 6750 XT = gfx1031,
# RX 6800 / 6800 XT / 6900 XT = gfx1030).
#
# Standalone, like docker/rocm-gfx1151.Dockerfile: docker/rocm.Dockerfile targets
# CDNA and builds components (aiter prebuilt kernels, MORI, Mooncake, CK) that
# assume wave64 and MFMA. RDNA2 is wave32 with no matrix cores and no FP8.
#
# ROCm comes from AMD's TheRock wheels, which publish native gfx103x device
# payloads, so HSA_OVERRIDE_GFX_VERSION is not needed. The torch/triton pins match
# the rocm1010 stack in docker/rocm.Dockerfile.
#
# Build (pick the arch of your card; one image per arch):
#   docker build -f docker/rocm-rdna2.Dockerfile --build-arg GPU_ARCH=gfx1031 -t sglang-rocm:rdna2 .
#
# Run:
#   docker run -it --rm \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --group-add render \
#     --security-opt seccomp=unconfined \
#     --ipc=host --shm-size 8g \
#     -p 30000:30000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     sglang-rocm:rdna2 \
#     python3 -m sglang.launch_server --model-path <model> --host 0.0.0.0
#
# First command to run inside the container: python3 scripts/rdna2/check_env.py

ARG BASE_IMAGE="ubuntu:24.04"

# =============================================================================
# Stage 1: TheRock ROCm SDK + PyTorch + Triton for one gfx103x target.
# =============================================================================
FROM ${BASE_IMAGE} AS rocm-torch

ARG GPU_ARCH=gfx1031
ARG ROCM_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
ARG ROCM_SDK_VERSION="10.1.0"
ARG TORCH_VERSION="2.12.0"
# No amd-torchvision-device-gfx103x wheel exists for torchvision 0.27, so
# torchvision runs its ops on CPU; nothing on the LLM serving path uses them.
ARG TORCHVISION_VERSION="0.27.0"
ARG TRITON_VERSION="3.8.0+git669b31ac"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libnuma-dev \
        libstdc++-12-dev \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

RUN set -eux; \
    case "${GPU_ARCH}" in \
      gfx103*) ;; \
      *) echo "GPU_ARCH=${GPU_ARCH} is not an RDNA2 (gfx103x) target"; exit 1 ;; \
    esac; \
    python3 -m pip install --no-cache-dir \
        --index-url ${ROCM_INDEX_URL} \
        "rocm-sdk-core==${ROCM_SDK_VERSION}" \
        "rocm-sdk-libraries==${ROCM_SDK_VERSION}" \
        "rocm-sdk-devel==${ROCM_SDK_VERSION}" \
        "rocm-sdk-device-${GPU_ARCH}==${ROCM_SDK_VERSION}" \
        "torch==${TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "amd-torch-device-${GPU_ARCH}==${TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "triton==${TRITON_VERSION}.rocm${ROCM_SDK_VERSION}"; \
    python3 -m pip show "rocm-sdk-device-${GPU_ARCH}" "amd-torch-device-${GPU_ARCH}" >/dev/null

RUN rocm-sdk init && rocm-sdk targets

# Same duplicate-libamd_smi collapse as docker/rocm.Dockerfile (rocm1000-base):
# two copies in one process make torch.cuda.device_count() return 0.
RUN set -eux; \
    SP="$VIRTUAL_ENV/lib/python3.12/site-packages"; \
    CORE=$(ls "$SP"/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1); \
    DEVEL="$SP/_rocm_sdk_devel/lib/libamd_smi.so"; \
    if [ -n "${CORE}" ] && [ -e "${DEVEL}" ] && [ ! -L "${DEVEL}" ]; then \
      ln -sf "${CORE}" "${DEVEL}"; \
    fi

ENV ROCM_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_devel
ENV ROCM_PATH=$ROCM_HOME
ENV CPATH=$ROCM_HOME/include
ENV LIBRARY_PATH=$ROCM_HOME/lib
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib
ENV PATH="$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH"
RUN mkdir -p /usr/lib64 && ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so \
    && ln -s ${ROCM_HOME} /opt/rocm

ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}

# Keep later pip installs from replacing the ROCm torch with a PyPI CUDA build.
RUN mkdir -p /etc/sglang/constraints \
    && python3 -m pip freeze | grep -iE '^(torch|torchvision|triton)==' \
         > /etc/sglang/constraints/torch-rocm.txt \
    && cat /etc/sglang/constraints/torch-rocm.txt
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"

RUN python3 -c "import torch; print('torch', torch.__version__); assert torch.version.hip" \
    && command -v hipcc

# =============================================================================
# Stage 2: SGLang.
# =============================================================================
FROM rocm-torch AS sglang

ARG GPU_ARCH=gfx1031
# Set to 0 to skip the AOT kernels and run on the Triton/torch paths only.
ARG BUILD_SGL_KERNEL=1
ARG MAX_JOBS=8

WORKDIR /sgl-workspace

# setuptools-rust builds the sglang-mm extension during the pip install below.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && rustc --version
ENV CARGO_BUILD_JOBS=8

COPY . /sgl-workspace/sglang

RUN cd /sgl-workspace/sglang \
    && rm -f python/pyproject.toml \
    && mv python/pyproject_other.toml python/pyproject.toml

RUN cd /sgl-workspace/sglang/python/sglang/kernels/aot \
    && if [ "${BUILD_SGL_KERNEL}" = "1" ]; then \
         rm -f pyproject.toml \
         && mv pyproject_rocm.toml pyproject.toml \
         && sh /sgl-workspace/sglang/docker/patches/sgl-kernel-rdna2.sh setup_rocm.py \
         && AMDGPU_TARGET=${GPU_ARCH} MAX_JOBS=${MAX_JOBS} python3 setup_rocm.py install; \
       else \
         echo "Skipping sgl-kernel build (BUILD_SGL_KERNEL=0)"; \
       fi

# Install the srt_hip_rocm101 dependency set flattened, as docker/rocm-gfx1151.Dockerfile
# does, minus the CDNA-only packages (petit_kernel, wave-lang, monokernels).
RUN cd /sgl-workspace/sglang \
    && python3 scripts/rdna2/flatten_requirements.py python/pyproject.toml \
         > /tmp/rdna2-requirements.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/rdna2-requirements.txt \
    && python3 -m pip install --no-cache-dir --no-deps -e python

# No aiter: its CK attention and RMSNorm kernels assume wave64/CDNA, and
# sglang imports it lazily on this path (see is_aiter_installed()).
ENV SGLANG_USE_AITER=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
