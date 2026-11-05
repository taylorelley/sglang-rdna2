#!/bin/sh
# Teach python/sglang/kernels/aot/setup_rocm.py to build for RDNA2 (gfx103x).
# Applied by docker/rocm-rdna2.Dockerfile and scripts/rdna2/install_native.sh
# against a scratch copy or an image-local tree, so the repo stays rebase-clean.
#
# Same two changes as sgl-kernel-gfx1151.sh, for the same reasons:
#   1. Lift the {gfx942, gfx950, gfx1250} allowlist, which otherwise sys.exit(1)s.
#      The 40KB non-gfx942 TopK dynamic-LDS budget fits RDNA2's 64KB workgroup LDS.
#   2. Pin WARP_SIZE to 32 on both the host and device compiler passes.
#      Without this the host pass sees 64 and the wave32 device pass sees 32,
#      and the MoE TopK launches exceed their __launch_bounds__.
#
# Every edit greps for the expected upstream text first and fails otherwise.
set -e

FILE="${1:?usage: sgl-kernel-rdna2.sh <path to setup_rocm.py>}"
UTILS="$(dirname "${FILE}")/include/utils.h"

GATE_OLD='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:'
GATE_NEW='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"] and not amdgpu_target.startswith("gfx103"):'

FLAGS_OLD='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",'
FLAGS_NEW='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    # RDNA2 is wave32; pin both compiler passes to it (see utils.h).
    *(["-DSGL_ROCM_WARP_SIZE=32"] if amdgpu_target.startswith("gfx103") else []),'

WARP_OLD='#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'
WARP_NEW='#if defined(SGL_ROCM_WARP_SIZE)
#define WARP_SIZE SGL_ROCM_WARP_SIZE
#elif defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'

for pattern in "${GATE_OLD}" "${FLAGS_OLD}"; do
    if ! grep -qF "${pattern}" "${FILE}"; then
        echo "ERROR: expected line not found in ${FILE}:" >&2
        echo "  ${pattern}" >&2
        echo "setup_rocm.py changed upstream; re-check this patch before building." >&2
        exit 1
    fi
done

if ! grep -qF "${WARP_OLD}" "${UTILS}"; then
    echo "ERROR: expected WARP_SIZE block not found in ${UTILS}." >&2
    echo "utils.h changed upstream; re-check the wave32 fix before building." >&2
    exit 1
fi

python3 - "${FILE}" "${UTILS}" "${GATE_OLD}" "${GATE_NEW}" "${FLAGS_OLD}" "${FLAGS_NEW}" "${WARP_OLD}" "${WARP_NEW}" <<'PY'
import sys

setup, utils, gate_old, gate_new, flags_old, flags_new, warp_old, warp_new = sys.argv[1:9]
for path, edits in (
    (setup, ((gate_old, gate_new), (flags_old, flags_new))),
    (utils, ((warp_old, warp_new),)),
):
    with open(path) as f:
        src = f.read()
    for old, new in edits:
        src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
PY

echo "Patched ${FILE} and ${UTILS} for RDNA2 (wave32):"
grep -nF -e "gfx103" "${FILE}"
grep -nF "SGL_ROCM_WARP_SIZE" "${UTILS}"
