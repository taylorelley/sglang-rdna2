# sglang-rdna2

This fork of [SGLang](https://github.com/sgl-project/sglang) targets AMD RDNA2 Radeon GPUs: gfx1030 and gfx1031, for example the RX 6700 XT. It is kept as a **thin patch layer** on top of upstream, so it can be rebased often.

## Hardware facts that shape the design

| | RDNA2 (gfx103x) | CDNA (MI300, gfx942) |
|---|---|---|
| Wavefront | 32 | 64 |
| Matrix cores | none (no MFMA/WMMA) | MFMA |
| FP8 | none | FNUZ |
| aiter / CK kernels | unsupported | primary path |
| RX 6700 XT memory | 12 GB, ~384 GB/s | 192 GB |

Decode on RDNA2 is limited by memory bandwidth, so int4 weights are the main lever for speed and for fitting 7B-14B models.

## Quick start

Docker:

```bash
docker build -f docker/rocm-rdna2.Dockerfile --build-arg GPU_ARCH=gfx1031 -t sglang-rocm:rdna2 .
docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host --shm-size 8g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface sglang-rocm:rdna2 bash
python3 scripts/rdna2/check_env.py
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0
```

Native venv (Python 3.12 and Rust/cargo required):

```bash
GPU_ARCH=gfx1031 scripts/rdna2/install_native.sh .venv-rdna2
source .venv-rdna2/bin/activate
python scripts/rdna2/check_env.py
```

ROCm, torch 2.12 and triton come from AMD's TheRock wheel index (`https://stable.repo.amd.com/rocm/whl-next/`). It publishes native `rocm-sdk-device-gfx1031` and `amd-torch-device-gfx1031` payloads, so `HSA_OVERRIDE_GFX_VERSION` is not needed. If you build for gfx1030 and run on a gfx1031 card, set `HSA_OVERRIDE_GFX_VERSION=10.3.0` as a fallback.

## What the patch layer changes

| File | Change |
|---|---|
| `python/sglang/srt/utils/common.py` | `is_gfx103_supported()`, `is_aiter_installed()` |
| `python/sglang/srt/runtime_context.py` | `is_rdna2` platform fact |
| `python/sglang/srt/arg_groups/model_override_base.py` | default attention backend `triton` on RDNA2 |
| `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `quark/schemes/quark_w4a4_mxfp4{,_moe}.py` | aiter imports made optional |
| `docker/rocm-rdna2.Dockerfile`, `docker/patches/sgl-kernel-rdna2.sh` | image; sgl-kernel arch gate and wave32 `WARP_SIZE` |
| `scripts/rdna2/*` | native install, requirement flattening, environment check |
| `docs/docs/hardware-platforms/amd_gpu.mdx` | RDNA2 section |

The sgl-kernel change is applied as a build-time patch, not committed to the source tree, so upstream edits to `setup_rocm.py` show up as a loud build failure instead of a rebase conflict.

## Feature status

| Feature | Status |
|---|---|
| FP16/BF16 dense models, Triton attention | Phase 1-3: needs validation on the card |
| CUDA (HIP) graphs | to validate; the fallback is `--disable-cuda-graph` |
| AWQ int4 | Triton dequant path exists upstream; fused W4A16 GEMV planned |
| GPTQ int4 | to check: HIP kernel path |
| GGUF | not yet: upstream only builds the ggml kernels for CUDA/MUSA |
| FP8 KV cache (storage only) | to validate |
| DeepSeek/MLA aiter paths, MXFP4, FP8 GEMM | not supported (CDNA-only) |

## Roadmap

1. **Detection and defaults.** Done.
2. **Image and native install.** Done; needs a first build on hardware.
3. **Correctness on the card.** Run the Triton attention tests and fix out-of-bounds masked loads only where tests fail. Tune wave32 Triton configs. Validate HIP graphs, and compare `triton` with `torch_native`.
4. **Quantized models.** AWQ/GPTQ fused W4A16 GEMV, GGUF kernels in `setup_rocm.py`, FP8 KV storage, optionally TurboQuant-style KV.
5. **Profiling-driven wave32 kernels** (RMSNorm, RoPE, SiLU) only where a profile shows them hot.

The reference fork [carlosfundora/sglang-1-bit-turbo](https://github.com/carlosfundora/sglang-1-bit-turbo) documents the Triton out-of-bounds faults it hit on RDNA2, and a GGUF MMVQ crash at batch > 1. Port fixes from it only when a test on the card reproduces the problem.

## Validating on the card

Measure accuracy with [sgl-eval](https://github.com/sgl-project/sgl-eval), run from its own venv:

```bash
sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples 200
```

Measure throughput:

```bash
python3 -m sglang.bench_serving --backend sglang --num-prompts 64 --max-concurrency 4
```

Report the model, quantization, flags and sgl-eval settings with every number.

## Staying current with upstream

```bash
git remote add upstream https://github.com/sgl-project/sglang.git   # once
git fetch upstream main
git rebase upstream/main      # conflicts should be limited to the files above
python3 -m pytest -q test/registered/unit/test_model_overrides.py -k rdna2
docker build -f docker/rocm-rdna2.Dockerfile --build-arg GPU_ARCH=gfx1031 .
```

If `sgl-kernel-rdna2.sh` fails with "expected line not found", upstream changed `setup_rocm.py` or `include/utils.h`. Update the patch strings to match the new upstream text.
