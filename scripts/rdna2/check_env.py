"""Smoke-test an RDNA2 (gfx103x) ROCm environment before serving with SGLang.

Run on the machine with the GPU, inside the image or the native venv:
    python3 scripts/rdna2/check_env.py

Each check prints PASS/WARN/FAIL; the exit code is non-zero if any check FAILs.
"""

import importlib.util
import os
import sys
import time

_results: list[tuple[str, str, str]] = []


def _record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def _check_torch():
    try:
        import torch
    except ImportError as e:
        _record("FAIL", "import torch", str(e))
        return None
    if not torch.version.hip:
        _record("FAIL", "torch is a ROCm build", f"torch {torch.__version__}")
        return None
    _record("PASS", "torch ROCm build", f"{torch.__version__}, HIP {torch.version.hip}")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        _record(
            "FAIL",
            "GPU visible",
            "no device; check --device=/dev/kfd --device=/dev/dri and the video/render groups",
        )
        return None
    return torch


def _check_arch(torch) -> None:
    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    detail = f"{props.name}, {arch}, {props.total_memory / 2**30:.1f} GiB"
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        detail += f", HSA_OVERRIDE_GFX_VERSION={override}"
    if arch.startswith("gfx103"):
        _record("PASS", "RDNA2 device", detail)
    else:
        _record("WARN", "RDNA2 device", f"{detail} (not gfx103x)")
    built = os.environ.get("PYTORCH_ROCM_ARCH")
    if built and arch not in built.split(";") and not override:
        _record(
            "WARN",
            "arch matches build",
            f"PYTORCH_ROCM_ARCH={built} but device is {arch}; rebuild with GPU_ARCH={arch}",
        )


def _check_gemm(torch) -> None:
    # Exercises rocBLAS/hipBLASLt, which need a Tensile library for this arch.
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    try:
        ref = a.float().cpu() @ b.float().cpu()
        out = a @ b
        torch.cuda.synchronize()
        err = (out.float().cpu() - ref).abs().max().item()
        iters = 20
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        tflops = 2 * 2048**3 * iters / (time.perf_counter() - start) / 1e12
    except RuntimeError as e:
        _record("FAIL", "fp16 GEMM", str(e).splitlines()[0])
        return
    status = "PASS" if err < 1.0 else "FAIL"
    _record(status, "fp16 GEMM", f"max abs err {err:.3f}, {tflops:.1f} TFLOPS")


def _check_triton(torch) -> None:
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        _record("FAIL", "import triton", str(e))
        return

    @triton.jit
    def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(
            out_ptr + offs,
            tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
            mask=mask,
        )

    n = 100_003
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    try:
        _add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - report any failure as a FAIL line
        _record("FAIL", "triton kernel", f"{type(e).__name__}: {e}")
        return
    ok = torch.allclose(out, x + y)
    target = triton.runtime.driver.active.get_current_target()
    _record(
        "PASS" if ok else "FAIL",
        "triton kernel",
        f"triton {triton.__version__}, target {target.arch} warp {target.warp_size}",
    )


def _check_sglang() -> None:
    try:
        from sglang.srt.utils.common import is_aiter_installed, is_gfx103_supported
    except ImportError as e:
        _record("FAIL", "import sglang", str(e))
        return
    _record(
        "PASS" if is_gfx103_supported() else "WARN",
        "sglang RDNA2 detection",
        f"is_gfx103_supported()={is_gfx103_supported()}",
    )
    if is_aiter_installed():
        _record("WARN", "aiter", "installed; keep SGLANG_USE_AITER=0 on RDNA2")
    if os.environ.get("SGLANG_USE_AITER", "0").lower() in ("1", "true"):
        _record(
            "FAIL", "SGLANG_USE_AITER", "set; aiter kernels assume CDNA (wave64/MFMA)"
        )
    try:
        import sgl_kernel  # noqa: F401

        _record("PASS", "sgl_kernel", "AOT kernels importable")
    except ImportError as e:
        _record(
            "FAIL", "sgl_kernel", f"not importable ({e}); required by HIP activations"
        )
    if importlib.util.find_spec("sglang.srt.layers.quantization") is None:
        _record("FAIL", "sglang.srt.layers.quantization", "missing")
        return
    try:
        import sglang.srt.layers.quantization  # noqa: F401
    except Exception as e:  # noqa: BLE001 - report any failure as a FAIL line
        _record(
            "FAIL", "import sglang.srt.layers.quantization", f"{type(e).__name__}: {e}"
        )
        return
    _record("PASS", "import sglang.srt.layers.quantization")


def main() -> int:
    torch = _check_torch()
    if torch is not None:
        _check_arch(torch)
        _check_gemm(torch)
        _check_triton(torch)
        _check_sglang()
    failed = [r for r in _results if r[0] == "FAIL"]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks without FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
