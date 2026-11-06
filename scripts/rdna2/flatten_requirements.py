"""Print the RDNA2 runtime requirements from a pyproject, one per line.

Expands the self-referencing `sglang[...]` extras of `srt_hip_rocm101` (pip's
resolver loops on that cycle from an editable checkout) and drops packages that
are CDNA-only or must come from the ROCm wheel index instead of PyPI.

Usage: python3 scripts/rdna2/flatten_requirements.py python/pyproject_other.toml
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT_EXTRA = "srt_hip_rocm101"
# torch/triton come from the ROCm index; the rest ship only CDNA kernels.
EXCLUDED = {"torch", "triton", "petit_kernel", "wave-lang", "sgl-amd-mono"}

_SELF_REF = re.compile(r"^sglang\[([^\]]+)\]$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _is_bare(req: str) -> bool:
    return _NAME.fullmatch(req.strip()) is not None


def _expand(extras: dict, group: str, seen: set) -> list[str]:
    if group in seen:
        return []
    seen.add(group)
    out = []
    for req in extras[group]:
        ref = _SELF_REF.match(req.strip())
        if ref:
            for sub in ref.group(1).split(","):
                out.extend(_expand(extras, sub.strip(), seen))
        else:
            out.append(req)
    return out


def main(path: str) -> None:
    project = tomllib.loads(Path(path).read_text())["project"]
    reqs = list(project["dependencies"])
    reqs.extend(_expand(project["optional-dependencies"], ROOT_EXTRA, set()))
    excluded = {_canonical(e) for e in EXCLUDED}
    chosen: dict[str, str] = {}
    for req in reqs:
        name = _canonical(_NAME.match(req).group(0))
        if name in excluded:
            continue
        # A versioned spec (srt_hip_rocm101's compressed-tensors pin) beats a bare name.
        if name not in chosen or (_is_bare(chosen[name]) and not _is_bare(req)):
            chosen[name] = req
    print("\n".join(chosen.values()))


if __name__ == "__main__":
    main(sys.argv[1])
