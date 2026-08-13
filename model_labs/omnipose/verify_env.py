"""Assert the pm-omnipose environment is genuinely GPU-capable, and record it.

`torch.cuda.is_available()` is not proof on this workstation. The GPU is an
RTX 5070 Ti Laptop -- Blackwell, compute capability 12.0 (sm_120). A torch wheel
built only for sm_90 and below still reports `is_available() == True` and then
dies at the first real kernel launch with "no kernel image is available for
execution on the device". So this verifier launches an actual matmul.

It also guards the install-order trap that already bit this environment once:
`pip install omnipose` pulls a CPU-only torch and will silently replace a
correct CUDA build (observed: 2.11.0+cu128 -> 2.13.0+cpu, exit code 0).

    python model_labs/omnipose/verify_env.py [--json out.json]

Exits non-zero with a specific message on any failure, so it is usable as a
precondition in a training script rather than only as a human check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys


def verify() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "FAIL: torch reports no CUDA device.\n"
            f"      installed torch = {torch.__version__}\n"
            "      A '+cpu' build here almost certainly means omnipose's resolver\n"
            "      replaced the CUDA build. Reinstall:\n"
            "        pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 \\\n"
            "          --index-url https://download.pytorch.org/whl/cu128")

    capability = torch.cuda.get_device_capability(0)
    target = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list()
    # An exact match is not what CUDA requires, and demanding one raised a false
    # failure on klone's L40S: torch 2.11.0+cu128 ships
    # ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120'] with no sm_89, yet
    # matmul and conv2d both run there. Cubins are binary compatible within a
    # major compute capability from a lower or equal minor upward, so sm_86
    # kernels execute on an sm_89 device. Accept that, and let the real kernel
    # launch below be the thing that actually decides.
    compatible = [
        arch for arch in arch_list
        if arch.startswith("sm_")
        and arch[3:].isdigit()
        and int(arch[3:-1] or 0) == capability[0]
        and int(arch[3:][-1]) <= capability[1]
    ]
    if target not in arch_list and not compatible:
        raise SystemExit(
            f"FAIL: this torch has no kernels usable on {target}.\n"
            f"      device     = {torch.cuda.get_device_name(0)} ({target})\n"
            f"      arch_list  = {arch_list}\n"
            f"      No sm_{capability[0]}x kernels at or below minor "
            f"{capability[1]} are present, so nothing can run here.\n"
            "      Install a build whose index covers this architecture.")

    # A real kernel launch. This is the check that a wrong-arch build fails.
    tensor = torch.randn(512, 512, device="cuda")
    value = float((tensor @ tensor).sum().item())
    if value != value:                                   # NaN
        raise SystemExit("FAIL: GPU matmul produced NaN")
    torch.cuda.synchronize()

    import cellpose_omni
    import numpy
    import omnipose

    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": __import__("torchvision").__version__,
        "numpy": numpy.__version__,
        "omnipose": omnipose.__version__,
        "cellpose_omni": cellpose_omni.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": f"{capability[0]}.{capability[1]}",
        "arch_list": arch_list,
        "gpu_kernel_launch_verified": True,
        "isolation": "pm-omnipose; Conversion_Efficiency/cpenv untouched",
    }
    record["environment_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write the environment record here")
    args = parser.parse_args(argv)

    record = verify()
    print(json.dumps(record, indent=2))
    print("\nOK: pm-omnipose is GPU-capable (real kernel launch on "
          f"{record['device']}, sm_{record['capability'].replace('.', '')}).")
    if args.json:
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
