#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, platform, shutil, subprocess
from textwrap import indent


def run(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return -1, str(e)


def section(title):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def main():
    section("System")
    print(f"OS          : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"Machine     : {platform.machine()}  (arm64=Apple Silicon, x86_64=Intel)")

    # -------- PyTorch import & basic CPU test --------
    section("PyTorch")
    try:
        import torch
        print(f"torch       : {torch.__version__}")
        print(f"DEBUG dtype : default={torch.get_default_dtype()}")
    except Exception as e:
        print("PyTorch IMPORT FAILED ❌")
        print(indent(str(e), "  "))
        return 1

    # CPU forward/backward sanity test
    try:
        x = torch.randn(2, 3, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        y = x @ w
        loss = y.pow(2).mean()
        loss.backward()
        print("CPU forward/backward: OK ✅")
    except Exception as e:
        print("CPU forward/backward FAILED ❌")
        print(indent(str(e), "  "))
        return 2

    # -------- CUDA check (rare on macOS; Intel+NVIDIA only) --------
    section("CUDA")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        cuda_ver = getattr(torch.version, "cuda", None)
        print(f"torch.cuda.is_available(): {has_cuda}")
        print(f"torch.version.cuda       : {cuda_ver}")

        # nvidia-smi / nvcc probes (optional)
        if shutil.which("nvidia-smi"):
            rc, out = run(["nvidia-smi", "-L"])
            print("nvidia-smi -L:")
            print(indent(out, "  "))
        else:
            print("nvidia-smi               : not found")

        if shutil.which("nvcc"):
            rc, out = run(["nvcc", "--version"])
            print("nvcc --version:")
            print(indent(out, "  "))
        else:
            print("nvcc                     : not found")

        if has_cuda:
            try:
                devs = torch.cuda.device_count()
                print(f"CUDA devices             : {devs}")
                for i in range(devs):
                    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
                # tiny CUDA compute
                device = torch.device("cuda:0")
                a = torch.randn(1024, 1024, device=device)
                b = torch.randn(1024, 1024, device=device)
                c = a @ b
                torch.cuda.synchronize()
                print("CUDA matmul              : OK ✅")
            except Exception as e:
                print("CUDA test FAILED ❌")
                print(indent(str(e), "  "))
        else:
            print("CUDA not available on this machine (正常：Apple Silicon/大多数 Mac 不支持 CUDA)。")
    except Exception as e:
        print("CUDA check error ❌")
        print(indent(str(e), "  "))

    # -------- Apple Metal (MPS) check --------
    section("Apple MPS (Metal)")
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        built_mps = getattr(torch.backends.mps, "is_built", lambda: False)()
        print(f"torch.backends.mps.is_built()   : {built_mps}")
        print(f"torch.backends.mps.is_available(): {has_mps}")
        if has_mps:
            try:
                device = torch.device("mps")
                a = torch.randn(2048, 2048, device=device)
                b = torch.randn(2048, 2048, device=device)
                c = a @ b
                # 简单反传测试
                x = torch.randn(128, 128, device=device, requires_grad=True)
                y = (x @ torch.randn(128, 64, device=device)).pow(2).mean()
                y.backward()
                print("MPS forward/backward      : OK ✅")
            except Exception as e:
                print("MPS test FAILED ❌")
                print(indent(str(e), "  "))
        else:
            print("MPS 不可用：需 macOS 12.3+ 且 PyTorch 构建支持 MPS；Intel Mac 无 MPS。")
    except Exception as e:
        print("MPS check error ❌")
        print(indent(str(e), "  "))

    section("Summary")

    return 0


if __name__ == "__main__":
    sys.exit(main())
