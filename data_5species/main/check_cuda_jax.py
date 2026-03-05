#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_cuda_jax.py — JAX CUDA 環境の診断

Usage:
    python check_cuda_jax.py
    JAX_PLATFORMS=cuda python check_cuda_jax.py
"""
from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str]:
    """Run command and return (returncode, output)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, str(e)


def main() -> int:
    print("=== JAX CUDA 環境診断 ===\n")

    # 1. nvidia-smi
    print("1. nvidia-smi:")
    rc, out = run(["nvidia-smi", "-L"])
    if rc == 0:
        for line in out.strip().split("\n"):
            print(f"   {line}")
    else:
        print("   [NG] nvidia-smi が実行できません（GPU なし or ドライバ未導入）")

    # 2. 環境変数
    print("\n2. 環境変数:")
    for k in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "LD_LIBRARY_PATH"):
        v = os.environ.get(k, "(未設定)")
        if k == "LD_LIBRARY_PATH" and len(str(v)) > 80:
            v = str(v)[:80] + "..."
        print(f"   {k}={v}")

    # 3. JAX デバイス
    print("\n3. JAX devices:")
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cuda")
    code = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; d=jax.devices(); print('  ', d); "
            "gpu=any('cuda' in str(x).lower() or 'gpu' in str(x).lower() for x in d); "
            "print('  GPU 利用:', 'OK' if gpu else 'NG (CPU フォールバック)')",
        ],
        env=env,
    ).returncode
    if code != 0:
        print("   [NG] JAX のインポートに失敗")

    # 4. jax-cuda12-plugin
    print("\n4. jax-cuda12-plugin:")
    try:
        import jax_cuda12_plugin  # noqa: F401

        print("   インストール済み")
    except ImportError:
        print("   未インストール → pip install jax[cuda12] を実行")

    # 5. 推奨
    print("\n5. CUDA が使えない場合の対処:")
    print("   - pip install jax[cuda12] で再インストール")
    print("   - JAX_PLATFORMS=cuda を export（ROCM 誤検出回避）")
    print("   - LD_LIBRARY_PATH にシステム CUDA が含まれる場合、unset して pip の nvidia-* を優先")
    print("   - nvidia-driver と jax-cuda12-plugin の CUDA バージョン互換性を確認")

    return 0


if __name__ == "__main__":
    sys.exit(main())
