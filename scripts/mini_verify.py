#!/usr/bin/env python3
"""M2 mini 验证：完全复用 train_centralized_prm.py 的代码和环境。

不自己 import 任何 ML 库，只负责：
1. 调用 python scripts/train_centralized_prm.py --config configs/m2_smoke.yaml
2. 检查返回码和 checkpoint 文件是否生成

Usage:
    python scripts/mini_verify.py
"""

import subprocess
import sys
from pathlib import Path


CONFIG = "configs/m2_smoke.yaml"
CKPT_DIR = Path("./experiments/M2_centralized_prm/results/checkpoints_smoke")


def main() -> int:
    print("=" * 50)
    print("M2 mini 验证")
    print(f"配置: {CONFIG}")
    print("调用: python scripts/train_centralized_prm.py --config " + CONFIG)
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # 优先使用项目本地的 .venv，保证和 M2 训练完全相同的 Python 环境
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        # 回退到 Linux/macOS 虚拟环境路径
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable

    print(f"[mini] 使用 Python: {python_exe}")

    result = subprocess.run(
        [python_exe, "scripts/train_centralized_prm.py", "--config", CONFIG],
        cwd=project_root,
    )

    if result.returncode != 0:
        print("[mini] 训练脚本返回非零退出码，验证失败")
        return 1

    # 检查 checkpoint 是否生成
    ckpt_files = sorted(CKPT_DIR.glob("*.pt"))
    if not ckpt_files:
        print("[mini] 未找到 checkpoint 文件")
        return 1

    print(f"[mini] 发现 checkpoint: {ckpt_files[-1].name}")
    print("=" * 50)
    print("M2 mini 验证通过")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
