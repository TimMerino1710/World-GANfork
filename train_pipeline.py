from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("._-") or "run"


def _reserve_run_dir(runs_root: Path, base_name: str) -> Tuple[str, Path]:
    """Reserve a unique run dir name under runs_root, adding _2/_3 suffixes if needed."""
    runs_root.mkdir(parents=True, exist_ok=True)
    name = _safe_name(base_name)
    out_dir = runs_root / name
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return name, out_dir
    i = 2
    while (runs_root / f"{name}_{i}").exists():
        i += 1
    name2 = f"{name}_{i}"
    out_dir2 = runs_root / name2
    out_dir2.mkdir(parents=True, exist_ok=True)
    return name2, out_dir2


def _cli_from_kv(d: Dict[str, Any]) -> List[str]:
    """
    Convert a dict to CLI args for Tap/argparse scripts.

    - bool True  -> ["--key"]
    - bool False/None -> []
    - list/tuple -> ["--key", "a", "b", ...]
    - other -> ["--key", str(val)]
    """
    args: List[str] = []
    for k, v in d.items():
        if v is None:
            continue
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)
            continue
        if isinstance(v, (list, tuple)):
            args.append(key)
            args.extend([str(x) for x in v])
            continue
        args.extend([key, str(v)])
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description="Train block2vec (word2vec) on a tensor sample, then train World-GAN using those representations.")
    ap.add_argument("--config", required=True, help="Path to YAML config file.")
    ap.add_argument("--sample", required=True, help="Path to the input sample tensor (.pt/.pth/.npy). Sample name is filename stem.")
    ap.add_argument("--gpu", default=None, help="Optional GPU index to use (sets CUDA_VISIBLE_DEVICES). Overrides config.runtime.gpu.")
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML mapping/dict.")

    sample_cfg = cfg.get("sample", {}) or {}
    block2vec = cfg.get("block2vec", {}) or {}
    gan = cfg.get("gan", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    if not all(isinstance(x, dict) for x in (sample_cfg, block2vec, gan, runtime)):
        raise ValueError("sample/block2vec/gan/runtime must be dicts.")

    tensor_path = Path(args.sample)
    tensor_layout = sample_cfg.get("tensor_layout", "1CXYZ")
    tensor_key = sample_cfg.get("tensor_key")
    tensor_index = sample_cfg.get("tensor_index")
    num_channels = sample_cfg.get("num_channels")

    out_root = Path(runtime.get("out_root", "output"))
    runs_root = out_root / "runs"

    sample_name = tensor_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = runtime.get("run_name") or f"{sample_name}_{ts}"
    run_name, run_dir = _reserve_run_dir(runs_root, base_run_name)

    # Paths for block2vec artifacts inside the run dir
    b2v_out_dir = run_dir / "block2vec"
    rep_path = b2v_out_dir / "representations.pkl"
    token_list_path = b2v_out_dir / "token_list.pt"

    # Environment
    env = os.environ.copy()
    wandb_mode = runtime.get("wandb_mode")
    if wandb_mode:
        env["WANDB_MODE"] = str(wandb_mode)

    gpu = args.gpu if args.gpu is not None else runtime.get("gpu")
    if gpu is not None and str(gpu).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Force main.py to use our run dir name (so both steps land under the same folder).
    env["WORLDGAN_RUN_NAME"] = run_name

    print(f"Run dir: {run_dir}")
    print(f"  block2vec -> {rep_path}")
    print(f"  token_list -> {token_list_path}")
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"WANDB_MODE={env.get('WANDB_MODE', '<unset>')}")

    # ----------------------------
    # 1) Train block2vec on tensor
    # ----------------------------
    b2v_args: Dict[str, Any] = {
        "tensor_path": str(tensor_path),
        "tensor_layout": tensor_layout,
        "tensor_key": tensor_key,
        "tensor_index": tensor_index,
        "num_channels": num_channels,
        # token_list_path here is for labeling vocab; if you already have one-hot channel labels, you can pass it.
        "token_list_path": sample_cfg.get("token_list_path"),
        "mappings_path": sample_cfg.get("mappings_path"),
        "block_types_path": sample_cfg.get("block_types_path", "assets/block_types.json"),
        # Write into <run_dir>/block2vec/
        "output_dir": str(run_dir),
        "output_name": "block2vec",
    }
    # Merge user-provided block2vec hyperparams, but don't allow overriding pipeline-controlled paths.
    b2v_user = {k: v for k, v in block2vec.items() if k not in {"tensor_path", "tensor_layout", "tensor_key", "tensor_index", "num_channels", "output_dir", "output_name"}}
    b2v_args.update(b2v_user)

    cmd_b2v = [sys.executable, str(Path("minecraft") / "block2vec" / "train_tensor.py")] + _cli_from_kv(b2v_args)

    # ----------------------------
    # 2) Train GAN using block2vec
    # ----------------------------
    gan_args: Dict[str, Any] = {
        "out": str(out_root),
        "input_type": "tensor",
        "tensor_path": str(tensor_path),
        "tensor_layout": tensor_layout,
        "tensor_key": tensor_key,
        "tensor_index": tensor_index,
        "num_channels": num_channels,
        # block2vec integration
        "repr_type": "block2vec",
        "block2repr_path": str(rep_path),
        # For block2vec, the ordering must match representations.pkl; use the token list produced in step 1.
        "token_list_path": str(token_list_path),
    }
    # Merge GAN hyperparams from config (must match fields in config.py), but don't allow overriding
    # pipeline-controlled input/representation wiring.
    protected = {
        "out", "input_type", "tensor_path", "tensor_layout", "tensor_key", "tensor_index", "num_channels",
        "repr_type", "block2repr_path", "token_list_path",
    }
    gan_user = {k: v for k, v in gan.items() if k not in protected}
    gan_args.update(gan_user)

    cmd_gan = [sys.executable, str(Path("main.py"))] + _cli_from_kv(gan_args)

    print("\n### Step 1: block2vec")
    print(" ".join(cmd_b2v))
    print("\n### Step 2: world-gan")
    print(" ".join(cmd_gan))

    if args.dry_run:
        return 0

    # Run step 1
    r1 = subprocess.run(cmd_b2v, env=env)
    if r1.returncode != 0:
        return r1.returncode
    if not rep_path.exists():
        raise RuntimeError(f"block2vec finished but did not produce {rep_path}")
    if not token_list_path.exists():
        raise RuntimeError(f"block2vec finished but did not produce {token_list_path}")

    # Run step 2
    r2 = subprocess.run(cmd_gan, env=env)
    return r2.returncode


if __name__ == "__main__":
    raise SystemExit(main())


