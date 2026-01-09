from __future__ import annotations

import json
import os
import pickle
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


def _load_token_list(path: str) -> List[str]:
    """Load a token list (channel -> token name) from disk."""
    if not path:
        raise ValueError("token_list_path is empty")
    _, ext = os.path.splitext(path.lower())
    if ext in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
    elif ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif ext in {".json"}:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        raise ValueError(f"Unsupported token_list_path extension: {ext}. Use .pt/.pth/.pkl/.json")

    if isinstance(obj, dict):
        # Allow either {int->str} or {str->int}; normalize to list[str] by sorting by index.
        if all(isinstance(k, int) for k in obj.keys()):
            return [obj[i] for i in sorted(obj.keys())]
        if all(isinstance(v, int) for v in obj.values()):
            inv = {v: k for k, v in obj.items()}
            return [inv[i] for i in sorted(inv.keys())]
        raise ValueError("token_list dict must be {int->str} or {str->int}")

    if isinstance(obj, (list, tuple)):
        if not all(isinstance(x, str) for x in obj):
            raise ValueError("token_list list/tuple must contain only strings")
        return list(obj)

    raise ValueError(f"Unsupported token_list object type: {type(obj)}")


def _load_tensor(path: str) -> torch.Tensor:
    """Load tensor-like data from .pt/.pth/.npy into a torch.Tensor on CPU."""
    if not path:
        raise ValueError("tensor_path is empty")
    _, ext = os.path.splitext(path.lower())
    if ext in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        t = _extract_tensor_like(obj, source_name=path)
        if t is None:
            raise ValueError(_format_no_tensor_error(obj, path))
        return t
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return torch.from_numpy(arr)
    raise ValueError(f"Unsupported tensor file extension: {ext}. Use .pt/.pth/.npy")


def _as_tensor(val: Any) -> Optional[torch.Tensor]:
    """Best-effort conversion of a tensor-like object to torch.Tensor (CPU)."""
    if isinstance(val, torch.Tensor):
        return val
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val)
    return None


def _summarize_value(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        return f"torch.Tensor shape={tuple(v.shape)} dtype={v.dtype}"
    if isinstance(v, np.ndarray):
        return f"np.ndarray shape={v.shape} dtype={v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)}"
    if isinstance(v, dict):
        return f"dict keys={list(v.keys())[:20]}"
    return type(v).__name__


def _format_no_tensor_error(obj: Any, path: str) -> str:
    msg = [f"{path} did not contain a recognizable tensor."]
    msg.append(f"Loaded object type: {type(obj)}")
    if isinstance(obj, dict):
        msg.append(f"Dict keys: {list(obj.keys())}")
        msg.append("Hint: pass --tensor_key <key> to select the tensor inside this dict.")
        for k, v in list(obj.items())[:50]:
            msg.append(f"  - {k}: {_summarize_value(v)}")
    elif isinstance(obj, (list, tuple)):
        msg.append(f"Sequence length: {len(obj)}")
        msg.append("Hint: pass --tensor_index <i> to select the tensor inside this list/tuple.")
        for i, v in enumerate(list(obj)[:50]):
            msg.append(f"  - [{i}]: {_summarize_value(v)}")
    else:
        msg.append(f"Summary: {_summarize_value(obj)}")
    return "\n".join(msg)


def _extract_tensor_like(obj: Any, *, source_name: str = "<object>", tensor_key: Optional[str] = None, tensor_index: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Extract a tensor-like value from a torch.load()-ed object.

    Supports:
    - torch.Tensor / np.ndarray directly
    - dict: pick tensor_key, or common keys, or the only tensor-like value
    - list/tuple: pick tensor_index, or the only tensor-like element
    """
    direct = _as_tensor(obj)
    if direct is not None:
        return direct

    if isinstance(obj, dict):
        if tensor_key is not None:
            if tensor_key not in obj:
                raise KeyError(f"{source_name}: tensor_key '{tensor_key}' not found. Available keys: {list(obj.keys())}")
            return _as_tensor(obj[tensor_key])

        # Common patterns
        for k in ("tensor", "data", "level", "x", "sample", "onehot", "one_hot", "voxels", "volume"):
            if k in obj:
                cand = _as_tensor(obj[k])
                if cand is not None:
                    return cand

        # If there's exactly one tensor-like value in the dict, use it.
        tensor_like_keys = [k for k, v in obj.items() if _as_tensor(v) is not None]
        if len(tensor_like_keys) == 1:
            return _as_tensor(obj[tensor_like_keys[0]])
        return None

    if isinstance(obj, (list, tuple)):
        if tensor_index is not None:
            if tensor_index < 0 or tensor_index >= len(obj):
                raise IndexError(f"{source_name}: tensor_index {tensor_index} out of range for length {len(obj)}")
            return _as_tensor(obj[tensor_index])

        tensor_like_idxs = [i for i, v in enumerate(obj) if _as_tensor(v) is not None]
        if len(tensor_like_idxs) == 1:
            return _as_tensor(obj[tensor_like_idxs[0]])
        return None


def _to_1cxyz_onehot(
    t: torch.Tensor,
    *,
    layout: str,
    num_channels: Optional[int],
) -> torch.Tensor:
    """
    Convert an input tensor to a (1, C, X, Y, Z) float tensor.

    Supported layouts:
    - "1CXYZ": (1, C, X, Y, Z)
    - "CXYZ":  (C, X, Y, Z)
    - "1XYZC": (1, X, Y, Z, C)
    - "XYZC":  (X, Y, Z, C)
    - "XYZ":   (X, Y, Z) integer ids -> one-hot (requires num_channels)
    """
    layout = layout.upper()
    if layout == "1CXYZ":
        if t.ndim != 5:
            raise ValueError(f"Expected 5D tensor for layout 1CXYZ, got shape {tuple(t.shape)}")
        out = t
    elif layout == "CXYZ":
        if t.ndim != 4:
            raise ValueError(f"Expected 4D tensor for layout CXYZ, got shape {tuple(t.shape)}")
        out = t.unsqueeze(0)
    elif layout == "1XYZC":
        if t.ndim != 5:
            raise ValueError(f"Expected 5D tensor for layout 1XYZC, got shape {tuple(t.shape)}")
        out = t.permute(0, 4, 1, 2, 3)
    elif layout == "XYZC":
        if t.ndim != 4:
            raise ValueError(f"Expected 4D tensor for layout XYZC, got shape {tuple(t.shape)}")
        out = t.permute(3, 0, 1, 2).unsqueeze(0)
    elif layout == "XYZ":
        if t.ndim != 3:
            raise ValueError(f"Expected 3D tensor for layout XYZ, got shape {tuple(t.shape)}")
        if num_channels is None:
            raise ValueError("num_channels is required when tensor_layout == 'XYZ'")
        ids = t.to(torch.long)
        # (X,Y,Z) -> (X,Y,Z,C) -> (1,C,X,Y,Z)
        oh = F.one_hot(ids, num_classes=int(num_channels)).to(torch.float32)
        out = oh.permute(3, 0, 1, 2).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor_layout: {layout}")

    # Normalize dtype to float32 (trainer expects float tensors).
    if out.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        out = out.to(torch.float32)
    else:
        out = out.to(torch.float32)

    return out


def read_level_from_tensor(opt) -> torch.Tensor:
    """
    Read a training level tensor according to Config options.

    Returns:
      real: torch.Tensor shaped (1, C, X, Y, Z) on CPU (caller moves to device).

    Side effects (mirrors minecraft loader expectations):
      - opt.token_list set to list[str] (length C)
      - opt.props set to list[dict] (length C; empty dicts)
      - opt.nc_current set to C
    """
    if not opt.tensor_path:
        raise ValueError("For --input_type tensor, you must pass --tensor_path")

    # Load and extract tensor-like value (supports dict/tuple files).
    if not opt.tensor_path:
        raise ValueError("For --input_type tensor, you must pass --tensor_path")
    _, ext = os.path.splitext(opt.tensor_path.lower())
    if ext in {".pt", ".pth"}:
        obj = torch.load(opt.tensor_path, map_location="cpu")
        t = _extract_tensor_like(obj, source_name=opt.tensor_path, tensor_key=getattr(opt, "tensor_key", None), tensor_index=getattr(opt, "tensor_index", None))
        if t is None:
            raise ValueError(_format_no_tensor_error(obj, opt.tensor_path))
    else:
        t = _load_tensor(opt.tensor_path)
    onehot = _to_1cxyz_onehot(t, layout=opt.tensor_layout, num_channels=opt.num_channels)

    c = int(onehot.shape[1])
    token_list: List[str]
    if opt.token_list_path:
        token_list = _load_token_list(opt.token_list_path)
        # For one-hot workflows, token_list_path is expected to describe channel order (len == C).
        # For block2vec workflows, token_list_path typically describes the embedding vocabulary order
        # (which may be smaller than C if your exemplar doesn't use all global channels).
        if getattr(opt, "repr_type", None) != "block2vec" and len(token_list) != c:
            raise ValueError(
                f"token_list length ({len(token_list)}) does not match tensor channels ({c})"
            )
    else:
        token_list = [f"tok_{i}" for i in range(c)]

    # Light sanity check: one-hotness (cheap sampling, not full scan).
    with torch.no_grad():
        x, y, z = onehot.shape[2], onehot.shape[3], onehot.shape[4]
        n = min(2048, x * y * z)
        if n > 0:
            idx = torch.randint(0, x * y * z, (n,))
            vals = onehot.view(1, c, -1)[0, :, idx]  # (C, n)
            sums = vals.sum(dim=0)
            frac_ok = (sums - 1.0).abs().lt(1e-3).float().mean().item()
            logger.info("Tensor input quick-check: {:.1f}% sampled voxels sum to ~1 across channels", frac_ok * 100)

    # If using block2vec representations, convert one-hot -> embedding channels (1, D, X, Y, Z)
    if getattr(opt, "repr_type", None) == "block2vec":
        block2repr = getattr(opt, "block2repr", None)
        if not isinstance(block2repr, dict) or len(block2repr) == 0:
            raise ValueError(
                "repr_type=block2vec requires opt.block2repr to be a non-empty dict. "
                "Provide --block2repr_path in config for tensor inputs."
            )

        # IMPORTANT: token_list for block2vec is the embedding vocabulary (dict keys), not the embedding dimension.
        vocab = list(block2repr.keys())
        if opt.token_list_path:
            # For block2vec, token_list_path should describe the same vocabulary as representations.pkl.
            # It does NOT need to match the original one-hot channel count.
            if len(token_list) != len(vocab) or set(token_list) != set(vocab):
                raise ValueError(
                    "token_list_path does not match block2repr keys. "
                    "For block2vec, token_list_path should describe the same tokens as representations.pkl."
                )
            vocab = token_list  # use explicit ordering

        # Build embedding matrix in vocab order: (V, D)
        emb_mat = torch.stack([block2repr[t].detach().to(torch.float32).cpu() for t in vocab], dim=0)
        d = int(emb_mat.shape[1])

        # Convert one-hot -> indices -> embeddings
        idx = onehot.argmax(dim=1).squeeze(0).to(torch.long)  # (X,Y,Z) indices into original channels
        max_idx = int(idx.max().item()) if idx.numel() else 0
        if max_idx >= len(vocab):
            raise ValueError(
                f"Your one-hot tensor uses channel indices up to {max_idx}, but your block2vec vocabulary has "
                f"only {len(vocab)} entries. This usually means you trained representations on a smaller vocab.\n"
                f"Fix options:\n"
                f"- Train block2vec with a vocabulary size that matches your channel space (C={c}), or\n"
                f"- Ensure the exemplar only uses indices < {len(vocab)} (compress/reindex channels for this exemplar)."
            )
        emb = emb_mat[idx]  # (X,Y,Z,D)
        emb = emb.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,D,X,Y,Z)

        opt.token_list = vocab
        opt.props = [{} for _ in range(len(vocab))]
        opt.nc_current = d
        return emb

    # Default: keep one-hot channels
    opt.token_list = token_list
    opt.props = [{} for _ in range(c)]
    opt.nc_current = c
    return onehot


