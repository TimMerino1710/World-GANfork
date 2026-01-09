from __future__ import annotations

import sys
import os
import pickle
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from tap import Tap
from torch.utils.data import DataLoader

# Ensure repo root is on PYTHONPATH so this script can be run from any working directory.
# (Matches the pattern used in minecraft/block2vec/train.py)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", ".."))

from skip_gram_model import SkipGramModel
from tensor_block2vec_dataset import TensorBlock2VecDataset
from tensor_level_utils import _extract_tensor_like, _format_no_tensor_error, _load_token_list


class TrainTensorBlock2VecArgs(Tap):
    # Input tensor (your preprocessed volume)
    tensor_path: str
    tensor_key: Optional[str] = None
    tensor_index: Optional[int] = None

    # Specify how to interpret the tensor you saved
    tensor_layout: str = "1CXYZ"  # supports same as tensor loader: 1CXYZ/CXYZ/1XYZC/XYZC/XYZ
    num_channels: Optional[int] = None  # required if tensor_layout == XYZ (integer id volume)
    token_list_path: Optional[str] = None  # list[str] mapping channel->token (optional)

    # If you don't have a ready-made token_list (length == vocab size), you can provide a mappings file
    # that maps channel index -> original block ID (what your visualizer uses).
    mappings_path: Optional[str] = None  # torch file containing block_mappings.index_to_block
    block_types_path: Optional[str] = "assets/block_types.json"  # json mapping block_id->name (optional)

    # Word2Vec/skip-gram settings
    emb_dimension: int = 32
    neighbor_radius: int = 1
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3

    # Output
    output_dir: str = "output/block2vec_tensor"
    output_name: Optional[str] = None  # default: basename of tensor file

    # Runtime
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 0

    def process_args(self) -> None:
        if self.output_name is None:
            self.output_name = Path(self.tensor_path).stem
        self.output_dir = str(Path(self.output_dir) / self.output_name)
        os.makedirs(self.output_dir, exist_ok=True)


def _load_any_tensor(path: str, *, tensor_key: Optional[str], tensor_index: Optional[int]) -> torch.Tensor:
    _, ext = os.path.splitext(path.lower())
    if ext not in {".pt", ".pth", ".npy"}:
        raise ValueError(f"Unsupported tensor_path extension {ext}. Use .pt/.pth/.npy")

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return torch.from_numpy(arr)

    obj = torch.load(path, map_location="cpu")
    t = _extract_tensor_like(obj, source_name=path, tensor_key=tensor_key, tensor_index=tensor_index)
    if t is None:
        raise ValueError(_format_no_tensor_error(obj, path))
    return t


def _to_indices_volume(t: torch.Tensor, *, layout: str, num_channels: Optional[int]) -> torch.Tensor:
    """
    Convert a saved tensor into an integer index volume shaped (X,Y,Z) with values [0..V-1].
    """
    layout = layout.upper()
    if layout == "XYZ":
        if t.ndim != 3:
            raise ValueError(f"Expected XYZ layout with 3D tensor, got shape {tuple(t.shape)}")
        return t.to(torch.long).contiguous()

    # One-hot-ish inputs: coerce to (1,C,X,Y,Z) then argmax over C.
    if layout == "1CXYZ":
        oh = t
    elif layout == "CXYZ":
        oh = t.unsqueeze(0)
    elif layout == "1XYZC":
        oh = t.permute(0, 4, 1, 2, 3)
    elif layout == "XYZC":
        oh = t.permute(3, 0, 1, 2).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor_layout: {layout}")

    if oh.ndim != 5:
        raise ValueError(f"Expected one-hot-like tensor after layout conversion, got shape {tuple(oh.shape)}")

    idx = oh.to(torch.float32).argmax(dim=1).squeeze(0)  # (X,Y,Z)
    return idx.to(torch.long).contiguous()


def save_representations(
    out_dir: str,
    *,
    token_list: List[str],
    embeddings: torch.Tensor,  # (V,D)
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save in the same format World-GAN expects: dict[token]->torch.Tensor(dim,)
    rep: Dict[str, torch.Tensor] = {tok: embeddings[i].detach().cpu() for i, tok in enumerate(token_list)}
    rep_path = out / "representations.pkl"
    with open(rep_path, "wb") as f:
        pickle.dump(rep, f)

    np.save(out / "embeddings.npy", embeddings.detach().cpu().numpy())
    torch.save(token_list, out / "token_list.pt")
    logger.info("Saved representations to {}", rep_path)
    return str(rep_path)


def main():
    args = TrainTensorBlock2VecArgs().parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        logger.warning("CUDA requested but not available; using CPU")

    t = _load_any_tensor(args.tensor_path, tensor_key=args.tensor_key, tensor_index=args.tensor_index)
    idx_vol = _to_indices_volume(t, layout=args.tensor_layout, num_channels=args.num_channels)

    # Define token strings (used as keys in representations.pkl)
    # For one-hot inputs, token_list must align with channel order (index 0..V-1).
    vocab_size = int(idx_vol.max().item()) + 1
    if args.token_list_path:
        token_list = _load_token_list(args.token_list_path)
    elif args.mappings_path:
        mp = torch.load(args.mappings_path, map_location="cpu", weights_only=False)
        if not isinstance(mp, dict) or "block_mappings" not in mp:
            raise ValueError("mappings_path must be a torch file with key 'block_mappings'")
        index_to_block = mp["block_mappings"].get("index_to_block")
        if not isinstance(index_to_block, dict):
            raise ValueError("mappings_path missing block_mappings.index_to_block dict")

        # Optional: map block_id -> readable name (purely for labeling embedding keys)
        block_id_to_name = None
        if args.block_types_path and os.path.exists(args.block_types_path):
            with open(args.block_types_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # json keys are strings; normalize to int->str
            block_id_to_name = {int(k): str(v) for k, v in raw.items()}

        token_list = []
        for i in range(vocab_size):
            if i not in index_to_block:
                raise ValueError(f"index_to_block missing entry for channel index {i} (vocab_size={vocab_size})")
            bid = int(index_to_block[i])
            if block_id_to_name is not None and bid in block_id_to_name:
                token_list.append(block_id_to_name[bid])
            else:
                token_list.append(str(bid))
    else:
        token_list = [f"tok_{i}" for i in range(vocab_size)]

    dataset = TensorBlock2VecDataset(idx_vol, neighbor_radius=args.neighbor_radius)
    vocab_size = dataset.stats.vocab_size
    if len(token_list) != vocab_size:
        raise ValueError(
            f"token_list length ({len(token_list)}) does not match inferred vocab size ({vocab_size}). "
            f"Fix token_list_path or ensure your tensor uses indices 0..{len(token_list)-1}."
        )

    logger.info("Training tensor-block2vec on volume {} with vocab={} dim={} ctx_radius={}",
                tuple(idx_vol.shape), vocab_size, args.emb_dimension, args.neighbor_radius)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = SkipGramModel(vocab_size, args.emb_dimension).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for target, context in loader:
            target = target.to(device)
            context = context.to(device)
            loss = model(target, context)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        logger.info("epoch {}/{}: loss={:.4f}", epoch + 1, args.epochs, float(np.mean(losses)))

        # Save embeddings each epoch so you can inspect early
        embeddings = model.target_embeddings.weight.detach()
        save_representations(args.output_dir, token_list=token_list, embeddings=embeddings)


if __name__ == "__main__":
    main()


