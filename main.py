# Code inspired by https://github.com/tamarott/SinGAN
import os
import subprocess
import re
from datetime import datetime
from pathlib import Path
from generate_samples import generate_samples
from train import train
from config import Config
from loguru import logger
import wandb
import sys
import torch

from tensor_level_utils import read_level_from_tensor


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    if opt.use_multiple_inputs:
        return [name.split(".")[0] for name in opt.input_names]
    else:
        return [opt.input_name.split(".")[0], str(opt.scales), str(opt.repr_type), opt.input_area_name]


def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """

    # torch.autograd.set_detect_anomaly(True)

    # Logger init
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

    # Parse arguments
    opt = Config().parse_args()

    def _safe_name(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", "_", s)
        # Keep filesystem-friendly characters
        s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
        return s.strip("._-") or "run"

    # Name runs like: <sample_name>_<YYYYMMDD_HHMMSS>
    # For tensor inputs, sample_name comes from the tensor filename before ".pt".
    if opt.input_type == "tensor" and getattr(opt, "tensor_path", None):
        sample_name = Path(opt.tensor_path).stem
    else:
        sample_name = Path(opt.input_name).stem

    # Allow external launchers to force a specific run name (useful for pipelines).
    forced = os.environ.get("WORLDGAN_RUN_NAME")
    forced_run_name = bool(forced)
    if forced_run_name:
        run_name = _safe_name(forced)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = _safe_name(f"{sample_name}_{ts}")

    # Init wandb
    run = wandb.init(project="world-gan", tags=get_tags(opt), name=run_name,
                     config=opt, dir=opt.out, mode=os.environ.get("WANDB_MODE", "offline"))

    # IMPORTANT:
    # Don't write training artifacts directly into wandb's internal run dir (especially on Windows).
    # Keep outputs in a normal folder, and let wandb.save() copy artifacts into the run.
    opt.wandb_run_dir = run.dir
    runs_root = os.path.join(opt.out, "runs")
    os.makedirs(runs_root, exist_ok=True)

    out_dir = os.path.join(runs_root, run_name)
    if forced_run_name:
        # If an external pipeline chose the name, don't auto-suffix; it is responsible for uniqueness.
        opt.out_ = out_dir
        os.makedirs(opt.out_, exist_ok=True)
    else:
        # Avoid collisions if you launch multiple runs in the same second.
        if os.path.exists(out_dir):
            i = 2
            while os.path.exists(f"{out_dir}_{i}"):
                i += 1
            out_dir = f"{out_dir}_{i}"
        opt.out_ = out_dir
        os.makedirs(opt.out_, exist_ok=True)

    # Relic from old code, where the results where rendered with a generator
    opt.ImgGen = None

    # Minecraft-only: check if wine is available (Linux) and clear the MC world examples will be saved to
    if opt.enable_minecraft_io and opt.input_type == "minecraft":
        try:
            from minecraft.level_utils import clear_empty_world
            subprocess.call(["wine", "--version"])
            clear_empty_world(opt.output_dir, opt.output_name)
        except OSError:
            pass

    # Read level according to input backend
    if opt.input_type == "minecraft":
        from minecraft.level_utils import read_level as mc_read_level
        real = mc_read_level(opt)
    elif opt.input_type == "tensor":
        real = read_level_from_tensor(opt)
    else:
        raise ValueError(f"Unknown input_type: {opt.input_type}")

    # Multi-Input: still not implemented for Minecraft worlds, but supported for tensor inputs externally.
    if opt.use_multiple_inputs and opt.input_type == "minecraft":
        logger.info("Multiple inputs are not implemented yet for Minecraft.")
        raise NotImplementedError

    real = real.to(opt.device)
    opt.level_shape = real.shape[2:]

    # Train!
    generators, noise_maps, reals, noise_amplitudes = train(real, opt)

    # Generate Samples of same size as level
    logger.info("Finished training! Generating random samples...")
    in_s = None
    if opt.use_multiple_inputs:
        use_reals = reals[0]
        use_maps = noise_maps[0]
    else:
        use_reals = reals
        use_maps = noise_maps

    # For tensor-only workflows, keep everything in torch land (no Minecraft world writing/rendering).
    generate_samples(
        generators,
        use_maps,
        use_reals,
        noise_amplitudes,
        opt,
        render_images=False,
        num_samples=100,
        in_s=in_s,
        save_tensors=True,
    )


if __name__ == "__main__":
    main()
