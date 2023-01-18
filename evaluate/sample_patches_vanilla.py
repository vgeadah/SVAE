#!/usr/bin/env python3
"""Samples patches from the BSDS300 dataset.

Writes train, test, and validation datasets to the "bsds300_patches" directory in
scratch.

"""
import logging
import pathlib

import hydra
import torch
from scipy import io

from svae import data

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def run(cfg, filename='patches.pt') -> None:
    """Write randomly-sampled train, test, and validation patch datasets.

    They are written to a dict with "train", "test", and "val" keys. Each references
    a tensor in NHW format.

    """
    path = pathlib.Path(cfg.paths.bsds300_prewhitened) / "IMAGES.mat"
    images = (
        torch.tensor(io.loadmat(path)["IMAGES"], dtype=torch.float32)
        .permute(2, 0, 1)
        .contiguous()
    )
    out = {
        "train": data.sample_patches(images, n_patches=65536, seed=42),
        "test": data.sample_patches(images, n_patches=16384, seed=43),
        "val": data.sample_patches(images, n_patches=16384, seed=44),
    }

    if cfg.bin.sample_patches_vanilla.destination == "cwd":
        base_path = pathlib.Path(".")
    else:
        base_path = pathlib.Path(cfg.paths.scratch)
    torch.save(out, base_path / filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
