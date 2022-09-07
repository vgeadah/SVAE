#!/usr/bin/env python3
"""Samples patches from the BSDS300 dataset.

Writes train, test, and validation datasets to the "bsds300_patches" directory in
scratch.

"""
import logging
import pathlib
import random
from typing import Iterable, List, Sequence

import hydra
import torch
import tqdm
from PIL import Image

from svae import data

logger = logging.getLogger(__name__)


def whiten_pils(images: Iterable[Image.Image]) -> List[Image.Image]:
    """Whitens an iterable of PIL images."""
    logger.info("Whitening images...")
    out = []
    for img in tqdm.tqdm(images):
        out.append(data.normalized_to_pil(data.whiten(data.normalize(img))))
    return out


def write_dataset(
    out_dir: pathlib.Path, whitened: Sequence[Image.Image], n_patches: int, seed: int
) -> None:
    """Write a randomly-sampled patch dataset to scratch.

    Args:
        suffix: The folder inside the output directory in which to store the images.
        whitened: An iterable of the whitened images from which to sample.
        n_patches: The number of patches to sample.
        seed: The random seed.

    """
    random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Writing patches to %s", out_dir)

    shuffled: Iterable[Image.Image] = data.shuffled_sample(whitened, n_patches)
    for i, patch in tqdm.tqdm(enumerate(data.to_patches(shuffled)), total=n_patches):
        patch.save(out_dir / "{0}.png".format(i))


@hydra.main("../conf/config.yaml")
def run(cfg) -> None:
    """Write randomly-sampled train, test, and validation patch datasets.

    They are placed in the "bsds300_patches" directory in scratch.

    """
    whitened = whiten_pils(data.load_img_folder(pathlib.Path(cfg.paths.bsds300)))
    write_dataset(
        pathlib.Path("./train"),
        whitened,
        n_patches=cfg.bin.sample_patches.n_test_patches,
        seed=cfg.bin.sample_patches.seed,
    )
    write_dataset(
        pathlib.Path("./test"),
        whitened,
        n_patches=cfg.bin.sample_patches.n_test_patches,
        seed=cfg.bin.sample_patches.seed + 1,
    )
    write_dataset(
        pathlib.Path("./val"),
        whitened,
        n_patches=cfg.bin.sample_patches.n_test_patches,
        seed=cfg.bin.sample_patches.seed + 2,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
