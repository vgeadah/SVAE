#!/usr/bin/env python3
import glob
import logging
import pathlib

import hydra
import numpy as np
import PIL.Image
import torch
import tqdm
from sklearn import decomposition, feature_extraction, model_selection

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def run(cfg) -> None:
    """Write randomly-sampled train, test, and validation patch datasets.

    They are written to a dict with "train", "test", and "val" keys. Each references
    a tensor in NHW format.

    """
    logging.info("Reading images...")
    image_paths = glob.glob(str(pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.bsds300) / "images/*/*.jpg"))
    images = []
    for path in tqdm.tqdm(image_paths):
        # Read image as black and white
        img = np.array(np.array(PIL.Image.open(path).convert(mode="L")))

        # Normalize image to [-1, 1]
        img = img / 255.0  # Transform to [0, 1]

        # Rotate portrait images to landscape
        if img.shape[0] > img.shape[1]:
            img = img.transpose()

        images.append(img)

    # Check that all dims are the same, this is just a smoke test
    dims = [i.shape for i in images]
    image_dim = dims[0]
    for i in dims:
        assert i == image_dim, "ERROR: Some of the images have different dimensions"

    logging.info("Sampling patches...")
    patch_dim = (12, 12)
    n_pixels = patch_dim[0] * patch_dim[1]
    patches_per_img = int(cfg.bin.sample_patches_custom.n_patches / len(images))
    patches_list = []
    for img in tqdm.tqdm(images):
        patches_list.append(
            feature_extraction.image.extract_patches_2d(
                img,
                patch_dim,
                max_patches=patches_per_img,
                random_state=cfg.bin.sample_patches_custom.seed,
            )
        )
    X_all = np.stack(patches_list).reshape(-1, n_pixels)

    # Split into tain/test/val sets (we have to do this before reducing dim
    # with PCA so that the transform will only be learned on the train set)
    train_frac = cfg.bin.sample_patches_custom.train_frac
    test_frac = cfg.bin.sample_patches_custom.test_frac
    val_frac = 1 - train_frac - test_frac

    X_train, X_test = model_selection.train_test_split(
        X_all, test_size=1 - train_frac, random_state=cfg.bin.sample_patches_custom.seed
    )
    X_val, X_test = model_selection.train_test_split(
        X_test,
        test_size=test_frac / (test_frac + val_frac),
        random_state=cfg.bin.sample_patches_custom.seed,
    )

    logging.info("Lowpass filtering with PCA...")
    n_components = int(cfg.bin.sample_patches_custom.pca_frac * n_pixels)
    pca = decomposition.PCA(n_components, whiten=True)
    pca.fit(X_train)

    def reconstruct(X: np.ndarray) -> np.ndarray:
        return X @ pca.components_ + pca.mean_

    X_train = reconstruct(pca.transform(X_train))
    X_test = reconstruct(pca.transform(X_test))
    X_val = reconstruct(pca.transform(X_val))

    # The lowpass filtering has disrupted the data scaling, so now we ned to rescale
    # everything again
    min_val = np.min(X_train)
    max_val = np.max(X_train)

    def rescale(X: np.ndarray) -> np.ndarray:
        return (X - min_val) / (max_val - min_val)

    X_train = rescale(X_train)
    X_test = rescale(X_test)
    X_val = rescale(X_val)

    logging.info("Saving patches...")
    out = {
        "train": torch.tensor(X_train, dtype=torch.float32),
        "test": torch.tensor(X_test, dtype=torch.float32),
        "val": torch.tensor(X_val, dtype=torch.float32),
    }

    if cfg.bin.sample_patches_custom.destination == "cwd":
        base_path = pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(".")
    else:
        base_path = pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch)
    torch.save(out, base_path / "patches.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
