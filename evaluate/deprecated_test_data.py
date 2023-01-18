"""Tests data.py."""
import pathlib
import random

import numpy as np
import pytest
from scipy import fftpack

from svae import data


@pytest.fixture(scope="module")
def patches():
    return pathlib.Path("./tests/fixtures/patches")


@pytest.fixture(scope="module")
def img_folder():
    return pathlib.Path("./tests/fixtures/img_folder")


@pytest.fixture(scope="module")
def train_dataset(patches):
    return data.AutoencoderImageFolder(patches / "train")


def test_normalize_normalizes_to_plus_minus_one():
    x = 100 * np.random.randn(100)
    x_normalized = data.normalize(x)
    assert np.min(x_normalized) == -1
    assert np.max(x_normalized) == 1


def test_radial_spatial_freqs_is_circularly_symmetric():
    R = fftpack.fftshift(data.radial_spatial_freqs(7, 9))
    center = (3, 4)
    assert R[center] == 0
    assert (np.flip(R, 0) == R).all()
    assert (np.flip(R, 1) == R).all()
    assert (R >= 0).all()
    assert R.shape == (7, 9)


def test_you_can_load_a_folder_of_jpg_images(img_folder):
    imgs = data.load_img_folder(img_folder)
    assert len(imgs) == 2


def test_you_can_randomly_sample_from_an_iterable():
    random.seed(42)
    x = (0, 1, 2)
    y = np.array(list(data.shuffled_sample(x, n_samples=6)))
    for i in x:
        assert np.sum(y == i) == 2
    assert not np.all(y == np.array([0, 1, 2, 0, 1, 2]))


def test_you_can_convert_paths_to_image_patches(img_folder):
    imgs = data.load_img_folder(img_folder)
    patches = data.to_patches(imgs)
    for i in patches:
        assert i.size == (12, 12)


def test_you_can_make_an_image_dataset(train_dataset):
    assert len(train_dataset) == 2


def test_you_can_convert_an_autoencoder_image_folder_to_a_tensor(train_dataset):
    assert train_dataset.to_tensor().size() == (2, 12, 12)


def test_you_can_get_a_train_and_test_dataloader(patches):
    train, test, val = data.get_dataloaders(patches, batch_size=1)
    batch = next(iter(train))
    assert batch.shape == (1, 12, 12)
