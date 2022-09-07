"""Tools for managing datasets."""
import itertools
import pathlib
import random
from typing import Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np
import torch
from PIL import Image
from scipy import fftpack
from torch.utils import data
from torchvision import transforms


def normalize(x: np.ndarray) -> np.ndarray:
    """Centers an array and normalizes to +/- 1."""
    y = x - np.min(x)
    y = y / np.max(y)
    return 2 * y - 1


def radial_spatial_freqs(m: int, n: int) -> np.ndarray:
    """Compute the radial polar coordinate of an image's spatial frequencies.

    We assume one time step per pixel.

    Args:
        m: The number of rows in the image.
        n: The number of columns in the image.

    Returns:
        A matrix containing the radial polar coordinate of the spatial frequencies
        corresponding to each element in an unshifted 2D fourier transform.

    """
    u = fftpack.fftfreq(m, (1 / m))
    v = fftpack.fftfreq(n, (1 / n))
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def whitening_filter(R: np.ndarray, r_0: int = 200) -> np.ndarray:
    """Compute the whitening filter.

    The details are found in

        B.A. Olshausen, D.J. Field, Vision Res. 37 (1997) 3311–3325.

    Args:
        R: The radial spatial frequency array.
        r_0: The cutoff frequency.

    Returns:
        The unshifted filter.

    """
    return R * np.exp(-((R / r_0) ** 4))


def whiten(img: np.ndarray) -> Image.Image:
    """Whiten and normalize an image."""
    fft = fftpack.fft2(img)
    filt = whitening_filter(radial_spatial_freqs(*fft.shape))
    img_w = np.abs(fftpack.ifft2(filt * fft))
    return normalize(img_w)


def normalized_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a centered -1/+1 normalized numpy array to a grayscale PIL image."""
    img_uint8 = np.uint8(255 * (0.5 * img + 0.5))
    return Image.fromarray(img_uint8)


def load_img_folder(path: pathlib.Path) -> List[pathlib.Path]:
    """Load all JPG or PNG images in a folder (or a mix).

    Args:
        path: The path to the image folder.

    Returns:
        A list of images.

    """
    paths = itertools.chain(path.rglob("*.jpg"), path.rglob("*.png"))
    return [Image.open(p).convert("L") for p in paths]


T = TypeVar("T")


def shuffled_sample(xs: Sequence[T], n_samples: int) -> Iterable[T]:
    """Randomly samples from an iterable without replacement.

    When we run out of elements in the iterable, it is reshuffled.

    Args:
        xs: The iterable from which to sample.
        n_samples: The number of samples to take.

    Returns:
        A generator yielding randomly-sampled values.

    """
    indices: List[int] = []
    for _ in range(n_samples):
        if not indices:
            indices = list(range(len(xs)))
            random.shuffle(indices)
        yield xs[indices.pop()]


def to_patches(images: Iterable[Image.Image]) -> Iterable[Image.Image]:
    """Transform image paths into randomly-cropped PIL image patches.

    This depends on PyTorch's random number generation, so be sure to seed it before
    calling for reproducibility.

    Args:
        paths: A sequence of image paths.
        seed: The random seed.

    Returns:
        A generator of 12 x 12 PIL images representing patches randomly sampled from the
        full images.

    """
    transform = transforms.Compose([transforms.RandomCrop(12), transforms.Grayscale()])
    for img in images:
        yield transform(img)


class AutoencoderImageFolder(data.Dataset):
    """An image dataset for autoencoders, which have only features (X), not labels."""

    def __init__(self, path: pathlib.Path) -> None:
        self._data: List[torch.Tensor] = []
        for i in path.rglob("*.png"):
            image = Image.open(i)
            self._data.append(torch.squeeze(transforms.functional.to_tensor(image)))

        if len(self._data) == 0:
            raise UserWarning("No images found in {0}".format(path))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def to_tensor(self) -> torch.Tensor:
        """Return the dataset as a single tensor, stacked along the first dimension."""
        return torch.stack(self._data)


def get_dataloaders(
    patch_dir: pathlib.Path,
    batch_size: int = 128,
    shuffle: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Get train, test, and validation loaders.

    Args:
        patch_dir: The directory containing the sampled patches.
        batch_size: The minibatch size to use.
        shuffle: Whether or not to shuffle the data.
        device: The device on which to place the loaded data. Since our patches
            are small and can fit in GPU memory, this should be a cuda device if we are
            training on the GPU.

    Returns:
        A tuple containing the train, test, and validation dataloaders.

    """
    if device is None:
        device = torch.device("cpu")

    train_dataset = AutoencoderImageFolder(patch_dir / "train").to_tensor().to(device)
    test_dataset = AutoencoderImageFolder(patch_dir / "test").to_tensor().to(device)
    val_dataset = AutoencoderImageFolder(patch_dir / "val").to_tensor().to(device)

    if shuffle:
        base_sampler: Type[data.Sampler] = data.RandomSampler
    else:
        base_sampler = data.SequentialSampler

    train_sampler = data.BatchSampler(
        base_sampler(train_dataset), batch_size, drop_last=False  # type: ignore
    )
    test_sampler = data.BatchSampler(  # type: ignore
        base_sampler(test_dataset), batch_size, drop_last=False  # type: ignore
    )
    val_sampler = data.BatchSampler(  # type: ignore
        base_sampler(val_dataset), batch_size, drop_last=False  # type: ignore
    )

    # Note that we're using a *batch* sampler as the sampler. This is a trick to
    # make sure that we skip collation and use fancy indexing instead, a major speedup
    train_loader = data.DataLoader(  # type: ignore
        train_dataset, sampler=train_sampler, collate_fn=lambda x: x[0]
    )
    test_loader = data.DataLoader(  # type: ignore
        test_dataset, sampler=test_sampler, collate_fn=lambda x: x[0]
    )
    val_loader = data.DataLoader(  # type: ignore
        val_dataset, sampler=val_sampler, collate_fn=lambda x: x[0]
    )

    return train_loader, test_loader, val_loader
