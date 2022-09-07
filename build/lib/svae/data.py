"""Tools for managing datasets."""
import pathlib
import random
from typing import Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

import torch
from torch.utils import data

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


def sample_patch(img: torch.Tensor, patch_size: int = 12) -> torch.Tensor:
    """Sample a random patch from a 2D torch Tensor.

    Uses the Pytorch RNG.

    Args:
        img: The tensor from which to sample the patch.
        patch_size: The patch size.

    """
    h, w = img.size()
    max_h, max_w = h - patch_size, w - patch_size
    i: int = torch.randint(max_h, (1,)).item()  # type: ignore
    j: int = torch.randint(max_w, (1,)).item()  # type: ignore
    return img[i : i + patch_size, j : j + patch_size]


def sample_patches(
    images: torch.Tensor, n_patches: int = 65536, seed: int = 42
) -> torch.Tensor:
    """Sample patches from a tensor of images.

    Args:
        images: The tensor of images, of shape `(N, H, W)`.
        n_patches: The number of patches to sample.
        seed: The seed of the RNG.

    Returns:
        A tensor of patches, of shape `(n_patches, 12, 12)`.

    """
    torch.random.manual_seed(seed)
    patches = tuple(
        sample_patch(i)  # type: ignore
        for i in shuffled_sample(images, n_samples=n_patches)  # type: ignore
    )
    return torch.stack(patches)

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

    patches = torch.load(patch_dir)

    train_dataset = patches["train"].to(device)
    val_dataset = patches["val"].to(device)
    test_dataset = patches["test"].to(device)

    if shuffle:
        base_sampler: Type[data.Sampler] = data.RandomSampler
    else:
        base_sampler = data.SequentialSampler

    train_sampler = data.BatchSampler(
        base_sampler(train_dataset), batch_size, drop_last=False  # type: ignore
    )
    val_sampler = data.BatchSampler(  # type: ignore
        base_sampler(val_dataset), batch_size, drop_last=False  # type: ignore
    )
    test_sampler = data.BatchSampler(  # type: ignore
        base_sampler(test_dataset), batch_size, drop_last=False  # type: ignore
    )

    # Note that we're using a *batch* sampler as the sampler. This is a trick to
    # make sure that we skip collation and use fancy indexing instead, a major speedup
    train_loader = data.DataLoader(  # type: ignore
        train_dataset, sampler=train_sampler, collate_fn=lambda x: x[0]  # type: ignore
    )
    val_loader = data.DataLoader(  # type: ignore
        val_dataset, sampler=val_sampler, collate_fn=lambda x: x[0]  # type: ignore
    )
    test_loader = data.DataLoader(  # type: ignore
        test_dataset, sampler=test_sampler, collate_fn=lambda x: x[0]  # type: ignore
    )

    return train_loader, val_loader, test_loader
