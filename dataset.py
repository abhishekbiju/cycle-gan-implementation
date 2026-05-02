"""
Unpaired image datasets for CycleGAN: independent draws from domain A and domain B.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Loads all images from ``root_a`` and ``root_b``. Each __getitem__ returns one random
    image from A and one random image from B (no correspondence between them).
    """

    def __init__(self, root_a, root_b, transform=None):
        self.root_a = Path(root_a)
        self.root_b = Path(root_b)
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths_a = sorted(
            p for p in self.root_a.iterdir() if p.suffix.lower() in exts and p.is_file()
        )
        self.paths_b = sorted(
            p for p in self.root_b.iterdir() if p.suffix.lower() in exts and p.is_file()
        )
        if len(self.paths_a) == 0 or len(self.paths_b) == 0:
            raise ValueError(
                f"Need non-empty image folders. Got {len(self.paths_a)} images in A "
                f"and {len(self.paths_b)} in B ({root_a}, {root_b})."
            )

        self.len_a = len(self.paths_a)
        self.len_b = len(self.paths_b)
        self.length = max(self.len_a, self.len_b)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path_a = self.paths_a[index % self.len_a]
        path_b = self.paths_b[index % self.len_b]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b
