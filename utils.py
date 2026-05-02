import os
from pathlib import Path

import torch
import torchvision.utils as vutils


def save_samples(path, images, nrow=4):
    """Save a grid tensor in [-1, 1] to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_checkpoint(path, epoch, G_xy, G_yx, D_x, D_y, opt_G, opt_D):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_xy": G_xy.state_dict(),
            "G_yx": G_yx.state_dict(),
            "D_x": D_x.state_dict(),
            "D_y": D_y.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        },
        path,
    )


def load_checkpoint(path, G_xy, G_yx, D_x, D_y, opt_G=None, opt_D=None, device="cpu"):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    G_xy.load_state_dict(ckpt["G_xy"])
    G_yx.load_state_dict(ckpt["G_yx"])
    D_x.load_state_dict(ckpt["D_x"])
    D_y.load_state_dict(ckpt["D_y"])
    if opt_G is not None and "opt_G" in ckpt:
        opt_G.load_state_dict(ckpt["opt_G"])
    if opt_D is not None and "opt_D" in ckpt:
        opt_D.load_state_dict(ckpt["opt_D"])
    return ckpt.get("epoch", 0)
