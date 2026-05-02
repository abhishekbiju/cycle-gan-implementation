"""
Compute FID(real_B, G_ab(real_A)) for a saved checkpoint and two image folders.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from discriminator import Discriminator
from fid import compute_fid_ab_translation, load_inception_for_fid
from generator import Generator
from utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="FID for CycleGAN A→B translation")
    p.add_argument("--train_a", type=str, required=True)
    p.add_argument("--train_b", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tf = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = ImageDataset(args.train_a, args.train_b, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    G_ab = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    G_ba = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    D_a = Discriminator(in_channels=3).to(device)
    D_b = Discriminator(in_channels=3).to(device)
    load_checkpoint(args.checkpoint, G_ab, G_ba, D_a, D_b, device=str(device))

    net = load_inception_for_fid(device)
    fid = compute_fid_ab_translation(
        G_ab, loader, device, max_samples=args.max_samples, net=net
    )
    print(f"FID (real_B vs G_ab(A)): {fid:.4f}")


if __name__ == "__main__":
    main()
