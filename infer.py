"""
Translate all images in a folder with a trained G : domain A → domain B checkpoint.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Run CycleGAN generator A→B on a folder of images")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    G_ab = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    G_ba = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    D_a = Discriminator(in_channels=3).to(device)
    D_b = Discriminator(in_channels=3).to(device)
    load_checkpoint(args.checkpoint, G_ab, G_ba, D_a, D_b, device=str(device))
    G_ab.eval()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts and p.is_file())

    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            y = G_ab(x).squeeze(0).cpu().clamp(-1, 1)
            out = transforms.functional.to_pil_image((y + 1) / 2)
            out.save(output_dir / path.name)


if __name__ == "__main__":
    main()
