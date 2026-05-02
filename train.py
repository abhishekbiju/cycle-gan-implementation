"""
Train CycleGAN with least-squares adversarial loss, cycle consistency, and optional identity mapping.
"""

import argparse
import itertools
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from discriminator import Discriminator
from generator import Generator
from fid import compute_fid_ab_translation, load_inception_for_fid
from utils import load_checkpoint, save_checkpoint, save_samples


def learning_rate(epoch_idx: int, epochs: int, base_lr: float, scheduler: str, constant_epochs: int) -> float:
    """CycleGAN-style linear decay after ``constant_epochs`` full-LR epochs (0-based epoch index)."""
    if scheduler == "constant":
        return base_lr
    if constant_epochs >= epochs or epoch_idx < constant_epochs:
        return base_lr
    denom = (epochs - 1) - constant_epochs
    if denom <= 0:
        return 0.0
    return base_lr * max(0.0, float(epochs - 1 - epoch_idx) / float(denom))


def parse_args():
    p = argparse.ArgumentParser(description="Train CycleGAN (unpaired domains)")
    p.add_argument("--train_a", type=str, required=True, help="Folder of domain A training images")
    p.add_argument("--train_b", type=str, required=True, help="Folder of domain B training images")
    p.add_argument("--output_dir", type=str, default="./runs/cyclegan")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument(
        "--lambda_identity",
        type=float,
        default=0.5,
        help="Weight on identity loss relative to cycle (set 0 to disable identity loss)",
    )
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    p.add_argument("--sample_every", type=int, default=500, help="Save image grid every N steps")
    p.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument(
        "--lr_scheduler",
        type=str,
        choices=("constant", "linear"),
        default="constant",
        help="linear = hold LR for lr_constant_epochs then decay linearly to 0 by the last epoch",
    )
    p.add_argument(
        "--lr_constant_epochs",
        type=int,
        default=None,
        help="With linear schedule: epochs at full LR before decay (default: same as --epochs, i.e. no decay)",
    )
    p.add_argument(
        "--fid_every",
        type=int,
        default=0,
        help="Run FID(real_B, G_ab(A)) every N epochs (0 = disabled; slower, loads Inception)",
    )
    p.add_argument(
        "--fid_max_samples",
        type=int,
        default=5000,
        help="Max images used for FID statistics each evaluation",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.lr_constant_epochs is None:
        args.lr_constant_epochs = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = Path(args.output_dir) / "samples"
    ckpt_dir = Path(args.output_dir) / "checkpoints"

    tf = transforms.Compose(
        [
            transforms.Resize(int(args.img_size * 1.12)),
            transforms.RandomCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageDataset(args.train_a, args.train_b, transform=tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    fid_loader = None
    fid_net = None
    if args.fid_every > 0:
        fid_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        fid_net = load_inception_for_fid(device)
        fid_log_path = Path(args.output_dir) / "fid_log.txt"

    G_ab = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    G_ba = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
    D_a = Discriminator(in_channels=3).to(device)
    D_b = Discriminator(in_channels=3).to(device)

    opt_G = torch.optim.Adam(
        itertools.chain(G_ab.parameters(), G_ba.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_a.parameters(), D_b.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, G_ab, G_ba, D_a, D_b, opt_G, opt_D, device=str(device)
        )
        print(f"Resumed from epoch {start_epoch}")

    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        lr_ep = learning_rate(
            epoch,
            args.epochs,
            args.lr,
            args.lr_scheduler,
            args.lr_constant_epochs,
        )
        for pg in opt_G.param_groups:
            pg["lr"] = lr_ep
        for pg in opt_D.param_groups:
            pg["lr"] = lr_ep

        G_ab.train()
        G_ba.train()
        D_a.train()
        D_b.train()

        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs} lr={lr_ep:.2e}")
        for real_a, real_b in pbar:
            real_a = real_a.to(device, non_blocking=True)
            real_b = real_b.to(device, non_blocking=True)

            # ---------------------
            #  Train discriminators
            # ---------------------
            fake_a = G_ba(real_b)
            fake_b = G_ab(real_a)

            D_a_real = D_a(real_a)
            D_a_fake = D_a(fake_a.detach())
            D_b_real = D_b(real_b)
            D_b_fake = D_b(fake_b.detach())

            real_labels = torch.ones_like(D_a_real)
            fake_labels = torch.zeros_like(D_a_fake)

            loss_D_a = (mse(D_a_real, real_labels) + mse(D_a_fake, fake_labels)) * 0.5
            loss_D_b = (mse(D_b_real, real_labels) + mse(D_b_fake, fake_labels)) * 0.5
            loss_D = loss_D_a + loss_D_b

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # -----------------
            #  Train generators
            # -----------------
            fake_a = G_ba(real_b)
            fake_b = G_ab(real_a)

            D_a_fake_g = D_a(fake_a)
            D_b_fake_g = D_b(fake_b)

            loss_G_adv_a = mse(D_a_fake_g, real_labels)
            loss_G_adv_b = mse(D_b_fake_g, real_labels)

            rec_a = G_ba(fake_b)
            rec_b = G_ab(fake_a)
            loss_cycle = args.lambda_cycle * (l1(rec_a, real_a) + l1(rec_b, real_b))

            loss_id = torch.tensor(0.0, device=device)
            if args.lambda_identity > 0:
                id_b = G_ab(real_b)
                id_a = G_ba(real_a)
                loss_id = (
                    args.lambda_cycle
                    * args.lambda_identity
                    * (l1(id_b, real_b) + l1(id_a, real_a))
                )

            loss_G = loss_G_adv_a + loss_G_adv_b + loss_cycle + loss_id

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            global_step += 1
            pbar.set_postfix(
                D=float(loss_D.detach()),
                G=float(loss_G.detach()),
                cyc=float(loss_cycle.detach()),
            )

            if global_step % args.sample_every == 0:
                G_ab.eval()
                G_ba.eval()
                with torch.no_grad():
                    fake_b_vis = G_ab(real_a[:4])
                    fake_a_vis = G_ba(real_b[:4])
                    grid = torch.cat([real_a[:4], fake_b_vis, real_b[:4], fake_a_vis], dim=0)
                save_samples(samples_dir / f"step_{global_step:07d}.png", grid, nrow=4)
                G_ab.train()
                G_ba.train()

        if (
            args.fid_every > 0
            and fid_loader is not None
            and fid_net is not None
            and (epoch + 1) % args.fid_every == 0
        ):
            fid_val = compute_fid_ab_translation(
                G_ab,
                fid_loader,
                device,
                max_samples=args.fid_max_samples,
                net=fid_net,
            )
            print(f"epoch {epoch + 1}: FID(real_B vs G_ab(A)) = {fid_val:.4f}")
            with open(fid_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch + 1}\t{fid_val:.6f}\n")

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                epoch + 1,
                G_ab,
                G_ba,
                D_a,
                D_b,
                opt_G,
                opt_D,
            )

    save_checkpoint(
        ckpt_dir / "latest.pth",
        args.epochs,
        G_ab,
        G_ba,
        D_a,
        D_b,
        opt_G,
        opt_D,
    )
    print(f"Done. Artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
