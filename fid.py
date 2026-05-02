"""
Fréchet Inception Distance (FID) using Inception v3 activations before the final linear
layer (2048-D), following common GAN evaluation practice.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3

try:
    from torchvision.models import Inception_V3_Weights
except ImportError:
    Inception_V3_Weights = None


def load_inception_for_fid(device: torch.device) -> torch.nn.Module:
    """Inception v3 with fc replaced by Identity; outputs 2048-D features."""
    # torchvision requires aux_logits=True when loading weights; strip aux after load.
    if Inception_V3_Weights is not None:
        w = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=w, transform_input=False).to(device)
    else:
        net = inception_v3(pretrained=True, transform_input=False).to(device)
    net.aux_logits = False
    net.AuxLogits = None
    net.fc = torch.nn.Identity()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _to_inception_input(x: torch.Tensor) -> torch.Tensor:
    """Map tensors in [-1, 1] (B,3,H,W) to Inception-normalized 299×299 inputs."""
    x = (x.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.458, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _features_forward(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return net(_to_inception_input(x))


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(diff.dot(diff) + tr_covmean)


@torch.no_grad()
def compute_fid_given_tensors(
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    net: Optional[torch.nn.Module] = None,
) -> float:
    """
    FID between two sets of images in [-1, 1] (B,3,H,W).
    Typically ``real`` is reference domain B and ``fake`` is G(A).
    """
    net = net or load_inception_for_fid(device)
    n = min(real.shape[0], fake.shape[0])
    real = real[:n]
    fake = fake[:n]
    feats_r = []
    feats_f = []
    bs = min(32, n)
    for i in range(0, n, bs):
        feats_r.append(_features_forward(net, real[i : i + bs].to(device)).cpu().numpy())
        feats_f.append(_features_forward(net, fake[i : i + bs].to(device)).cpu().numpy())
    r = np.concatenate(feats_r, axis=0).astype(np.float64)
    f = np.concatenate(feats_f, axis=0).astype(np.float64)
    mu_r, sigma_r = r.mean(axis=0), np.cov(r, rowvar=False)
    mu_f, sigma_f = f.mean(axis=0), np.cov(f, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


@torch.no_grad()
def compute_fid_ab_translation(
    G_ab: torch.nn.Module,
    data_loader,
    device: torch.device,
    max_samples: int = 5000,
    net: Optional[torch.nn.Module] = None,
) -> float:
    """
    FID between real domain-B images and fake_B = G_ab(real_A).
    """
    net = net or load_inception_for_fid(device)
    G_ab.eval()
    reals = []
    fakes = []
    n = 0
    for real_a, real_b in data_loader:
        real_a = real_a.to(device, non_blocking=True)
        real_b = real_b.to(device, non_blocking=True)
        fake_b = G_ab(real_a)
        reals.append(real_b.cpu())
        fakes.append(fake_b.cpu())
        n += real_b.shape[0]
        if n >= max_samples:
            break
    real = torch.cat(reals, dim=0)[:max_samples]
    fake = torch.cat(fakes, dim=0)[:max_samples]
    return compute_fid_given_tensors(real, fake, device, net=net)
