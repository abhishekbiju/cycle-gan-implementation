# CycleGAN — Unpaired Image-to-Image Translation

This repository implements **CycleGAN** from [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593) (Zhu et al., ICCV 2017). Training uses **unpaired** images from two domains $A$ and $B$: there is no one-to-one correspondence between training samples.

---

## What problem does CycleGAN solve?

Classical image-to-image translation (e.g. pix2pix) assumes **paired** examples $(x_i, y_i)$. Many tasks only offer **two collections** of images—photos of horses and photos of zebras—without aligned pairs. CycleGAN learns mappings between domains using:

1. **Adversarial losses** so generated images look like the target domain.
2. **Cycle consistency** so mappings are approximate inverses: translating $A \to B$ and back should recover the original $A$ (and symmetrically for $B$).

Without cycle consistency, mapping $A \to B$ is severely **under-constrained** (many outputs can look “real” in $B$ while ignoring input structure). Cycle consistency ties the two mappings together.

---

## Key ideas from the paper

### Two generators and two discriminators

- **$G_{AB}$** maps domain $A \to B$; **$G_{BA}$** maps $B \to A$.
- **$D_B$** discriminates real $B$ vs. fake $G_{AB}(A)$; **$D_A$** discriminates real $A$ vs. fake $G_{BA}(B)$.

Each domain has its own discriminator so distribution matching is enforced **per domain**.

### Adversarial objective

The paper uses adversarial training so $G_{AB}(A)$ is indistinguishable from images drawn from $B$, and similarly for $G_{BA}(B)$ and domain $A$. This implementation follows the **least-squares GAN (LSGAN)** formulation from the authors’ [PyTorch reference](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): discriminators output **logits** over spatial patches, and we minimize **mean squared error** to target labels $1$ for real and $0$ for fake. That tends to produce stable training compared to the original GAN cross-entropy.

### PatchGAN discriminator

Instead of a single real/fake score for the whole image, **PatchGAN** classifies **local patches** (the network’s receptive field over the feature map). The output is a grid of predictions (e.g. $30 \times 30$ for $256 \times 256$ inputs in this repo). This encourages **high-frequency** structure and texture to match the target domain while keeping the architecture lightweight.

In code, `Discriminator` stacks strided convolutions and ends with a $1 \times 1$ conv to one channel—**no sigmoid**; LSGAN uses raw logits with `MSELoss`.

### Cycle consistency loss

Let $x \sim A$ and $y \sim B$. Forward cycle: $x \to G_{AB}(x) \to G_{BA}(G_{AB}(x)) \approx x$. Backward cycle: $y \to G_{BA}(y) \to G_{AB}(G_{BA}(y)) \approx y$.

The paper uses **L1** between reconstructions and inputs, scaled by a weight $\lambda_{\text{cycle}}$ (default **10** here, as in common setups). This is the main constraint that preserves **content/layout** while appearance changes.

### Optional identity loss

For tasks where **colors** should stay aligned (e.g. season transfer), the paper optionally adds an **identity** term: $G_{BA}(y) \approx y$ and $G_{AB}(x) \approx x$ when feeding an image from the “wrong” domain through the generator that maps *into* that domain. In this repo, identity loss is weighted by `lambda_cycle * lambda_identity`; set `--lambda_identity 0` to disable it.

### Generator architecture

The generator is an encoder–**residual**–decoder stack (ResNet style): downsampling $\to$ several **residual blocks** $\to$ upsampling, with **instance normalization** and **reflection padding** at conv boundaries as in the paper’s implementation. Outputs are in **$[-1, 1]$** via `Tanh`.

---

## Repository layout

| File | Role |
|------|------|
| `generator.py` | `ConvBlock`, `ResidualBlock`, `Generator` |
| `discriminator.py` | PatchGAN-style `Discriminator` |
| `dataset.py` | Unpaired `ImageDataset` (one random $A$, one random $B$ per step) |
| `train.py` | LSGAN + cycle (+ optional identity), LR schedule, optional FID, checkpoints, samples |
| `infer.py` | Run $G_{AB}$ on a folder of images using a checkpoint |
| `fid.py` | Inception-v3 features and Fréchet Inception Distance |
| `eval_fid.py` | Standalone FID between $\mathrm{real}_B$ and $G_{AB}(A)$ for a checkpoint |
| `utils.py` | Checkpoint I/O and sample saving |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data layout

Put **all** images from domain $A$ in one folder and **all** from domain $B$ in another. Filenames do not need to match. Supported extensions include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

Example:

```text
data/
  trainA/   # horses, winter photos, etc.
  trainB/   # zebras, summer photos, etc.
```

---

## Training

```bash
python train.py \
  --train_a data/trainA \
  --train_b data/trainB \
  --output_dir runs/my_experiment \
  --epochs 100 \
  --batch_size 1 \
  --lambda_cycle 10 \
  --lambda_identity 0.5
```

Useful flags:

- `--img_size` — crop size (default `256`; generator assumes three stride-2 downsamples, so use sizes divisible appropriately; `256` matches the reference design).
- `--sample_every` — save an image grid under `output_dir/samples/` every $N$ optimizer steps.
- `--checkpoint_every` — save `checkpoints/epoch_XXXX.pth`.
- `--resume path/to/checkpoint.pth` — continue training.
- **Learning rate** — `--lr_scheduler constant` (default) keeps `--lr` fixed for all epochs. Use `--lr_scheduler linear` with `--lr_constant_epochs K` where $K$ is less than the value of `--epochs`: training holds the initial LR for the first $K$ epochs (0-based: epochs `0 … K-1`), then **linearly decays** to **0** by the final epoch (same recipe as the reference [CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) script). Example: `--epochs 200 --lr_scheduler linear --lr_constant_epochs 100`.
- **FID** — `--fid_every N` runs **Fréchet Inception Distance** every $N$ epochs (0 = off). Compares **real images from domain $B$** to **$G_{AB}(A)$** using Inception v3 pool features (2048-D). Uses at most `--fid_max_samples` images per evaluation; append-only log at `output_dir/fid_log.txt`. First run downloads ImageNet weights for Inception (requires network cache).

Checkpoints include both generators, both discriminators, and both optimizers.

### Learning rate schedule (detail)

At the **start** of each epoch, both Adam optimizers get the same LR from `learning_rate(epoch, …)`. With the **linear** schedule, if total epochs is $E$ and `lr_constant_epochs` is $K$, then for epoch index $e \in \{0,\ldots,E-1\}$: full LR while $e < K$, then linear decay so LR reaches **0** when $e = E-1$.

### Fréchet Inception Distance (FID)

FID compares the distribution of **2048-dimensional Inception-v3 activations** (before the classifier) for two image sets. Lower is usually better. Here we report $\mathrm{FID}(\mathrm{real}_B,\, G_{AB}(A))$, measuring how close generated “$B$-looking” outputs are to real $B$ under this metric. **Note:** FID is sensitive to resolution and preprocessing; comparisons are meaningful when setup matches.

Standalone evaluation (center-crop, no random augmentation):

```bash
python eval_fid.py \
  --train_a data/trainA \
  --train_b data/trainB \
  --checkpoint runs/my_experiment/checkpoints/latest.pth \
  --max_samples 5000
```

---

## Inference (A → B)

After training, `G_AB` maps domain $A \to B$. Example:

```bash
python infer.py \
  --input_dir path/to/test_images_A \
  --output_dir path/to/translated_B \
  --checkpoint runs/my_experiment/checkpoints/latest.pth
```

---

## Relation to the paper and reference code

The mathematical objective and architectures follow [arXiv:1703.10593](https://arxiv.org/abs/1703.10593). Hyperparameters (e.g. $\lambda_{\text{cycle}}$, Adam $\beta_1=0.5$, lr $2\times10^{-4}$) align with the commonly released implementation. For datasets, preprocessing recipes, and paper figures, see the [official PyTorch CycleGAN and pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

---

## Citation

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={ICCV},
  year={2017}
}
```
