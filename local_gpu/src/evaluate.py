import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, TensorDataset

from src.data import get_mnist_dataloader, flat_to_image, denormalise
from src.generator import Generator


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model.aux_logits = False
        self.model     = model.to(device).eval()
        self.transform = transforms.Resize((299, 299), antialias=True)

    @torch.no_grad()
    def get_features(self, images):
        imgs = self.transform(images.repeat(1, 3, 1, 1)).to(self.device)
        return self.model(imgs).cpu().numpy()


def compute_fid(real_feats, fake_feats):
    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    sqrt_cov, _ = sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * sqrt_cov))


def _collect_features(extractor, images_tensor, batch_size=50):
    feats = []
    for (batch,) in DataLoader(TensorDataset(images_tensor), batch_size=batch_size):
        feats.append(extractor.get_features(batch))
    return np.concatenate(feats, axis=0)


def evaluate_fid_per_class(generator, device, n_samples=500, output_dir="outputs"):
    extractor = InceptionFeatureExtractor(device)
    generator.eval()
    scores = {}

    for digit in range(10):
        print(f"[FID] Class {digit}...")
        loader   = get_mnist_dataloader(batch_size=n_samples, digit=digit, subset_size=n_samples)
        real, _  = next(iter(loader))
        real_01  = denormalise(real.to(device))

        z        = torch.randn(n_samples, generator.latent_dim, device=device)
        with torch.no_grad():
            fake_01 = denormalise(flat_to_image(generator(z)))

        fid = compute_fid(_collect_features(extractor, real_01), _collect_features(extractor, fake_01))
        scores[digit] = fid
        print(f"[FID]   {digit}: {fid:.2f}")

    _plot_fid_bar(scores, output_dir)
    generator.train()
    return scores


def generate_visual_grid(generator, device, n_images=100, output_dir="outputs", filename="generated_grid.png"):
    generator.eval()
    with torch.no_grad():
        imgs = denormalise(flat_to_image(generator(torch.randn(n_images, generator.latent_dim, device=device)))).cpu().numpy()

    nrow = 10
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for i, ax in enumerate(np.array(axes).flatten()):
        ax.imshow(imgs[i, 0] if i < n_images else np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle("LSTM-QGAN Generated Images", fontsize=10)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL] Grid saved to {output_dir}/{filename}")
    generator.train()


def _plot_fid_bar(scores, output_dir):
    classes = list(scores.keys())
    vals    = [scores[c] for c in classes]
    avg     = np.mean(vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(classes, vals, color="steelblue")
    ax.axhline(avg, color="red", linestyle="--", label=f"Avg FID = {avg:.2f}")
    ax.set_xlabel("Digit class")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID per Class — LSTM-QGAN")
    ax.set_xticks(classes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    path = f"{output_dir}/plots/fid_per_class.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[EVAL] FID chart saved to {path}  |  Avg FID: {avg:.2f}")
