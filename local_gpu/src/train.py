import os
import random
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.generator import Generator
from src.discriminator import Discriminator
from src.losses import (
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
    bce_discriminator_loss,
    bce_generator_loss,
)
from src.data import get_mnist_dataloader, flat_to_image, denormalise

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainConfig:
    latent_dim:    int   = 64
    patch_size:    int   = 196
    n_steps:       int   = 4
    image_size:    int   = 784
    batch_size:    int   = 128
    lr:            float = 2e-4
    beta1:         float = 0.5
    beta2:         float = 0.9
    n_epochs:      int   = 1000
    n_critic:      int   = 5
    lambda_gp:     float = 10.0
    loss_type:     str   = "wasserstein"
    save_interval: int   = 10
    ckpt_interval: int   = 10
    output_dir:    str   = "outputs"
    device:        str   = "cpu"
    subset_size:   int | None = None
    resume:        bool  = False


def _save_grid(images, path, nrow=10, title=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n    = min(images.shape[0], nrow * nrow)
    imgs = images[:n].cpu().detach().numpy()
    ncols, nrows = nrow, (n + nrow - 1) // nrow

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(np.array(axes).flatten()):
        if i < n:
            ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def train(cfg=None):
    if cfg is None:
        cfg = TrainConfig()

    set_seed(SEED)
    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/generated_images", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/plots", exist_ok=True)

    print(f"[TRAIN] device={device}  loss={cfg.loss_type}  epochs={cfg.n_epochs}  bs={cfg.batch_size}")

    dataloader = get_mnist_dataloader(batch_size=cfg.batch_size, subset_size=cfg.subset_size)
    print(f"[TRAIN] batches/epoch={len(dataloader)}")

    generator     = Generator(latent_dim=cfg.latent_dim, patch_size=cfg.patch_size, n_steps=cfg.n_steps).to(device)
    discriminator = Discriminator(image_size=cfg.image_size, use_sigmoid=(cfg.loss_type == "bce")).to(device)

    print(f"[TRAIN] G params={sum(p.numel() for p in generator.parameters()):,}  "
          f"D params={sum(p.numel() for p in discriminator.parameters()):,}")

    opt_g = optim.Adam(generator.parameters(),     lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    g_losses, d_losses = [], []
    fixed_noise  = torch.randn(100, cfg.latent_dim, device=device)
    start_epoch  = 1
    ckpt_path    = f"{cfg.output_dir}/checkpoint_latest.pth"

    if cfg.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        g_losses    = ckpt.get("g_losses", [])
        d_losses    = ckpt.get("d_losses", [])
        fixed_noise = ckpt.get("fixed_noise", fixed_noise).to(device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[TRAIN] Resumed from epoch {start_epoch}")
    elif cfg.resume:
        print(f"[TRAIN] No checkpoint at {ckpt_path}, starting fresh.")

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        g_ep, d_ep = [], []

        for real_imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch:4d}/{cfg.n_epochs}", leave=False):
            real_imgs  = real_imgs.view(real_imgs.size(0), -1).to(device)
            batch_size = real_imgs.size(0)

            for _ in range(cfg.n_critic):
                opt_d.zero_grad()
                z = torch.randn(batch_size, cfg.latent_dim, device=device)
                fake = generator(z).detach()
                d_real, d_fake = discriminator(real_imgs), discriminator(fake)
                if cfg.loss_type == "wasserstein":
                    loss_d, _ = wasserstein_discriminator_loss(d_real, d_fake, discriminator, real_imgs, fake, cfg.lambda_gp)
                else:
                    loss_d = bce_discriminator_loss(d_real, d_fake)
                loss_d.backward()
                opt_d.step()
            d_ep.append(loss_d.item())

            opt_g.zero_grad()
            z = torch.randn(batch_size, cfg.latent_dim, device=device)
            d_fake_g = discriminator(generator(z))
            loss_g = wasserstein_generator_loss(d_fake_g) if cfg.loss_type == "wasserstein" else bce_generator_loss(d_fake_g)
            loss_g.backward()
            opt_g.step()
            g_ep.append(loss_g.item())

        mg, md = float(np.mean(g_ep)), float(np.mean(d_ep))
        g_losses.append(mg)
        d_losses.append(md)
        print(f"Epoch {epoch:4d}/{cfg.n_epochs}  |  G: {mg:+.4f}  D: {md:+.4f}")

        if epoch % cfg.save_interval == 0 or epoch == 1 or epoch == cfg.n_epochs:
            generator.eval()
            with torch.no_grad():
                samples = denormalise(flat_to_image(generator(fixed_noise)))
            _save_grid(samples, f"{cfg.output_dir}/generated_images/epoch_{epoch:04d}.png", title=f"Epoch {epoch}")
            generator.train()

        if epoch % cfg.ckpt_interval == 0 or epoch == cfg.n_epochs:
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_losses": g_losses,
                "d_losses": d_losses,
                "fixed_noise": fixed_noise.cpu(),
            }, ckpt_path)
            print(f"[TRAIN] Checkpoint saved (epoch {epoch})")
            _plot_losses(g_losses, d_losses, cfg)

    torch.save(generator.state_dict(),     f"{cfg.output_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{cfg.output_dir}/discriminator_final.pth")
    print("[TRAIN] Done.")
    _plot_losses(g_losses, d_losses, cfg)
    return {"g_losses": g_losses, "d_losses": d_losses, "epochs": list(range(1, cfg.n_epochs + 1))}


def _plot_losses(g_losses, d_losses, cfg):
    epochs = range(1, len(g_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, g_losses, label="Generator",     color="tab:blue")
    ax.plot(epochs, d_losses, label="Discriminator", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss — {cfg.loss_type.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"{cfg.output_dir}/plots/loss_{cfg.loss_type}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[TRAIN] Loss plot saved to {path}")
