#!/usr/bin/env python3
import sys, time

def main():
    print("Importing...")
    t0 = time.perf_counter()
    import torch
    from src.generator import Generator
    from src.discriminator import Discriminator
    from src.data import get_mnist_dataloader
    print(f"Imports OK ({time.perf_counter()-t0:.2f}s)")

    device = torch.device("cpu")
    g = Generator(latent_dim=64).to(device)
    d = Discriminator(image_size=784, use_sigmoid=True).to(device)

    z   = torch.randn(1, 64)
    out = g(z)
    assert out.shape == (1, 784), out.shape
    print(f"Generator OK  shape={tuple(out.shape)}")

    score = d(out.detach())
    assert score.shape == (1, 1), score.shape
    print(f"Discriminator OK  shape={tuple(score.shape)}")

    loader = get_mnist_dataloader(batch_size=8, subset_size=32)
    imgs, _ = next(iter(loader))
    d(imgs.view(imgs.size(0), -1))
    print("Real batch through D OK")

    out.sum().backward()
    print("Backward OK\n\nAll checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise
