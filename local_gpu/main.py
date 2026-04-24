import argparse
import os
import torch

from src.train import train, TrainConfig
from src.generator import Generator
from src.evaluate import evaluate_fid_per_class, generate_visual_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         choices=["train", "evaluate", "both"], default="both")
    p.add_argument("--loss",         choices=["wasserstein", "bce"],        default="wasserstein")
    p.add_argument("--epochs",       type=int,   default=1000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--n_critic",     type=int,   default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--latent_dim",   type=int,   default=64)
    p.add_argument("--subset",       type=int,   default=None)
    p.add_argument("--output_dir",   type=str,   default="outputs")
    p.add_argument("--checkpoint",   type=str,   default=None)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--fid_samples",  type=int,   default=500)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--ckpt_interval",type=int,   default=10)
    return p.parse_args()


def run_train(args):
    cfg = TrainConfig()
    cfg.n_epochs      = args.epochs
    cfg.batch_size    = args.batch_size
    cfg.lr            = args.lr
    cfg.latent_dim    = args.latent_dim
    cfg.loss_type     = args.loss
    cfg.output_dir    = args.output_dir
    cfg.device        = args.device
    cfg.subset_size   = args.subset
    cfg.n_critic      = args.n_critic
    cfg.resume        = args.resume
    cfg.ckpt_interval = args.ckpt_interval

    train(cfg)

    device = torch.device(args.device)
    g = Generator(latent_dim=args.latent_dim).to(device)
    g.load_state_dict(torch.load(f"{args.output_dir}/generator_final.pth", map_location=device))
    return g


def run_evaluate(args, generator=None):
    device = torch.device(args.device)

    if generator is None:
        ckpt = args.checkpoint or f"{args.output_dir}/generator_final.pth"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        generator = Generator(latent_dim=args.latent_dim).to(device)
        generator.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[MAIN] Loaded {ckpt}")

    generate_visual_grid(generator, device, output_dir=args.output_dir)
    scores = evaluate_fid_per_class(generator, device, n_samples=args.fid_samples, output_dir=args.output_dir)

    avg = sum(scores.values()) / len(scores)
    print(f"\n[MAIN] FID per class: { {k: round(v,2) for k,v in scores.items()} }")
    print(f"[MAIN] Average FID: {avg:.2f}")


def main():
    args = parse_args()
    print(f"LSTM-QGAN | mode={args.mode} | loss={args.loss} | device={args.device}")

    if args.mode == "train":
        run_train(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:
        run_evaluate(args, generator=run_train(args))


if __name__ == "__main__":
    main()
