import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad


def wasserstein_discriminator_loss(d_real, d_fake, discriminator, real_data, fake_data, lambda_gp=10.0):
    loss = -(d_real.mean() - d_fake.mean())
    gp   = _gradient_penalty(discriminator, real_data, fake_data, lambda_gp)
    return loss + gp, gp


def wasserstein_generator_loss(d_fake):
    return -d_fake.mean()


def _gradient_penalty(discriminator, real_data, fake_data, lambda_gp):
    batch  = real_data.size(0)
    device = real_data.device
    eps    = torch.rand(batch, 1, device=device).expand_as(real_data)
    x_hat  = (eps * real_data + (1 - eps) * fake_data).requires_grad_(True)
    grads  = torch_grad(
        outputs=discriminator(x_hat),
        inputs=x_hat,
        grad_outputs=torch.ones(batch, 1, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    return lambda_gp * ((grads.view(batch, -1).norm(2, dim=1) - 1) ** 2).mean()


_bce = nn.BCELoss()


def bce_discriminator_loss(d_real, d_fake):
    return _bce(d_real, torch.ones_like(d_real)) + _bce(d_fake, torch.zeros_like(d_fake))


def bce_generator_loss(d_fake):
    return _bce(d_fake, torch.ones_like(d_fake))
