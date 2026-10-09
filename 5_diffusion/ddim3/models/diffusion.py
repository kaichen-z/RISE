"""Forward (training) and feed-forward DDIM sampling for the two-stage SRHD model.

This release keeps only what a feed-forward model needs:

  * GaussianDiffusionTrainer  -- q(x_t | x_0) + epsilon-prediction MSE loss
  * DDIMSampler               -- deterministic (eta=0) reverse chain, no gradients

The ancestral DDPM sampler and the reverse (latent-noise) optimisation of the
research code are intentionally absent -- see ../../README.md.

Speed notes vs the original implementation:
  * the schedule buffers are float32 (they were float64, which forced every
    `extract` to build a float64 tensor and then cast it back),
  * the DDIM timestep grid is built once in __init__ instead of per call,
  * `extract` uses index_select on a 1-D buffer (no dtype conversion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """Gather schedule entries for a batch of timesteps and broadcast to x."""
    out = v.index_select(0, t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    """Adds noise to x_0 and asks the model to predict it."""

    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas_bar = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('betas', betas.float())
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).float())
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).float())

    def forward(self, x_0, cond, cond2=None):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        predict_noise = self.model(x_t, t, cond) if cond2 is None else self.model(x_t, t, cond, cond2)
        return F.mse_loss(predict_noise, noise, reduction='none')


class DDIMSampler(nn.Module):
    """Feed-forward DDIM sampling.

    `ddim_steps` evenly spaced timesteps are used out of T; eta=0 (the default)
    makes the chain deterministic given x_T.
    """

    def __init__(self, model, beta_1, beta_T, T, img_size=128, mean_type='epsilon',
                 eta=0.0, ddim_steps=20):
        super().__init__()
        assert mean_type == 'epsilon', "only epsilon-prediction is released"
        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.eta = float(eta)
        self.default_steps = ddim_steps
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas_bar = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('alphas_bar', alphas_bar.float())
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar).float())
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1).float())

    def timesteps(self, ddim_steps):
        """Descending list of (t, t_next); identical grid to the original code."""
        step_size = max(self.T // ddim_steps, 1)
        grid = list(reversed(range(0, self.T, step_size)))
        return [(i, max(i - step_size, 0)) for i in grid]

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps)

    def ddim_step(self, x_t, cond1, cond2, t, t_next):
        eps = self.model(x_t, t, cond1) if cond2 is None else self.model(x_t, t, cond1, cond2)
        x_0 = torch.clip(self.predict_xstart_from_eps(x_t, t, eps), -1., 1.)
        alpha_t = extract(self.alphas_bar, t, x_t.shape)
        alpha_t_next = extract(self.alphas_bar, t_next, x_t.shape)
        if self.eta > 0:
            sigma_t = self.eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * \
                torch.sqrt(1 - alpha_t / alpha_t_next)
            noise = torch.randn_like(x_t)
            noise = noise * (t_next > 0).view(-1, *([1] * (x_t.dim() - 1)))
        else:
            sigma_t = torch.zeros_like(alpha_t)
            noise = 0.
        return (torch.sqrt(alpha_t_next) * x_0 +
                torch.sqrt((1 - alpha_t_next - sigma_t ** 2).clamp_min(0)) * eps +
                sigma_t * noise)

    @torch.no_grad()
    def forward(self, x_T, cond1, cond2=None, ddim_steps=None):
        x_t = x_T
        ones = torch.ones(x_T.shape[0], dtype=torch.long, device=x_T.device)
        for i, i_next in self.timesteps(ddim_steps or self.default_steps):
            x_t = self.ddim_step(x_t, cond1, cond2, ones * i, ones * i_next)
        return x_t


# Backwards-compatible alias with the `github` release.
GaussianDiffusionSampler_DDIM = DDIMSampler
