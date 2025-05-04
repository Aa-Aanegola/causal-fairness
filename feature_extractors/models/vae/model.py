from __future__ import annotations

import os
import yaml
from typing import List, Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VAE", "get_model"]


def _conv_block(cfg: Dict[str, Any], transposed: bool = False) -> nn.Module:
    op_cls = nn.ConvTranspose2d if transposed else nn.Conv2d
    layers: List[nn.Module] = [
        op_cls(**cfg),
        nn.BatchNorm2d(cfg["out_channels"]),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, cfg: Dict[str, Any], beta: float = 1e-3):
        super().__init__()

        enc_spec: Sequence[Dict[str, Any]] = cfg["model"]["encoder"]
        dec_spec: Sequence[Dict[str, Any]] = cfg["model"]["decoder"]
        latent_dim: int = int(cfg["model"]["latent_dim"])
        self.input_shape: Sequence[int] = cfg["model"]["input_shape"]
        self.beta: float = beta

        self.encoder = nn.Sequential(*[_conv_block(layer, False) for layer in enc_spec])
        self.decoder_conv = nn.Sequential(*[_conv_block(layer, True) for layer in dec_spec])

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            fm = self.encoder(dummy)
            self.fm_shape = fm.shape[1:]
            flattened = fm.numel() // fm.shape[0]

        self.enc_mu = nn.Linear(flattened, latent_dim)
        self.enc_logvar = nn.Linear(flattened, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flattened)

        self.final_act = nn.Tanh()

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.encoder(x).flatten(1)
        mu = self.enc_mu(fm)
        return mu

    def forward(self, x: torch.Tensor):
        fm = self.encoder(x).flatten(1)
        mu = self.enc_mu(fm)
        logvar = self.enc_logvar(fm)
        z = self._reparameterize(mu, logvar)

        fm_dec = self.dec_fc(z).view(x.size(0), *self.fm_shape)
        x_rec = self.decoder_conv(fm_dec)
        x_rec = self.final_act(x_rec)
        return x_rec, mu, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl, {
            "recon": recon_loss.detach(),
            "kl": kl.detach(),
        }


def get_model(config: Dict[str, Any]) -> VAE:
    """Instantiate a VAE from a root config dict or file path."""
    if isinstance(config, str):
        with open(os.path.join(config, "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    return VAE(config)