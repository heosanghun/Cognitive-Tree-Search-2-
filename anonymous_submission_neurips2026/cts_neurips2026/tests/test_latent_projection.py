"""Tests for Wproj Latent-to-Text Projection (paper §4.5)."""

import torch

from cts.latent.bottleneck import (
    LatentProjection,
    LatentDecoder,
    add_exploration_noise,
    init_z0,
    validate_information_retention,
)


def test_latent_projection_shape():
    proj = LatentProjection(latent_dim=64, model_dim=256)
    z = torch.randn(8, 64)
    out = proj(z)
    assert out.shape == (8, 256)


def test_latent_decoder_logits():
    dec = LatentDecoder(latent_dim=64, model_dim=128, vocab_size=1000)
    z = torch.randn(8, 64)
    logits = dec.greedy_logits(z)
    assert logits.shape == (8, 1000)


def test_latent_decoder_soft_prompt():
    dec = LatentDecoder(latent_dim=64, model_dim=128, vocab_size=1000)
    z = torch.randn(8, 64)
    prompt = dec.project_to_soft_prompt(z)
    assert prompt.shape == (8, 128)


def test_add_exploration_noise():
    gen = torch.Generator().manual_seed(42)
    z0 = init_z0(8, 64, torch.device("cpu"), gen)
    gen2 = torch.Generator().manual_seed(42)
    z0_copy = init_z0(8, 64, torch.device("cpu"), gen2)
    gen3 = torch.Generator().manual_seed(99)
    z_noised = add_exploration_noise(z0, nu_expl=2.0, generator=gen3)
    assert not torch.allclose(z0_copy, z_noised)


def test_validate_information_retention():
    dec = LatentDecoder(latent_dim=64, model_dim=128, vocab_size=100)
    z = torch.randn(8, 64)
    ref_tokens = torch.randint(0, 100, (8,))
    result = validate_information_retention(z, dec, ref_tokens, threshold=0.0)
    assert "match_rate" in result
    assert "passed" in result
    assert isinstance(result["match_rate"], float)
