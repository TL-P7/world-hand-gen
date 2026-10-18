from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.wrappers.hamer_wrapper import HaMeRDecoderTokenWrapper
from src.model.wrappers.vggt_omega_wrapper import VGGTOmegaCameraTokenWrapper


class _FakeHamerBackbone(nn.Module):
    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        assert crops.shape[-2:] == (256, 192)
        return torch.zeros(crops.shape[0], 8, 2, 2, device=crops.device)


class _FakeHamerTransformer(nn.Module):
    def forward(self, query: torch.Tensor, *, context: torch.Tensor) -> torch.Tensor:
        assert query.shape[1:] == (1, 1)
        assert context.shape[0] == query.shape[0]
        return torch.zeros(query.shape[0], 1, 1024, device=query.device)


class _FakeHamerHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _FakeHamerTransformer()


class _FakeHamer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeHamerBackbone()
        self.mano_head = _FakeHamerHead()


class _FakeOmega(nn.Module):
    def __init__(self, token_count: int = 17) -> None:
        super().__init__()
        self.token_count = token_count
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_shape = tuple(images.shape)
        batch, time = images.shape[:2]
        return {
            "camera_and_register_tokens": torch.zeros(
                batch, time, self.token_count, 2048, device=images.device
            )
        }


def _hamer() -> HaMeRDecoderTokenWrapper:
    wrapper = HaMeRDecoderTokenWrapper.__new__(HaMeRDecoderTokenWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = _FakeHamer()
    wrapper.model.requires_grad_(False)
    wrapper.eval()
    return wrapper


def _vggt(token_count: int = 17, resolution: int = 8) -> VGGTOmegaCameraTokenWrapper:
    wrapper = VGGTOmegaCameraTokenWrapper.__new__(VGGTOmegaCameraTokenWrapper)
    nn.Module.__init__(wrapper)
    wrapper.image_resolution = resolution
    wrapper.model = _FakeOmega(token_count)
    wrapper.model.requires_grad_(False)
    wrapper.eval()
    return wrapper


def test_hamer_returns_decoder_token_and_stays_frozen() -> None:
    wrapper = _hamer()
    token = wrapper(torch.rand(2, 3, 256, 256))
    assert token.shape == (2, 1024)
    assert wrapper.training is False
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_hamer_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\[N,3,256,256\]"):
        _hamer()(torch.rand(2, 3, 224, 224))


def test_vggt_resizes_and_returns_all_tokens() -> None:
    wrapper = _vggt()
    tokens = wrapper(torch.rand(2, 3, 3, 4, 6))
    assert tokens.shape == (2, 3, 17, 2048)
    assert wrapper.model.seen_shape == (2, 3, 3, 8, 8)
    assert wrapper.training is False
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_vggt_rejects_invalid_images() -> None:
    wrapper = _vggt()
    with pytest.raises(ValueError, match="finite floating point"):
        wrapper(torch.full((1, 1, 3, 8, 8), float("nan")))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        wrapper(torch.full((1, 1, 3, 8, 8), 2.0))
    with pytest.raises(ValueError, match=r"\[B,T,3,H,W\]"):
        wrapper(torch.rand(1, 3, 8, 8))


def test_vggt_requires_seventeen_tokens() -> None:
    with pytest.raises(ValueError, match="17,2048"):
        _vggt(token_count=1)(torch.rand(1, 1, 3, 8, 8))


def test_vggt_state_dict_unwraps_supported_payloads() -> None:
    state = {"weight": torch.ones(1)}
    assert VGGTOmegaCameraTokenWrapper._state_dict(state) is state
    for key in ("state_dict", "model", "module"):
        payload = {key: state}
        assert VGGTOmegaCameraTokenWrapper._state_dict(payload) is state
    with pytest.raises(ValueError, match="tensor state_dict"):
        VGGTOmegaCameraTokenWrapper._state_dict({"state_dict": {"x": 1}})
