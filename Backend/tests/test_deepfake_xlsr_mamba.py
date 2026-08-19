"""Model C (XLSR-Mamba) — architecture, preprocessing and label tests.

None of these download the 1.3 GB checkpoint; they exercise the pieces we
had to write ourselves, which is exactly where a Tier B integration goes
wrong silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.tasks.deepfake import service, xlsr_mamba


# --- preprocessing --------------------------------------------------------
# The reference pads by tile-repeating the clip. Zero-padding instead would
# load fine and score differently, with nothing raised.


def test_pad_or_tile_truncates_a_long_clip():
    long_clip = np.arange(100_000, dtype=np.float32)

    result = xlsr_mamba.pad_or_tile(long_clip)

    assert result.shape[0] == xlsr_mamba.EVAL_CUT_SAMPLES
    assert np.array_equal(result, long_clip[: xlsr_mamba.EVAL_CUT_SAMPLES])


def test_pad_or_tile_repeats_rather_than_zero_padding():
    short_clip = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = xlsr_mamba.pad_or_tile(short_clip, max_len=8)

    assert np.array_equal(result, np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.float32))
    assert 0.0 not in result  # a zero-pad would have put silence here


def test_pad_or_tile_matches_reference_implementation():
    """Transcribed directly from the reference utils.pad."""

    def reference_pad(x, max_len):
        if x.shape[0] >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x.shape[0]) + 1
        return np.tile(x, (1, num_repeats))[:, :max_len][0]

    for length in (1, 7, 1000, 66_799, 66_800, 70_000):
        clip = np.random.RandomState(length).randn(length).astype(np.float32)
        assert np.array_equal(
            xlsr_mamba.pad_or_tile(clip), reference_pad(clip, xlsr_mamba.EVAL_CUT_SAMPLES)
        )


# --- label order ----------------------------------------------------------


def test_model_c_label_order_is_inverted_relative_to_model_a():
    """The single most dangerous detail in this integration.

    Model A's checkpoint declares {0: real, 1: fake}. Model C has an EMPTY
    config.json, and its training code uses the opposite convention:
    `1 if label == 'bonafide' else 0`, scored as batch_out[:, 1].
    """
    assert xlsr_mamba.BONAFIDE_INDEX == 1
    assert xlsr_mamba.SPOOF_INDEX == 0
    assert xlsr_mamba.SPOOF_INDEX != xlsr_mamba.BONAFIDE_INDEX


def test_model_c_is_registered_as_tier_b():
    spec = service.get_model_spec("xlsr-mamba")

    assert spec.tier == "B"
    assert spec.model_id == "AustinXiao/XLSR-Mamba-LA"
    assert spec.gated is False
    # Tier A models must not be dragged onto the Tier B path.
    assert service.get_model_spec("xlsr-deepfake").tier == "A"
    assert service.get_model_spec("ast-fakeaudio").tier == "A"


def test_all_three_models_are_registered():
    assert set(service.MODEL_SPECS) == {"xlsr-deepfake", "ast-fakeaudio", "xlsr-mamba"}


# --- architecture ---------------------------------------------------------


def test_rms_norm_matches_explicit_formula():
    norm = xlsr_mamba.RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])

    expected = norm.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)

    assert torch.allclose(norm(x), expected)


def test_mamba_mixer_preserves_shape():
    mixer = xlsr_mamba.MambaMixer().eval()
    x = torch.randn(2, 13, xlsr_mamba.D_MODEL)

    with torch.inference_mode():
        out = mixer(x)

    assert out.shape == (2, 13, xlsr_mamba.D_MODEL)
    assert torch.isfinite(out).all()


def test_selective_scan_matches_an_independent_reference():
    """Validate the recurrence itself, not just its output shape.

    This reimplements the scan the long way round (einsum per step over an
    explicitly built state) and checks the mixer agrees. It is the only
    check available that the CUDA kernel we cannot run has been replaced
    with the same arithmetic.
    """
    torch.manual_seed(0)
    mixer = xlsr_mamba.MambaMixer().eval()
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    x = torch.randn(1, 9, xlsr_mamba.D_MODEL)

    with torch.inference_mode():
        actual = mixer(x)

        # --- reference path ---
        import torch.nn.functional as F

        projected = mixer.in_proj(x)
        u, z = projected.chunk(2, dim=-1)
        conv = mixer.conv1d(u.transpose(1, 2))[..., : x.shape[1]]
        u = F.silu(conv).transpose(1, 2)
        dbl = mixer.x_proj(u)
        dt, B, C = torch.split(
            dbl, [xlsr_mamba.DT_RANK, xlsr_mamba.D_STATE, xlsr_mamba.D_STATE], dim=-1
        )
        dt = F.softplus(mixer.dt_proj(dt))
        A = -torch.exp(mixer.A_log)

        state = torch.zeros(1, xlsr_mamba.D_INNER, xlsr_mamba.D_STATE)
        collected = []
        for step in range(x.shape[1]):
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt[:, step], A))
            dB = torch.einsum("bd,bn,bd->bdn", dt[:, step], B[:, step], u[:, step])
            state = dA * state + dB
            collected.append(torch.einsum("bdn,bn->bd", state, C[:, step]))
        y = torch.stack(collected, dim=1) + u * mixer.D
        expected = mixer.out_proj(y * F.silu(z))

    assert torch.allclose(actual, expected, atol=1e-5)


def test_duabimamba_produces_two_logits():
    block = xlsr_mamba.DuaBiMamba().eval()
    x = torch.randn(1, 11, xlsr_mamba.D_MODEL)

    with torch.inference_mode():
        logits = block(x)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_the_scan_is_order_dependent():
    """The recurrence must actually depend on time order.

    This is the property the whole bidirectional design rests on: if the
    scan were order-agnostic, running a second reversed column would be
    pointless and the architecture would be silently half-dead.
    """
    torch.manual_seed(2)
    mixer = xlsr_mamba.MambaMixer().eval()
    x = torch.randn(1, 15, xlsr_mamba.D_MODEL)

    with torch.inference_mode():
        forward = mixer(x)
        # Reverse the input, then undo the reversal on the output. For an
        # order-independent op these would coincide.
        reversed_then_restored = mixer(x.flip([1])).flip([1])

    assert (forward - reversed_then_restored).abs().max() > 1e-3


def test_duabimamba_reads_the_clip_in_both_directions():
    """A time-reversed clip must not give the identical answer."""
    torch.manual_seed(1)
    block = xlsr_mamba.DuaBiMamba().eval()
    x = torch.randn(1, 15, xlsr_mamba.D_MODEL)

    with torch.inference_mode():
        forward_logits = block(x)
        reversed_logits = block(x.flip([1]))

    # Attention pooling is permutation-invariant over time, so the ONLY
    # thing that can make these differ is the order-dependent scan.
    assert (forward_logits - reversed_logits).abs().max() > 1e-5


# --- fairseq -> transformers rename ---------------------------------------


def test_remap_renames_fairseq_transformer_names():
    checkpoint = {
        "ssl_model.model.encoder.layers.0.fc1.weight": torch.zeros(1),
        "ssl_model.model.encoder.layers.0.fc2.bias": torch.zeros(1),
        "ssl_model.model.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "ssl_model.model.encoder.layers.0.self_attn_layer_norm.bias": torch.zeros(1),
        "ssl_model.model.post_extract_proj.weight": torch.zeros(1),
        "ssl_model.model.layer_norm.bias": torch.zeros(1),
        "ssl_model.model.feature_extractor.conv_layers.0.0.weight": torch.zeros(1),
        "ssl_model.model.feature_extractor.conv_layers.1.2.1.bias": torch.zeros(1),
        "conformer.classifier.weight": torch.zeros(1),  # not an ssl tensor
    }

    remapped = xlsr_mamba.remap_ssl_state_dict(checkpoint, target_keys=set())

    assert "encoder.layers.0.feed_forward.intermediate_dense.weight" in remapped
    assert "encoder.layers.0.feed_forward.output_dense.bias" in remapped
    assert "encoder.layers.0.attention.q_proj.weight" in remapped
    assert "encoder.layers.0.layer_norm.bias" in remapped
    assert "feature_projection.projection.weight" in remapped
    assert "feature_projection.layer_norm.bias" in remapped
    assert "feature_extractor.conv_layers.0.conv.weight" in remapped
    assert "feature_extractor.conv_layers.1.layer_norm.bias" in remapped
    # Non-ssl tensors belong to the head and must not leak into the encoder.
    assert not any(key.startswith("conformer") for key in remapped)


def test_remap_drops_pretraining_only_heads():
    checkpoint = {
        "ssl_model.model.quantizer.vars": torch.zeros(1),
        "ssl_model.model.project_q.weight": torch.zeros(1),
        "ssl_model.model.final_proj.bias": torch.zeros(1),
        "ssl_model.model.mask_emb": torch.zeros(1),
    }

    remapped = xlsr_mamba.remap_ssl_state_dict(checkpoint, target_keys=set())

    # Wav2Vec2Model has no quantizer/project_q/final_proj; mask_emb is kept
    # under the name transformers gives it.
    assert set(remapped) == {"masked_spec_embed"}


@pytest.mark.parametrize("parametrized", [True, False])
def test_remap_follows_the_weight_norm_layout_torch_uses(parametrized):
    """torch >= 2.1 stores weight-norm under `parametrizations`."""
    checkpoint = {
        "ssl_model.model.encoder.pos_conv.0.weight_g": torch.zeros(1),
        "ssl_model.model.encoder.pos_conv.0.weight_v": torch.zeros(1),
        "ssl_model.model.encoder.pos_conv.0.bias": torch.zeros(1),
    }
    target = (
        {"encoder.pos_conv_embed.conv.parametrizations.weight.original0"}
        if parametrized
        else set()
    )

    remapped = xlsr_mamba.remap_ssl_state_dict(checkpoint, target_keys=target)

    if parametrized:
        assert "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in remapped
        assert "encoder.pos_conv_embed.conv.parametrizations.weight.original1" in remapped
    else:
        assert "encoder.pos_conv_embed.conv.weight_g" in remapped
        assert "encoder.pos_conv_embed.conv.weight_v" in remapped
    assert "encoder.pos_conv_embed.conv.bias" in remapped
