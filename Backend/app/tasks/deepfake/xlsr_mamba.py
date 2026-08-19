"""XLSR-Mamba (Model C) - a CPU-only, pure-PyTorch reimplementation.

WHY THIS FILE EXISTS
--------------------
Model C is Tier B: the checkpoint (AustinXiao/XLSR-Mamba-LA, MIT) ships
weights and an EMPTY config.json, so nothing describes the architecture. The
reference implementation (github.com/swagshaw/XLSR-Mamba, MIT) additionally
depends on `mamba-ssm`, `causal-conv1d` and `fairseq`, none of which can be
installed here: mamba-ssm publishes no wheel for this platform, needs nvcc to
build from source, and its kernels require a CUDA device. This machine runs a
CPU-only torch build.

The way through is that Mamba's selective scan (Xiao and Das, IEEE SPL 2025,
eq. 1: h_t = A_bar h_{t-1} + B_bar x_t) is just a linear recurrence. The CUDA
kernel is a SPEED optimisation, not a semantic one, so the same maths runs as
an ordinary loop in pure PyTorch. That is what this module does: it declares
the architecture to match the checkpoint's own parameter names and shapes, so
the trained weights load with strict=True.

Everything here is transcribed from the reference implementation's actual
behaviour, not guessed. The points that would otherwise fail SILENTLY are
marked QUIRK below.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Architecture constants, from the checkpoint's own tensor shapes and the
# reference's MambaConfig(d_model=emb_size, n_layer=num_encoders // 2).
# The released checkpoint is ES144_NE12 -> emb_size 144, num_encoders 12.
D_MODEL = 144
N_LAYER = 6  # per column; 6 forward + 6 backward = the paper's 12 blocks
D_STATE = 16
D_CONV = 4
EXPAND = 2
D_INNER = D_MODEL * EXPAND  # 288
DT_RANK = math.ceil(D_MODEL / 16)  # 9
SSL_DIM = 1024
NORM_EPS = 1e-5

# QUIRK: the reference pads by TILING the clip, not with zeros, and cuts at
# 66800 samples (~4.2 s) in the ASVspoof eval loader. Zero-padding instead
# would change the scores without any error being raised.
EVAL_CUT_SAMPLES = 66_800

# QUIRK: labels are inverted relative to Model A. data_utils.genSpoof_list
# assigns `1 if label == 'bonafide' else 0`, and main.produce_evaluation_file
# scores with `batch_out[:, 1]`. So index 1 is BONA FIDE and index 0 is SPOOF
# - the opposite of Model A's {0: real, 1: fake}.
BONAFIDE_INDEX = 1
SPOOF_INDEX = 0


def pad_or_tile(waveform, max_len: int = EVAL_CUT_SAMPLES):
    """Reference `utils.pad`: truncate, or tile-repeat until long enough."""
    import numpy as np

    waveform = np.asarray(waveform)
    if waveform.shape[0] >= max_len:
        return waveform[:max_len]
    repeats = int(max_len / waveform.shape[0]) + 1
    return np.tile(waveform, (1, repeats))[:, :max_len][0]


class RMSNorm(nn.Module):
    """Weight-only RMS norm, matching mamba_ssm's (no bias)."""

    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.eps)


class MambaMixer(nn.Module):
    """One Mamba block, parameter-for-parameter with mamba_ssm's Mamba.

    The scan is a sequential loop instead of the fused CUDA kernel: same
    arithmetic, no GPU required.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, D_INNER * 2, bias=False)
        self.conv1d = nn.Conv1d(
            D_INNER,
            D_INNER,
            kernel_size=D_CONV,
            groups=D_INNER,
            padding=D_CONV - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(D_INNER, DT_RANK + D_STATE * 2, bias=False)
        self.dt_proj = nn.Linear(DT_RANK, D_INNER, bias=True)
        self.A_log = nn.Parameter(torch.zeros(D_INNER, D_STATE))
        self.D = nn.Parameter(torch.ones(D_INNER))
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)

    def forward(self, hidden_states):
        batch, length, _ = hidden_states.shape

        projected = self.in_proj(hidden_states)
        x, z = projected.chunk(2, dim=-1)

        # Depthwise causal conv: left-pad by D_CONV-1, then drop the overhang.
        x = self.conv1d(x.transpose(1, 2))[..., :length]
        x = F.silu(x).transpose(1, 2)

        # Selection: dt, B and C are all input-dependent, which is what makes
        # the state space model "selective".
        projected_state = self.x_proj(x)
        dt, B, C = torch.split(projected_state, [DT_RANK, D_STATE, D_STATE], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        # Discretise, then run the recurrence over time.
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        state = torch.zeros(batch, D_INNER, D_STATE, dtype=x.dtype, device=x.device)
        outputs = []
        for step in range(length):
            state = dA[:, step] * state + dBx[:, step]
            outputs.append((state * C[:, step].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1)

        y = y + x * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm block that passes the running residual onward.

    Matches mamba_ssm's Block: returns (mixer output, new residual) rather
    than adding them, so the caller owns the residual stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mixer = MambaMixer()
        self.norm = RMSNorm(D_MODEL)

    def forward(self, hidden_states, residual):
        residual = hidden_states if residual is None else hidden_states + residual
        return self.mixer(self.norm(residual)), residual


class DuaBiMamba(nn.Module):
    """The paper's dual-column bidirectional Mamba (reference MixerModel).

    Named `conformer` in the checkpoint because the reference adapted it from
    XLSR-Conformer; the parameter names are kept identical so the released
    weights load unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forward_layers = nn.ModuleList(MambaBlock() for _ in range(N_LAYER))
        self.backward_layers = nn.ModuleList(MambaBlock() for _ in range(N_LAYER))
        self.norm_f = RMSNorm(D_MODEL)
        self.f_attention_pool = nn.Linear(D_MODEL, 1)
        self.b_attention_pool = nn.Linear(D_MODEL, 1)
        self.LL = nn.Linear(D_MODEL * 2, D_MODEL)
        self.classifier = nn.Linear(D_MODEL, 2)

    def forward(self, hidden_states):
        forward_states = hidden_states
        backward_states = hidden_states.flip([1])

        forward_residual = None
        for layer in self.forward_layers:
            forward_states, forward_residual = layer(forward_states, forward_residual)

        backward_residual = None
        for layer in self.backward_layers:
            backward_states, backward_residual = layer(backward_states, backward_residual)

        # QUIRK: the reference runs with fused_add_norm=True (set in its own
        # MambaConfig), and on that path the final norm is applied to
        # `hidden_states + residual` - the ORIGINAL block input, not the last
        # layer's output. Its own comment calls this "use long-range features
        # as the key for residual connections". Using the last layer's output
        # here instead would load fine and score subtly wrong.
        forward_states = self.norm_f(hidden_states + forward_residual)
        backward_states = self.norm_f(hidden_states + backward_residual)

        forward_pooled = self._attention_pool(self.f_attention_pool, forward_states)
        backward_pooled = self._attention_pool(self.b_attention_pool, backward_states)

        pooled = torch.cat((forward_pooled, backward_pooled), dim=1)
        return self.classifier(self.LL(pooled))

    @staticmethod
    def _attention_pool(scorer, states):
        weights = F.softmax(scorer(states), dim=1)
        return torch.matmul(weights.transpose(-1, -2), states).squeeze(-2)


class XLSRMamba(nn.Module):
    """Full pipeline: XLS-R front end -> linear projection -> DuaBiMamba."""

    def __init__(self, ssl_model) -> None:
        super().__init__()
        self.ssl_model = ssl_model
        self.LL = nn.Linear(SSL_DIM, D_MODEL)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = DuaBiMamba()

    def forward(self, waveform):
        features = self.ssl_model(waveform).last_hidden_state
        projected = self.LL(features)
        normed = self.first_bn(projected.unsqueeze(dim=1))
        return self.conformer(self.selu(normed).squeeze(dim=1))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
# The checkpoint stores the XLS-R front end with FAIRSEQ parameter names
# (fc1/fc2/self_attn), because the reference loads xlsr2_300m.pt through
# fairseq. We rebuild it as a transformers Wav2Vec2Model instead -- same
# architecture, installable here -- which means renaming every ssl tensor.
# The rename is verified by loading with strict=True: any name we get wrong
# surfaces as a missing/unexpected key rather than as a silent accuracy loss.

SSL_PREFIX = "ssl_model.model."

_LAYER_RENAMES = {
    "self_attn.q_proj": "attention.q_proj",
    "self_attn.k_proj": "attention.k_proj",
    "self_attn.v_proj": "attention.v_proj",
    "self_attn.out_proj": "attention.out_proj",
    "self_attn_layer_norm": "layer_norm",
    "fc1": "feed_forward.intermediate_dense",
    "fc2": "feed_forward.output_dense",
    "final_layer_norm": "final_layer_norm",
}

# Pretraining-only heads: present in the checkpoint, absent from Wav2Vec2Model.
_SSL_DISCARD_PREFIXES = ("quantizer.", "project_q.", "final_proj.")


def remap_ssl_state_dict(state_dict: dict, target_keys) -> dict:
    """Rename fairseq wav2vec2 tensors to their transformers equivalents."""
    import re

    target_keys = set(target_keys)
    # torch >= 2.1 stores weight-norm under `parametrizations`; older builds
    # use the flat weight_g/weight_v names. Pick whichever this build wants.
    parametrized = any("parametrizations" in key for key in target_keys)
    pos_conv = "encoder.pos_conv_embed.conv"
    weight_norm_names = (
        {"weight_g": f"{pos_conv}.parametrizations.weight.original0",
         "weight_v": f"{pos_conv}.parametrizations.weight.original1"}
        if parametrized
        else {"weight_g": f"{pos_conv}.weight_g", "weight_v": f"{pos_conv}.weight_v"}
    )

    remapped: dict = {}
    for key, tensor in state_dict.items():
        if not key.startswith(SSL_PREFIX):
            continue
        name = key[len(SSL_PREFIX):]

        if name.startswith(_SSL_DISCARD_PREFIXES):
            continue

        if name == "mask_emb":
            remapped["masked_spec_embed"] = tensor
            continue

        if name.startswith("post_extract_proj."):
            remapped[name.replace("post_extract_proj.", "feature_projection.projection.")] = tensor
            continue

        if name.startswith("layer_norm."):
            remapped[name.replace("layer_norm.", "feature_projection.layer_norm.")] = tensor
            continue

        if name.startswith("feature_extractor.conv_layers."):
            # `.0` is the conv itself; `.2.1` is the LayerNorm inside the
            # Sequential(TransposeLast, LayerNorm, TransposeLast) wrapper.
            converted = re.sub(r"^(feature_extractor\.conv_layers\.\d+)\.0\.", r"\1.conv.", name)
            converted = re.sub(r"^(feature_extractor\.conv_layers\.\d+)\.2\.1\.", r"\1.layer_norm.", converted)
            remapped[converted] = tensor
            continue

        if name.startswith("encoder.pos_conv."):
            suffix = name.rsplit(".", 1)[-1]
            remapped[weight_norm_names.get(suffix, f"{pos_conv}.{suffix}")] = tensor
            continue

        if name.startswith("encoder.layers."):
            match = re.match(r"^encoder\.layers\.(\d+)\.(.+?)\.(weight|bias)$", name)
            if match:
                index, module, param = match.groups()
                renamed = _LAYER_RENAMES.get(module)
                if renamed:
                    remapped[f"encoder.layers.{index}.{renamed}.{param}"] = tensor
                    continue

        # encoder.layer_norm.* and anything else already matches.
        remapped[name] = tensor

    return remapped


XLSR_BASE_ID = "facebook/wav2vec2-xls-r-300m"


def build_ssl_model():
    """An empty XLS-R 300m body (config only -- weights come from Model C)."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    config = Wav2Vec2Config.from_pretrained(XLSR_BASE_ID)
    # The checkpoint carries no adapter/spec-augment state, and we never mask
    # at inference time.
    config.apply_spec_augment = False
    return Wav2Vec2Model(config)


def load_xlsr_mamba(repo_id: str, revision: str = "main", token: str | None = None):
    """Build the model and load the released checkpoint into it, strictly."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(
        repo_id, "model.safetensors", revision=revision,
        **({"token": token} if token else {}),
    )
    checkpoint = load_file(path)

    ssl_model = build_ssl_model()
    ssl_state = remap_ssl_state_dict(checkpoint, ssl_model.state_dict().keys())
    # strict=True is the whole safety argument for the rename: 429 tensors
    # have to land on exactly the right parameters or this raises.
    ssl_model.load_state_dict(ssl_state, strict=True)

    model = XLSRMamba(ssl_model)
    head_state = {
        key: value for key, value in checkpoint.items() if not key.startswith(SSL_PREFIX)
    }
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    # Everything outside ssl_model must be accounted for; only ssl_model.*
    # may appear as "missing" here since it was loaded separately above.
    unaccounted = [key for key in missing if not key.startswith("ssl_model.")]
    if unaccounted or unexpected:
        raise RuntimeError(
            f"XLSR-Mamba head did not load cleanly. missing={unaccounted} "
            f"unexpected={list(unexpected)}"
        )

    model.eval()
    return model
