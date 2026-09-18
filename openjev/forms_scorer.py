"""openjev.forms_scorer — tiny byte-level option scorer for form/UI decisions.

Contract: a text context plus a list of typed option strings in, one
probability per option out. No generation — a single forward pass scores all
options, and the caller argmax-executes.

Vendored from trycua/cua libs/cua-s1 (MIT), which adapted AttentionHead /
TinyScorer / the byte-collation shape from jevlike commit 94f5fd1 (MIT,
Copyright 2026 Minimal Labs). Kept here as a single dependency-free module so
openjev-lm carries its own form specialist: train it with
`openjev.train_forms` (synth data recipe also from cua-s1), score with
`score_options(context, options)`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT_NAME = "cua-s1"
FORMAT_VERSION = 1

TensorBatch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ChoiceExample:
    """A context, its available options, and the selected option index."""

    context: str
    options: tuple[str, ...]
    label: int


def validate_example(payload: Mapping[str, Any]) -> ChoiceExample:
    """Validate a JSON-compatible choice example."""
    context = payload.get("context")
    options = payload.get("options")
    label = payload.get("label")
    if not isinstance(context, str):
        raise ValueError("choice context must be a string")
    if not isinstance(options, (list, tuple)) or len(options) < 2:
        raise ValueError("choice options must contain at least two strings")
    if any(not isinstance(option, str) or not option for option in options):
        raise ValueError("choice options must be non-empty strings")
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(options):
        raise ValueError("choice label must be a valid option index")
    return ChoiceExample(context=context, options=tuple(options), label=label)


def _byte_ids(text: str, length: int) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]


class ByteCollator:
    """Collate variable-length text choices into padded UTF-8 byte tensors."""

    def __init__(self, context_tokens: int, option_tokens: int) -> None:
        if context_tokens <= 0 or option_tokens <= 0:
            raise ValueError("token limits must be positive")
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: Sequence[ChoiceExample]) -> TensorBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        contexts = [_byte_ids(item.context, self.context_tokens) for item in examples]
        option_rows = [
            [_byte_ids(option, self.option_tokens) for option in item.options] for item in examples
        ]
        batch = len(examples)
        max_context = max(1, max(map(len, contexts)))
        max_options = max(len(row) for row in option_rows)
        max_option_tokens = max(1, max(len(t) for row in option_rows for t in row))
        context_ids = torch.full((batch, max_context), 0, dtype=torch.long)
        option_ids = torch.full((batch, max_options, max_option_tokens), 0, dtype=torch.long)
        option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
        for row, tokens in enumerate(contexts):
            if tokens:
                context_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        for row, options in enumerate(option_rows):
            option_mask[row, : len(options)] = True
            for column, tokens in enumerate(options):
                if tokens:
                    option_ids[row, column, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return {
            "context_ids": context_ids,
            "context_mask": context_ids.ne(0),
            "option_ids": option_ids,
            "option_token_mask": option_ids.ne(0),
            "option_mask": option_mask,
            "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
        }


class AttentionHead(nn.Module):
    """Turn context tokens and option vectors into one score per option."""

    def __init__(self, input_width: int, rank: int) -> None:
        super().__init__()
        if input_width <= 0 or rank <= 0:
            raise ValueError("input width and attention rank must be positive")
        self.context_norm = nn.LayerNorm(input_width)
        self.option_norm = nn.LayerNorm(input_width)
        self.query = nn.Linear(input_width, rank, bias=False)
        self.key = nn.Linear(input_width, rank, bias=False)
        self.value = nn.Linear(input_width, rank, bias=False)
        self.rank = rank

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
        shuffle_context: bool = False,
    ) -> torch.Tensor:
        context = self.context_norm(context.float())
        options = self.option_norm(options.float())
        if shuffle_context and context.shape[0] > 1:
            context = context.roll(1, dims=0)
            context_mask = context_mask.roll(1, dims=0)
        query = self.query(options)
        key = self.key(context)
        value = self.value(context)
        scores = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)
        scores = scores.masked_fill(~context_mask[:, None, :], torch.finfo(scores.dtype).min)
        attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), value)
        logits = (query * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class TinyScorer(nn.Module):
    """Byte embedding scorer with mean-pooled option representations."""

    def __init__(self, width: int, rank: int, context_tokens: int) -> None:
        super().__init__()
        if context_tokens <= 0:
            raise ValueError("context token limit must be positive")
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Embedding(context_tokens, width)
        self.head = AttentionHead(width, rank)

    def forward(self, batch: TensorBatch, shuffle_context: bool = False) -> torch.Tensor:
        context_ids = batch["context_ids"]
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context = self.embedding(context_ids) + self.position(positions)
        option_tokens = self.embedding(batch["option_ids"])
        weights = batch["option_token_mask"].unsqueeze(-1)
        options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
        return self.head(
            context, batch["context_mask"], options, batch["option_mask"], shuffle_context
        )


class TinyTransformerScorer(nn.Module):
    """Byte scorer with contextual encoders for contexts and options."""

    def __init__(
        self,
        width: int,
        rank: int,
        context_tokens: int,
        option_tokens: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(width, rank, context_tokens, option_tokens, layers, heads) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Embedding(max(context_tokens, option_tokens), width)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        option_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, batch_first=True, norm_first=True
        )
        self.option_encoder = nn.TransformerEncoder(option_layer, 1)
        self.head = AttentionHead(width, rank)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[-1], device=ids.device)
        return self.embedding(ids) + self.position(positions)

    def forward(self, batch: TensorBatch, shuffle_context: bool = False) -> torch.Tensor:
        context_mask = batch["context_mask"]
        safe_context_mask = context_mask.clone()
        safe_context_mask[:, 0] = True
        context = self.encoder(
            self._embed(batch["context_ids"]),
            src_key_padding_mask=~safe_context_mask,
        )
        option_ids = batch["option_ids"]
        batch_size, option_count, token_count = option_ids.shape
        flat_ids = option_ids.reshape(batch_size * option_count, token_count)
        flat_mask = batch["option_token_mask"].reshape(batch_size * option_count, token_count)
        safe_mask = flat_mask.clone()
        safe_mask[:, 0] = True
        hidden = self.option_encoder(self._embed(flat_ids), src_key_padding_mask=~safe_mask)
        weights = flat_mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        options = pooled.reshape(batch_size, option_count, -1)
        return self.head(
            context, context_mask, options, batch["option_mask"], shuffle_context
        )


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model config field {key!r} must be a positive integer")
    return value


def make_system(config: Mapping[str, Any], device: torch.device | str) -> tuple[nn.Module, ByteCollator]:
    """Construct a scorer and matching collator from validated configuration."""
    target = select_device(device) if isinstance(device, str) else device
    encoder = config.get("encoder")
    rank = _positive_int(config, "rank")
    context_tokens = _positive_int(config, "context_tokens")
    option_tokens = _positive_int(config, "option_tokens")
    if encoder == "tiny":
        model: nn.Module = TinyScorer(_positive_int(config, "width"), rank, context_tokens)
    elif encoder == "tinyx":
        model = TinyTransformerScorer(
            _positive_int(config, "width"),
            rank,
            context_tokens,
            option_tokens,
            int(config.get("layers", 2)),
            int(config.get("heads", 4)),
            float(config.get("dropout", 0.1)),
        )
    else:
        raise ValueError("model config field 'encoder' must be 'tiny' or 'tinyx'")
    return model.to(target), ByteCollator(context_tokens, option_tokens)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- checkpoint IO (safetensors + JSON, no pickle) ----------

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _state_signature(tensors: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(dict(config)))
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(_canonical_json([name, str(tensor.dtype), list(tensor.shape)]))
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def resolve_checkpoint_paths(directory_or_path: str | Path) -> tuple[Path, Path]:
    """Return the (safetensors, config.json) paths for a checkpoint location."""
    path = Path(directory_or_path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin", ".pkl", ".pickle"}:
        raise ValueError("pickle checkpoints are not supported; use safetensors + JSON")
    if suffix == ".safetensors":
        return path, path.with_suffix(".json")
    if suffix == ".json":
        return path.with_suffix(".safetensors"), path
    if suffix:
        raise ValueError("checkpoint path must be a directory, .safetensors file, or .json file")
    return path / "model.safetensors", path / "config.json"


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().contiguous()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def save_checkpoint(
    directory_or_path: str | Path,
    model: nn.Module,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save trainable state as safetensors with JSON config and metadata."""
    from safetensors.torch import save_file

    weights_path, config_path = resolve_checkpoint_paths(directory_or_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state(model)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "config": dict(config),
        "signature": _state_signature(state, config),
    }
    if metadata:
        payload["training"] = dict(metadata)
    fd, tmp = tempfile.mkstemp(prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent)
    os.close(fd)
    Path(tmp).write_bytes(_canonical_json(payload) + b"\n")
    os.replace(tmp, config_path)
    save_file(state, str(weights_path))
    return weights_path, config_path


def load_checkpoint(
    directory_or_path: str | Path, device: torch.device | str
) -> tuple[nn.Module, ByteCollator, dict[str, Any]]:
    """Load a model from safetensors + JSON (no serialized code)."""
    from safetensors.torch import load_file

    weights_path, config_path = resolve_checkpoint_paths(directory_or_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config missing")
    state_dict = load_file(str(weights_path))
    signature = payload.get("signature")
    if signature and signature != _state_signature(state_dict, config):
        raise ValueError("checkpoint signature mismatch")
    model, collator = make_system(config, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, collator, config


@torch.inference_mode()
def score_options(
    model: nn.Module,
    collator: ByteCollator,
    context: str,
    options: Sequence[str],
) -> list[tuple[str, float]]:
    """Score each option for a context -> [(option, prob)] sorted desc."""
    batch = collator([ChoiceExample(context=context, options=tuple(options), label=0)])
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    probs = model(batch)[0].softmax(-1).tolist()
    order = sorted(range(len(options)), key=lambda i: -probs[i])
    return [(options[i], probs[i]) for i in order]
