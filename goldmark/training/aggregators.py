"""MIL aggregation models."""

from __future__ import annotations

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear_weights(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class LegacyGMAGatedAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: bool = True, n_tasks: int = 1) -> None:
        super().__init__()
        seq_a = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        seq_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            seq_a.append(nn.Dropout(0.25))
            seq_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*seq_a)
        self.attention_b = nn.Sequential(*seq_b)
        self.attention_c = nn.Linear(hidden_dim, n_tasks)
        _init_linear_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_a(x)
        b = self.attention_b(x)
        attn = self.attention_c(a * b)
        return attn, x


class GMA(nn.Module):
    """Legacy GMA (gated attention MIL) used in MIL_CODE."""

    def __init__(self, ndim: int, dropout: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        hidden_dim = 512
        attn_dim = 384
        self.fc1 = nn.Linear(ndim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.25) if dropout else nn.Identity()
        self.attention_net = LegacyGMAGatedAttention(hidden_dim, attn_dim, dropout=dropout, n_tasks=1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        _init_linear_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, tiles, dim)
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        attn_logits, features = self.attention_net(h)
        attn_logits = attn_logits.transpose(1, 2)  # (batch, tasks=1, tiles)
        weights = F.softmax(attn_logits, dim=2)
        pooled = torch.sum(weights.transpose(1, 2) * features, dim=1)
        logits = self.classifier(pooled)
        return weights.squeeze(1), pooled, logits


class ABMIL(nn.Module):
    """Attention-based MIL."""

    def __init__(self, ndim: int, num_classes: int = 2, dropout: bool = True) -> None:
        super().__init__()
        self.proj_v = nn.Linear(ndim, ndim // 2)
        self.proj_u = nn.Linear(ndim // 2, 1)
        layers = [nn.Linear(ndim, ndim // 2), nn.ReLU(), nn.Linear(ndim // 2, num_classes)]
        if dropout:
            layers = [nn.Dropout(0.5)] + layers[:2] + [nn.Dropout(0.5), layers[-1]]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.proj_v(x))
        attn = self.proj_u(v)
        weights = F.softmax(attn, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        logits = self.classifier(pooled)
        return weights.squeeze(-1), pooled, logits


class DSMIL(nn.Module):
    """Dual-stream MIL."""

    def __init__(self, ndim: int, num_classes: int = 2, dropout: bool = True) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(ndim, ndim // 2),
            nn.ReLU(),
            nn.Linear(ndim // 2, 1),
        )
        self.transform = nn.Sequential(nn.Linear(ndim, ndim), nn.ReLU(), nn.Linear(ndim, ndim))
        layers = [nn.Linear(ndim * 2, ndim), nn.ReLU(), nn.Linear(ndim, num_classes)]
        if dropout:
            layers = [nn.Dropout(0.5)] + layers[:2] + [nn.Dropout(0.5), layers[-1]]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_features = torch.max(x, dim=1)[0]
        transformed = self.transform(max_features)
        attn = self.attention(x)
        weights = F.softmax(attn, dim=1)
        attended = torch.sum(x * weights, dim=1)
        combined = torch.cat([transformed, attended], dim=1)
        logits = self.classifier(combined)
        return weights.squeeze(-1), combined, logits


class PPEG(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        cls = x[:, :1, :]
        tokens = x[:, 1:, :]
        b, n, c = tokens.shape
        feat_map = tokens.transpose(1, 2).reshape(b, c, h, w)
        feat_map = feat_map + self.conv7(feat_map) + self.conv5(feat_map) + self.conv3(feat_map)
        tokens = feat_map.view(b, c, -1).transpose(1, 2)
        return torch.cat([cls, tokens], dim=1)


class TransMIL(nn.Module):
    """Transformer-based MIL aggregator."""

    def __init__(self, ndim: int, num_classes: int = 2, embed_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(ndim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_layer = PPEG(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.layer1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer2 = nn.TransformerEncoder(copy.deepcopy(encoder_layer), num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        device = x.device
        h = self.embed(x)
        side = int(math.ceil(math.sqrt(n)))
        total = side * side
        if n < total:
            pad = h[:, : total - n, :]
            h = torch.cat([h, pad], dim=1)
        cls = self.cls_token.expand(b, -1, -1).to(device)
        h = torch.cat([cls, h], dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        cls_feat = self.norm(h)[:, 0]
        logits = self.head(cls_feat)
        weights = torch.ones(b, n, device=device) / float(n)
        return weights, cls_feat, logits


def create_aggregator(name: str, feature_dim: int, num_classes: int = 2, dropout: bool = True) -> nn.Module:
    name = name.lower()
    if name == "gma":
        return GMA(feature_dim, dropout=dropout, num_classes=num_classes)
    if name == "abmil":
        return ABMIL(feature_dim, num_classes=num_classes, dropout=dropout)
    if name == "dsmil":
        return DSMIL(feature_dim, num_classes=num_classes, dropout=dropout)
    if name == "transmil":
        return TransMIL(feature_dim, num_classes=num_classes, dropout=0.1 if dropout else 0.0)
    raise ValueError(f"Unsupported aggregator '{name}'")
