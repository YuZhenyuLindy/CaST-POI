"""Shared encoders and losses used by the CaST-POI ranker."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_r, lon1_r = torch.deg2rad(lat1), torch.deg2rad(lon1)
    lat2_r, lon2_r = torch.deg2rad(lat2), torch.deg2rad(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))


class TemporalEncoding(nn.Module):
    """Periodic time features from local hour-of-day and day-of-week plus a slot embedding."""

    def __init__(self, slot_embed_dim: int = 16, num_slots: int = 4, dropout: float = 0.1):
        super().__init__()
        self.omega_h = 2 * math.pi / 24.0
        self.omega_w = 2 * math.pi / 7.0
        self.slot_embedding = nn.Embedding(num_slots, slot_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 4 + slot_embed_dim
        nn.init.normal_(self.slot_embedding.weight, std=0.02)

    @staticmethod
    def time_slot(hour: torch.Tensor) -> torch.Tensor:
        slot = torch.full(hour.shape, 3, dtype=torch.long, device=hour.device)
        slot[(hour >= 6) & (hour < 12)] = 0
        slot[(hour >= 12) & (hour < 18)] = 1
        slot[hour >= 18] = 2
        return slot

    def forward(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        h, w = hour.float(), dow.float()
        feats = torch.stack([
            torch.sin(self.omega_h * h), torch.cos(self.omega_h * h),
            torch.sin(self.omega_w * w), torch.cos(self.omega_w * w),
        ], dim=-1)
        return self.dropout(torch.cat([feats, self.slot_embedding(self.time_slot(h))], dim=-1))


class SpatialEncoding(nn.Module):
    """Displacement features from the previous POI: log distance, a bucket embedding, and dlat/dlon."""

    DIST_BUCKETS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, float("inf")]

    def __init__(self, output_dim: int = 32, num_dist_buckets: int = 8,
                 dist_embed_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_dist_buckets = num_dist_buckets
        self.dist_embedding = nn.Embedding(num_dist_buckets, dist_embed_dim)
        self.register_buffer("_edges", torch.tensor(self.DIST_BUCKETS[1:-1]), persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(1 + dist_embed_dim + 2, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim), nn.Dropout(dropout),
        )
        self.output_dim = output_dim
        nn.init.normal_(self.dist_embedding.weight, std=0.02)

    def _bucket(self, d: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(d, self._edges).clamp_(max=self.num_dist_buckets - 1)

    def forward(self, locations: torch.Tensor, prev_locations: torch.Tensor) -> torch.Tensor:
        lat1, lon1 = prev_locations[..., 0], prev_locations[..., 1]
        lat2, lon2 = locations[..., 0], locations[..., 1]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        feats = torch.cat([
            torch.log1p(dist).unsqueeze(-1),
            self.dist_embedding(self._bucket(dist)),
            ((lat2 - lat1) / 0.1).unsqueeze(-1),
            ((lon2 - lon1) / 0.1).unsqueeze(-1),
        ], dim=-1)
        return self.projection(feats)


class InputEncoder(nn.Module):
    """POI embedding concatenated with temporal and spatial encodings."""

    def __init__(self, num_pois, poi_embed_dim, slot_embed_dim, spatial_dim,
                 num_dist_buckets, dist_embed_dim, dropout):
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois, poi_embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        with torch.no_grad():
            self.poi_embedding.weight[0].zero_()
        self.temporal_enc = TemporalEncoding(slot_embed_dim, dropout=dropout)
        self.spatial_enc = SpatialEncoding(spatial_dim, num_dist_buckets, dist_embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = poi_embed_dim + self.temporal_enc.output_dim + spatial_dim

    def forward(self, poi_ids, hour, dow, locations, prev_locations):
        return self.dropout(torch.cat([
            self.poi_embedding(poi_ids),
            self.temporal_enc(hour, dow),
            self.spatial_enc(locations, prev_locations),
        ], dim=-1))


def repeat_features(poi_ids: torch.Tensor, num_pois: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-candidate revisit signals from the visible history: log visit count, recency, visited flag.

    History is left-padded with 0; index 0 is the padding slot and is zeroed out.
    """
    B, L = poi_ids.shape
    valid = (poi_ids != 0).to(torch.float32)

    counts = torch.zeros(B, num_pois, device=poi_ids.device, dtype=torch.float32)
    counts.scatter_add_(1, poi_ids, valid)
    counts[:, 0] = 0.0

    pos = torch.arange(1, L + 1, device=poi_ids.device, dtype=torch.float32) / L
    pos = pos.unsqueeze(0).expand(B, L) * valid
    recency = torch.zeros(B, num_pois, device=poi_ids.device, dtype=torch.float32)
    recency.scatter_reduce_(1, poi_ids, pos, reduce="amax", include_self=True)
    recency[:, 0] = 0.0

    return torch.log1p(counts), recency, (counts > 0).to(torch.float32)


class RepeatGate(nn.Module):
    """Map the sequence state to three (unconstrained) weights over the revisit features."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        return self.net(x)


class CELoss(nn.Module):
    """Cross-entropy with label smoothing and a hard-negative BPR margin.

    forward() scores a sampled candidate set; forward_full() scores the full vocabulary.
    """

    def __init__(self, label_smoothing=0.02, explore_weight=1.5, bpr_weight=0.5, margin=1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.explore_weight = explore_weight
        self.bpr_weight = bpr_weight
        self.margin = margin

    def forward(self, pos_scores, neg_scores, is_explore: Optional[torch.Tensor] = None):
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        B, C = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            target = torch.full_like(log_probs, smooth)
            target[:, 0] = 1.0 - self.label_smoothing + smooth
        else:
            target = torch.zeros_like(log_probs)
            target[:, 0] = 1.0
        ce = -(target * log_probs).sum(1)

        k = min(10, neg_scores.size(1))
        hard_neg, _ = neg_scores.topk(k, dim=1)
        bpr = F.relu(self.margin - (pos_scores.unsqueeze(1) - hard_neg)).mean(1)

        loss = ce + self.bpr_weight * bpr
        if is_explore is not None and self.explore_weight > 1.0:
            loss = loss * (1.0 + (self.explore_weight - 1.0) * is_explore)
        return loss.mean()

    def forward_full(self, logits, target_ids, is_explore: Optional[torch.Tensor] = None):
        B, V = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        tgt_col = target_ids.unsqueeze(1)

        target = torch.zeros_like(log_probs)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (V - 1)
            target[:, 1:] = smooth
            target.scatter_(1, tgt_col, 1.0 - self.label_smoothing + smooth)
        else:
            target.scatter_(1, tgt_col, 1.0)
        ce = -(target * log_probs).sum(1)

        masked = logits.scatter(1, tgt_col, float("-inf"))
        masked[:, 0] = float("-inf")
        k = min(10, V - 2)
        hard_neg, _ = masked.topk(k, dim=1)
        pos_scores = logits.gather(1, tgt_col).squeeze(1)
        bpr = F.relu(self.margin - (pos_scores.unsqueeze(1) - hard_neg)).mean(1)

        loss = ce + self.bpr_weight * bpr
        if is_explore is not None and self.explore_weight > 1.0:
            loss = loss * (1.0 + (self.explore_weight - 1.0) * is_explore)
        return loss.mean()


