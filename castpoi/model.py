"""CaST-POI: candidate-conditioned spatiotemporal ranker for next-POI recommendation."""
import math
import torch
import torch.nn as nn

from .layers import InputEncoder, RepeatGate, repeat_features

TIME_BOUNDS_H = [0.0, 1.0, 6.0, 24.0, 168.0, 720.0, 1e9]
DIST_BOUNDS_KM = [0.0, 0.5, 2.0, 5.0, 20.0, 100.0, 1e9]


def _bucketize(x, bounds):
    out = torch.zeros_like(x, dtype=torch.long)
    for i in range(len(bounds) - 1):
        out = torch.where((x >= bounds[i]) & (x < bounds[i + 1]), torch.full_like(out, i), out)
    return out


def _haversine_km(a, b):
    R = 6371.0
    lat1, lon1 = torch.deg2rad(a[..., 0]), torch.deg2rad(a[..., 1])
    lat2, lon2 = torch.deg2rad(b[..., 0]), torch.deg2rad(b[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.sqrt(torch.clamp(h, 0, 1)))


class _PointWiseFFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class CaSTPOI(nn.Module):
    def __init__(self, num_pois, poi_locations, config):
        super().__init__()

        d = config["poi_embed_dim"]
        self.d = d
        self.num_pois = num_pois
        self.K = 1
        self.H = config.get("castpoi_heads", 4)
        self.dh = d // self.H
        self.n_backbone = config.get("castpoi_backbone_layers", 2)
        self.n_cross = config.get("castpoi_cross_layers", 2)
        self.cand_chunk = config.get("castpoi_cand_chunk", 1024)
        max_len = config["max_history_len"]

        self.use_backbone = config.get("use_backbone", True)
        self.candidate_conditioned = config.get("castpoi_candidate_conditioned", True)
        self.use_temporal_bias = config.get("castpoi_temporal_bias", True)
        self.use_spatial_bias = config.get("castpoi_spatial_bias", True)
        self.use_repeat = config.get("use_repeat", True)

        self.input_encoder = InputEncoder(
            num_pois, d, config["slot_embed_dim"], config["spatial_dim"],
            config["num_dist_buckets"], config["dist_embed_dim"], config["dropout"])
        self.poi_embedding = self.input_encoder.poi_embedding
        self.in_proj = nn.Linear(self.input_encoder.output_dim, d)
        self.pos_embedding = nn.Embedding(max_len, d)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        self.emb_drop = nn.Dropout(config["dropout"])

        # causal self-attention backbone
        self.bb_ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.n_backbone))
        self.bb_q = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_backbone))
        self.bb_k = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_backbone))
        self.bb_v = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_backbone))
        self.bb_o = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_backbone))
        self.bb_ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.n_backbone))
        self.bb_ffn = nn.ModuleList(_PointWiseFFN(d, config["dropout"]) for _ in range(self.n_backbone))
        self.last_ln = nn.LayerNorm(d)

        # candidate-conditioned cross-attention
        self.cq = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_cross))
        self.ck = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_cross))
        self.cv = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_cross))
        self.co = nn.ModuleList(nn.Linear(d, d) for _ in range(self.n_cross))
        self.cln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.n_cross))
        self.cln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.n_cross))
        self.cffn = nn.ModuleList(_PointWiseFFN(d, config["dropout"]) for _ in range(self.n_cross))
        self.t_bias = nn.Embedding(len(TIME_BOUNDS_H) - 1, 1)
        self.s_bias = nn.Embedding(len(DIST_BOUNDS_KM) - 1, 1)
        nn.init.zeros_(self.t_bias.weight)
        nn.init.zeros_(self.s_bias.weight)
        self.cand_head = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(config["dropout"]), nn.Linear(d, 1))
        self.cand_gate = nn.Parameter(torch.tensor(0.0))

        # repeat gate
        if self.use_repeat:
            self.repeat_gate = RepeatGate(d, config.get("repeat_gate_hidden", 32))
            self.register_buffer("_pad_onehot", torch.zeros(num_pois), persistent=False)
            self._pad_onehot[0] = 1.0

        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.register_buffer("poi_locations", torch.as_tensor(poi_locations, dtype=torch.float32))
        pad = torch.zeros(num_pois); pad[0] = -1e9
        self.register_buffer("_pad_mask", pad, persistent=False)

    def _split(self, t):
        B, N, _ = t.shape
        return t.view(B, N, self.H, self.dh)

    def forward(self, poi_ids, ts_hours, hour, dow, locations, seq_lengths,
                query_hour=None, query_dow=None, query_location=None,
                prev_query_location=None, repeat_hist=None):
        B, L = poi_ids.shape
        pad = (poi_ids == 0)
        prev = torch.empty_like(locations)
        prev[:, 1:] = locations[:, :-1]; prev[:, 0] = locations[:, 0]
        x = self.in_proj(self.input_encoder(poi_ids, hour, dow, locations, prev))
        pos = torch.arange(L, device=poi_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb_drop(x + self.pos_embedding(pos)).masked_fill(pad.unsqueeze(-1), 0.0)

        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=poi_ids.device), 1)
        block = causal.view(1, 1, L, L) | pad.view(B, 1, 1, L)
        h = x
        if self.use_backbone:
            for l in range(self.n_backbone):
                q = self._split(self.bb_q[l](self.bb_ln1[l](h))).transpose(1, 2)
                k = self._split(self.bb_k[l](h)).transpose(1, 2)
                v = self._split(self.bb_v[l](h)).transpose(1, 2)
                logit = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
                logit = logit.masked_fill(block, -1e9)
                a = torch.softmax(logit, dim=-1) @ v
                h = h + self.bb_o[l](a.transpose(1, 2).reshape(B, L, self.d))
                h = self.bb_ffn[l](self.bb_ln2[l](h)).masked_fill(pad.unsqueeze(-1), 0.0)
            h = self.last_ln(h)
        h_seq = h[:, -1, :]

        Ks = [self._split(self.ck[l](h)).transpose(1, 2) for l in range(self.n_cross)]
        Vs = [self._split(self.cv[l](h)).transpose(1, 2) for l in range(self.n_cross)]

        time_bias = None
        if self.use_temporal_bias:
            ref = ts_hours.gather(1, (L - 1) * torch.ones(B, 1, dtype=torch.long, device=poi_ids.device))
            dt = (ref - ts_hours).clamp(min=0)
            time_bias = self.t_bias(_bucketize(dt, TIME_BOUNDS_H)).squeeze(-1)

        rep = None
        if self.use_repeat:
            src = repeat_hist if repeat_hist is not None else poi_ids
            cnt, rec, vis = repeat_features(src, self.num_pois)
            w = self.repeat_gate(h_seq)
            rep = w[:, 0:1] * cnt + w[:, 1:2] * rec + w[:, 2:3] * vis
            rep = rep * (1.0 - self._pad_onehot)

        cache = {"h_seq": h_seq, "Ks": Ks, "Vs": Vs, "time_bias": time_bias,
                 "pad": pad, "hist_locs": locations, "B": B, "L": L}
        return cache, None, rep

    def _cand_cond_score(self, cache, cand_ids, cand_locs):
        B, C = cand_ids.shape
        L = cache["L"]
        e_cand = self.poi_embedding(cand_ids)
        kpm = cache["pad"].view(B, 1, 1, L)
        bias = None
        if cache["time_bias"] is not None:
            bias = cache["time_bias"].view(B, 1, 1, L)
        if self.use_spatial_bias:
            hh = cache["hist_locs"].unsqueeze(1); cc = cand_locs.unsqueeze(2)
            sb = self.s_bias(_bucketize(_haversine_km(hh, cc), DIST_BOUNDS_KM)).squeeze(-1).unsqueeze(1)
            bias = sb if bias is None else bias + sb
        ur = e_cand
        for l in range(self.n_cross):
            q = self._split(self.cq[l](ur)).transpose(1, 2)
            logit = (q @ cache["Ks"][l].transpose(-1, -2)) / math.sqrt(self.dh)
            if bias is not None:
                logit = logit + bias
            logit = logit.masked_fill(kpm, -1e9)
            attn = torch.softmax(logit, dim=-1)
            out = (attn @ cache["Vs"][l]).transpose(1, 2).reshape(B, C, self.d)
            ur = self.cln1[l](ur + self.co[l](out))
            ur = self.cln2[l](self.cffn[l](ur))
        feat = torch.cat([ur, e_cand, ur * e_cand], dim=-1)
        return self.cand_head(feat).squeeze(-1)

    def _score(self, cache, cand_ids, cand_locs):
        e_cand = self.poi_embedding(cand_ids)
        s = (cache["h_seq"].unsqueeze(1) * e_cand).sum(-1)
        s = s + self.poi_bias[cand_ids]
        if self.candidate_conditioned:
            s = s + self.cand_gate * self._cand_cond_score(cache, cand_ids, cand_locs)
        return s

    def compute_sampled_scores(self, cache, pos_ids, neg_ids, rep=None):
        cand = torch.cat([pos_ids.unsqueeze(1), neg_ids], dim=1)
        s = self._score(cache, cand, self.poi_locations[cand])
        if rep is not None:
            s = s + rep.gather(1, cand)
        return s[:, 0], s[:, 1:]

    def compute_all_scores(self, cache, rep=None):
        B = cache["B"]
        device = cache["h_seq"].device
        scores = torch.full((B, self.num_pois), -1e9, device=device)
        for lo in range(1, self.num_pois, self.cand_chunk):
            hi = min(lo + self.cand_chunk, self.num_pois)
            ids = torch.arange(lo, hi, device=device)
            cand = ids.unsqueeze(0).expand(B, hi - lo)
            cand_locs = self.poi_locations[lo:hi].unsqueeze(0).expand(B, hi - lo, 2)
            scores[:, lo:hi] = self._score(cache, cand, cand_locs)
        if rep is not None:
            scores = scores + rep + self._pad_mask
        return scores
