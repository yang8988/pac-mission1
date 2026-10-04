"""Actor-critic network over candidate sets (section 5, M5): pi_theta(a|s) and V_theta(s).

    heightmap --CNN--+
    glob, inv --MLP--+--> g --> V(s)
    candidates --MLP--> c_j --(self-attention, + g)--> logit_j --mask--> softmax

Candidate embeddings share weights, so the policy works for any number of candidates and
any slot order. Masked slots get -inf logits (probability exactly 0).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .obs import CAND_DIM, GLOB_DIM, N_TYPES, TYPE_DIM

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, hm_shape=(1, 24, 20), hidden: int = 128, cand_hidden: int = 64) -> None:
        super().__init__()
        self.hm_shape = tuple(hm_shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *self.hm_shape)).shape[1]
        self.hm_fc = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU())
        self.ctx_fc = nn.Sequential(nn.Linear(GLOB_DIM + N_TYPES * TYPE_DIM, 64), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.ReLU())
        self.cand_enc = nn.Sequential(
            nn.Linear(CAND_DIM, cand_hidden), nn.ReLU(), nn.Linear(cand_hidden, cand_hidden), nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(cand_hidden, num_heads=4, batch_first=True)
        self.score = nn.Sequential(nn.Linear(2 * cand_hidden + hidden, cand_hidden), nn.ReLU(), nn.Linear(cand_hidden, 1))
        self.value = nn.Sequential(nn.Linear(hidden + cand_hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs: Dict[str, torch.Tensor]):
        """Returns (masked logits [B, N], value [B])."""
        mask = obs["mask"]
        g_hm = self.hm_fc(self.cnn(obs["hm"]))
        g_ctx = self.ctx_fc(torch.cat([obs["glob"], obs["inv"]], dim=-1))
        g = self.glob(torch.cat([g_hm, g_ctx], dim=-1))  # [B, H]

        c = self.cand_enc(obs["cands"])  # [B, N, C]
        # Attention over the other feasible candidates (padding keys ignored). A row with no
        # valid key never occurs because the environment only asks when something is feasible.
        a, _ = self.attn(c, c, c, key_padding_mask=~mask)
        n = c.shape[1]
        logits = self.score(torch.cat([c, a, g.unsqueeze(1).expand(-1, n, -1)], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, NEG_INF)

        m = mask.unsqueeze(-1).float()
        pooled = (c * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.value(torch.cat([g, pooled], dim=-1)).squeeze(-1)
        return logits, value


def to_tensors(obs_list, device="cpu") -> Dict[str, torch.Tensor]:
    """Stack a list of numpy observation dicts into a batch of tensors."""
    import numpy as np

    out = {}
    for k in ("hm", "glob", "inv", "cands", "mask"):
        arr = np.stack([o[k] for o in obs_list])
        out[k] = torch.as_tensor(arr, device=device)
    return out
