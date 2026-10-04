"""Stage 5: Expert Iteration (section 6.3).

    python -m palletizing_core.rl.expert --init models/ppo.pt --iters 3 --episodes 64 --out-dir models/ei

Each iteration:
  1. Expert = lookahead planner (inventory scenarios, Successive Halving) whose K0 candidates
     are chosen with the current policy prior (and optionally rollouts end with V_theta).
  2. The expert plays training episodes; every decision becomes a sample
     (observation, slot the expert chose, Monte-Carlo return-to-go of the expert's play).
  3. The student (policy + value network) is fine-tuned on all samples so far (DAgger-style
     aggregation): cross-entropy on the expert's choice + MSE on the return.
  4. The new student becomes the prior (and value) of the next expert.

The same episode seeds are used in every iteration, so the expert's utilisation on its own
data is directly comparable across iterations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from ..benchmark import build_config
from ..config import Config
from ..lookahead import LookaheadPlanner
from ..model import PalletState
from ..planner import PLACE, GreedyPlanner
from .env import REWARD_SCALE, TaskDistribution
from .obs import N_CANDS, encode
from .policy import to_tensors
from .train import KEYS, eval_specs, evaluate, load_model, save

_MODELS: Dict[str, object] = {}


def _init_worker() -> None:
    torch.set_num_threads(1)


def _model(path: Optional[str]):
    if path is None:
        return None
    if path not in _MODELS:
        _MODELS[path] = load_model(path)
    return _MODELS[path]


def play_expert_episode(task) -> dict:
    """Run one episode with the expert and return its training samples."""
    cfg, seed, model_path = task
    rng = np.random.default_rng(seed)
    spec = TaskDistribution().sample(rng, cfg)
    expert = LookaheadPlanner(model=_model(model_path))
    ranker = GreedyPlanner("irap", top_n=N_CANDS)
    state = PalletState.new(cfg, spec.types, spec.counts)
    v_pal = cfg.pallet.length * cfg.pallet.width * cfg.pallet.max_height
    obs_list, acts, steps, rewards = [], [], [], []
    for box in spec.sequence:
        state.consume(box.type_id)
        dec_q = ranker.decide(state, box)
        if dec_q.action != PLACE:
            state.reject(box)
            rewards.append(0.0)
            continue
        dec = expert.decide(state, box)
        obs, slots = encode(state, box, dec_q.ranked, rng)
        key = (dec.best.i, dec.best.j, dec.best.o)
        slot = next((k for k, c in enumerate(slots) if c is not None and (c.i, c.j, c.o) == key), None)
        if slot is not None:  # the expert's choice is among the observed candidates
            obs_list.append(obs)
            acts.append(slot)
            steps.append(len(rewards))
        state.commit(dec.best)
        rewards.append(REWARD_SCALE * box.volume / v_pal)
    togo = np.cumsum(np.asarray(rewards)[::-1])[::-1]
    return {
        "obs": obs_list,
        "act": acts,
        "ret": [float(togo[t]) for t in steps],
        "util": state.utilization(),
        "order": spec.order,
    }


def stack(samples: List[dict]) -> Dict[str, np.ndarray]:
    obs = [o for s in samples for o in s["obs"]]
    out = {k: np.stack([o[k] for o in obs]) for k in KEYS}
    out["act"] = np.array([a for s in samples for a in s["act"]], dtype=np.int64)
    out["ret"] = np.array([r for s in samples for r in s["ret"]], dtype=np.float32)
    return out


def fit(model, data: Dict[str, np.ndarray], epochs: int, batch: int, lr: float, log) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = {k: torch.as_tensor(data[k]) for k in KEYS}
    act, ret = torch.as_tensor(data["act"]), torch.as_tensor(data["ret"])
    N = len(act)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot_pi = tot_v = acc = 0.0
        for k in range(0, N, batch):
            idx = perm[k : k + batch]
            logits, v = model({key: t[key][idx] for key in KEYS})
            l_pi = nn.functional.cross_entropy(logits, act[idx])
            l_v = nn.functional.mse_loss(v, ret[idx])
            loss = l_pi + 0.5 * l_v
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tot_pi += l_pi.item() * len(idx)
            tot_v += l_v.item() * len(idx)
            acc += (logits.argmax(-1) == act[idx]).float().sum().item()
        log(f"  [fit] epoch {ep + 1}/{epochs} pi_loss {tot_pi / N:.3f} v_mse {tot_v / N:.3f} expert-match {acc / N:.3f}")
    model.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="models/ppo.pt", help="starting policy/value checkpoint")
    ap.add_argument("--out-dir", default="models/ei")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=64, help="expert episodes per iteration")
    ap.add_argument("--rollouts", type=int, default=32, help="expert rollouts per decision")
    ap.add_argument("--value-leaf", action="store_true", help="expert ends rollouts with V_theta")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=32)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "expert.log", "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.workers)
    cfg: Config = build_config(None, args.rollouts, None)
    cfg.lookahead = dataclasses.replace(cfg.lookahead, value_leaf=args.value_leaf)
    specs = eval_specs(Config(), args.eval_episodes)
    model_path = args.init
    model = load_model(model_path)
    history = []
    log(f"start: init {model_path}, student eval_util {evaluate(model, specs, Config()):.4f}")
    all_samples: List[dict] = []
    seeds = [args.seed * 10000 + k for k in range(args.episodes)]
    for it in range(args.iters):
        t0 = time.time()
        with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
            samples = pool.map(play_expert_episode, [(cfg, s, model_path) for s in seeds], chunksize=1)
        all_samples += samples
        expert_util = float(np.mean([s["util"] for s in samples]))
        n_new = sum(len(s["act"]) for s in samples)
        log(f"iter {it + 1}: expert util {expert_util:.4f} on {len(samples)} episodes, {n_new} samples "
            f"({(time.time() - t0) / 60:.1f} min)")  # fmt: skip
        data = stack(all_samples)
        fit(model, data, args.epochs, args.batch, args.lr, log)
        student_util = evaluate(model, specs, Config())
        model_path = str(out / f"ei_{it + 1}.pt")
        save(model, Path(model_path), args, extra={"eval_util": student_util, "iteration": it + 1})
        history.append({"iteration": it + 1, "expert_util": expert_util, "student_eval_util": student_util,
                        "samples_total": int(len(data["act"]))})  # fmt: skip
        log(f"iter {it + 1}: student eval_util {student_util:.4f}, saved {model_path}")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
