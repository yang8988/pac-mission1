"""Stage 4 training: behaviour cloning from the greedy teacher, then masked PPO.

    python -m palletizing_core.rl.train --out models/ppo.pt --workers 4 --updates 150

1. Behaviour cloning (BC): imitate the greedy IRAP choice (argmax Q_now) and regress the
   value head on Monte-Carlo returns, so PPO starts from a policy as good as the baseline.
2. PPO with action masks and GAE, on environments stepped in worker processes.
3. Periodic deterministic evaluation on fixed episodes against the greedy baseline; the best
   checkpoint by evaluation utilisation is kept.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..config import Config
from ..simulate import run_episode
from ..planner import GreedyPlanner
from .env import PalletEnv, TaskDistribution, greedy_slot
from .obs import obs_spec
from .policy import ActorCritic, to_tensors

KEYS = ("hm", "glob", "inv", "cands", "mask")


# ---------------------------------------------------------------------- vectorised envs
def _worker(remote, seeds: List[int]) -> None:
    torch.set_num_threads(1)
    envs = [PalletEnv(seed=s) for s in seeds]
    obs = [_reset(e) for e in envs]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "obs":
                remote.send(obs)
            elif cmd == "greedy":
                remote.send([greedy_slot(e.slots) for e in envs])
            elif cmd == "step":
                out = []
                for k, (e, a) in enumerate(zip(envs, data)):
                    o, r, done, _, info = e.step(int(a))
                    if done:
                        o = _reset(e)
                    obs[k] = o
                    out.append((o, r, done, info))
                remote.send(out)
            elif cmd == "close":
                break
    finally:
        remote.close()


def _reset(env: PalletEnv):
    while True:
        o, info = env.reset()
        if not info["done"]:
            return o


class VecEnv:
    def __init__(self, n_envs: int, n_workers: int, seed: int) -> None:
        ctx = mp.get_context("fork")
        per = np.array_split(np.arange(n_envs), n_workers)
        self.remotes, self.procs = [], []
        for idx in per:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, [seed * 1000 + int(i) for i in idx]), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.sizes = [len(i) for i in per]

    def _call(self, cmd, data=None):
        if data is None:
            for r in self.remotes:
                r.send((cmd, None))
        else:
            k = 0
            for r, n in zip(self.remotes, self.sizes):
                r.send((cmd, data[k : k + n]))
                k += n
        return [x for r in self.remotes for x in r.recv()]

    def obs(self):
        return self._call("obs")

    def greedy(self):
        return self._call("greedy")

    def step(self, actions):
        return self._call("step", list(actions))

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


# ---------------------------------------------------------------------- evaluation
def eval_specs(cfg: Config, n: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    tasks = TaskDistribution(orders=("random", "small_first", "clustered"), order_probs=(0.5, 0.25, 0.25))
    return [tasks.sample(rng, cfg) for _ in range(n)]


def evaluate(model: Optional[ActorCritic], specs, cfg: Config) -> float:
    """Mean utilisation with the deterministic policy (argmax), or the greedy teacher if model is None."""
    if model is None:
        return float(np.mean([run_episode(cfg, s.types, s.counts, s.sequence, GreedyPlanner()).state.utilization() for s in specs]))
    env = PalletEnv(cfg, shuffle_slots=False)
    utils = []
    model.eval()
    with torch.no_grad():
        for s in specs:
            o, info = env.reset(s)
            done = info["done"]
            while not done:
                logits, _ = model(to_tensors([o]))
                o, _, done, _, _ = env.step(int(logits.argmax(-1)))
            utils.append(env.state.utilization())
    model.train()
    return float(np.mean(utils))


# ---------------------------------------------------------------------- training phases
def behaviour_cloning(model, opt, venv: VecEnv, steps: int, epochs: int, batch: int, gamma: float, log) -> None:
    """Roll out the greedy teacher, then fit policy (cross-entropy) and value (MC return)."""
    n = len(venv.sizes) and sum(venv.sizes)
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    obs = venv.obs()
    for _ in range(steps // n):
        acts = venv.greedy()
        res = venv.step(acts)
        obs_buf.append(obs)
        act_buf.append(acts)
        rew_buf.append([r for _, r, _, _ in res])
        done_buf.append([d for _, _, d, _ in res])
        obs = [o for o, _, _, _ in res]
    T = len(obs_buf)
    rew, done = np.array(rew_buf, dtype=np.float32), np.array(done_buf)
    ret = np.zeros_like(rew)
    running = np.zeros(n, dtype=np.float32)  # truncated tails start at 0 (slightly biased low)
    for t in reversed(range(T)):
        running = rew[t] + gamma * running * (~done[t])
        ret[t] = running
    flat_obs = [o for row in obs_buf for o in row]
    data = to_tensors(flat_obs)
    acts = torch.as_tensor(np.array(act_buf).reshape(-1))
    rets = torch.as_tensor(ret.reshape(-1))
    N = len(flat_obs)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot, acc = 0.0, 0.0
        for k in range(0, N, batch):
            idx = perm[k : k + batch]
            logits, v = model({key: data[key][idx] for key in KEYS})
            loss_pi = nn.functional.cross_entropy(logits, acts[idx])
            loss_v = nn.functional.mse_loss(v, rets[idx])
            loss = loss_pi + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(idx)
            acc += (logits.argmax(-1) == acts[idx]).float().sum().item()
        log(f"[bc] epoch {ep + 1}/{epochs} loss {tot / N:.4f} imitation acc {acc / N:.3f} ({N} samples)")


def ppo(model, opt, venv: VecEnv, args, specs, cfg, out: Path, log) -> None:
    n = sum(venv.sizes)
    obs = venv.obs()
    best = -1.0
    ep_utils: List[float] = []
    rows = []
    t_start = time.time()
    for upd in range(1, args.updates + 1):
        frac = 1.0 - (upd - 1) / args.updates
        for gparam in opt.param_groups:
            gparam["lr"] = args.lr * frac
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        model.eval()
        for _ in range(args.n_steps):
            with torch.no_grad():
                logits, v = model(to_tensors(obs))
                dist = Categorical(logits=logits)
                a = dist.sample()
            res = venv.step(a.tolist())
            buf["obs"].append(obs)
            buf["act"].append(a)
            buf["logp"].append(dist.log_prob(a))
            buf["val"].append(v)
            buf["rew"].append(torch.tensor([r for _, r, _, _ in res], dtype=torch.float32))
            buf["done"].append(torch.tensor([d for _, _, d, _ in res], dtype=torch.float32))
            ep_utils += [info["utilization"] for _, _, d, info in res if d]
            obs = [o for o, _, _, _ in res]
        with torch.no_grad():
            _, last_v = model(to_tensors(obs))
        model.train()

        # GAE
        T = args.n_steps
        vals = torch.stack(buf["val"])
        rews, dones = torch.stack(buf["rew"]), torch.stack(buf["done"])
        adv = torch.zeros(T, n)
        gae = torch.zeros(n)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else vals[t + 1]
            delta = rews[t] + args.gamma * nxt * (1 - dones[t]) - vals[t]
            gae = delta + args.gamma * args.lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals

        flat_obs = [o for row in buf["obs"] for o in row]
        data = to_tensors(flat_obs)
        act = torch.stack(buf["act"]).reshape(-1)
        old_logp = torch.stack(buf["logp"]).reshape(-1)
        adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
        N = len(flat_obs)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(N)
            for k in range(0, N, args.batch):
                idx = perm[k : k + args.batch]
                logits, v = model({key: data[key][idx] for key in KEYS})
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                a_n = adv_f[idx]
                a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                ratio = (logp - old_logp[idx]).exp()
                l_pi = -torch.min(ratio * a_n, ratio.clamp(1 - args.clip, 1 + args.clip) * a_n).mean()
                l_v = nn.functional.mse_loss(v, ret_f[idx])
                ent = dist.entropy().mean()
                loss = l_pi + args.vf_coef * l_v - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = (old_logp[idx] - logp).mean().item()
                stats.append((l_pi.item(), l_v.item(), ent.item(), kl, ((ratio - 1).abs() > args.clip).float().mean().item()))
        s = np.mean(stats, axis=0)
        train_util = float(np.mean(ep_utils[-50:])) if ep_utils else float("nan")
        row = {"update": upd, "steps": upd * T * n, "train_util": train_util, "pi_loss": s[0], "v_loss": s[1],
               "entropy": s[2], "kl": s[3], "clip_frac": s[4], "eval_util": "", "minutes": (time.time() - t_start) / 60}  # fmt: skip
        if upd % args.eval_every == 0 or upd == args.updates:
            util = evaluate(model, specs, cfg)
            row["eval_util"] = util
            if util > best:
                best = util
                save(model, out, args, extra={"eval_util": util, "update": upd})
            log(
                f"[ppo] upd {upd}/{args.updates} train_util {train_util:.3f} eval_util {util:.4f} (best {best:.4f}) "
                f"ent {s[2]:.3f} kl {s[3]:.4f} clip {s[4]:.3f} vloss {s[1]:.3f}"
            )
        rows.append(row)
        with open(out.with_suffix(".csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


def save(model: ActorCritic, path: Path, args, extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hm_shape": model.hm_shape, "args": vars(args), **(extra or {})}, path)


def load_model(path: str | Path) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(ckpt["hm_shape"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/ppo.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4, help="torch threads in the learner")
    ap.add_argument("--bc-steps", type=int, default=12000, help="teacher transitions for behaviour cloning (0 = skip)")
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=64, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=24)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(out.with_suffix(".log"), "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    cfg = Config()
    specs = eval_specs(cfg, args.eval_episodes)
    log(f"greedy teacher eval_util {evaluate(None, specs, cfg):.4f} on {len(specs)} fixed episodes")

    probe = PalletEnv(cfg)
    probe.reset()
    model = ActorCritic(obs_spec(probe.state).hm_shape)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    venv = VecEnv(args.n_envs, args.workers, args.seed)
    try:
        if args.bc_steps > 0:
            behaviour_cloning(model, opt, venv, args.bc_steps, args.bc_epochs, args.batch, args.gamma, log)
            util = evaluate(model, specs, cfg)
            save(model, out.with_name(out.stem + "_bc.pt"), args, extra={"eval_util": util, "update": 0})
            log(f"[bc] eval_util {util:.4f}")
        ppo(model, opt, venv, args, specs, cfg, out, log)
    finally:
        venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
