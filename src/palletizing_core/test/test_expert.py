import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from palletizing_core.config import Config  # noqa: E402
from palletizing_core.lookahead import LookaheadPlanner  # noqa: E402
from palletizing_core.rl.env import PalletEnv, TaskDistribution  # noqa: E402
from palletizing_core.rl.expert import play_expert_episode, stack  # noqa: E402
from palletizing_core.rl.obs import obs_spec  # noqa: E402
from palletizing_core.rl.policy import ActorCritic  # noqa: E402
from palletizing_core.simulate import run_episode, validate  # noqa: E402


def small_cfg(**la):
    cfg = Config()
    params = dict(k0=4, max_scenarios=4, first_round=1, depth=3, time_budget=None, max_rollouts=6)
    params.update(la)
    cfg.lookahead = dataclasses.replace(cfg.lookahead, **params)
    return cfg


def tiny_model():
    torch.manual_seed(0)
    env = PalletEnv(seed=0)
    env.reset()
    return ActorCritic(obs_spec(env.state).hm_shape).eval()


def small_spec(cfg, seed=0):
    tasks = TaskDistribution(n_types=(4, 4), fill=(0.5, 0.5), orders=("random",), order_probs=(1.0,))
    return tasks.sample(np.random.default_rng(seed), cfg)


@pytest.mark.parametrize("leaf", [False, True])
def test_model_guided_lookahead_is_safe_and_deterministic(leaf):
    cfg = small_cfg(value_leaf=leaf)
    s = small_spec(cfg)
    model = tiny_model()
    a = run_episode(cfg, s.types, s.counts, s.sequence, LookaheadPlanner(model=model))
    b = run_episode(cfg, s.types, s.counts, s.sequence, LookaheadPlanner(model=model))
    assert validate(a.state) == []
    assert [(p.x, p.y, p.z) for p in a.state.placed] == [(p.x, p.y, p.z) for p in b.state.placed]
    assert any("value" in d.info for d in a.decisions)
    assert all(d.info.get("rollouts", 0) <= cfg.lookahead.max_rollouts for d in a.decisions)


def test_expert_samples_are_consistent(tmp_path):
    from palletizing_core.rl.train import save

    cfg = small_cfg()
    path = tmp_path / "m.pt"

    class Args:
        pass

    save(tiny_model(), path, Args())
    out = play_expert_episode((cfg, 3, str(path)))
    assert out["act"] and len(out["act"]) == len(out["obs"]) == len(out["ret"])
    for o, a in zip(out["obs"], out["act"]):
        assert o["mask"][a]  # the expert's choice is a feasible slot
    assert all(r1 >= r2 - 1e-9 for r1, r2 in zip(out["ret"], out["ret"][1:]))  # return-to-go shrinks
    assert out["ret"][0] <= 10 * out["util"] + 1e-6  # never more than the whole episode return
    data = stack([out])
    assert data["cands"].shape[0] == len(out["act"])
