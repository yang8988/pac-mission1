import numpy as np
import pytest

from palletizing_core.config import Config
from palletizing_core.planner import GreedyPlanner
from palletizing_core.rl.env import PalletEnv, TaskDistribution, greedy_slot
from palletizing_core.rl.obs import CAND_DIM, N_CANDS
from palletizing_core.simulate import run_episode, validate


def spec(seed=0, order="random"):
    tasks = TaskDistribution(n_types=(4, 4), fill=(0.6, 0.6), orders=(order,), order_probs=(1.0,))
    return tasks.sample(np.random.default_rng(seed), Config())


def test_observation_shapes_and_mask():
    env = PalletEnv(seed=0)
    obs, info = env.reset(spec())
    assert not info["done"]
    assert obs["cands"].shape == (N_CANDS, CAND_DIM)
    assert obs["hm"].ndim == 3 and obs["hm"].dtype == np.float32
    assert obs["mask"].sum() == sum(s is not None for s in env.slots) > 0
    assert np.all(obs["cands"][~obs["mask"]] == 0)


def test_following_the_teacher_reproduces_the_greedy_planner():
    s = spec(1)
    env = PalletEnv(seed=0, shuffle_slots=True)
    obs, info = env.reset(s)
    done, ret = info["done"], 0.0
    while not done:
        obs, r, done, _, inf = env.step(greedy_slot(env.slots))
        ret += r
    ref = run_episode(Config(), s.types, s.counts, s.sequence, GreedyPlanner())
    assert env.state.utilization() == pytest.approx(ref.state.utilization())
    assert ret == pytest.approx(10 * ref.state.utilization())


def test_random_masked_actions_never_violate_constraints():
    rng = np.random.default_rng(0)
    for seed in range(3):
        env = PalletEnv(seed=seed)
        obs, info = env.reset(spec(seed, "small_first"))
        done = info["done"]
        while not done:
            a = rng.choice(np.flatnonzero(env.action_masks()))
            obs, _, done, _, _ = env.step(a)
        assert validate(env.state) == []


def test_masked_action_raises():
    env = PalletEnv(seed=0)
    env.reset(spec())
    masked = np.flatnonzero(~env.action_masks())
    if masked.size:
        with pytest.raises(ValueError):
            env.step(int(masked[0]))


def test_policy_forward_and_planner():
    torch = pytest.importorskip("torch")
    from palletizing_core.rl.obs import obs_spec
    from palletizing_core.rl.planner import RLPlanner
    from palletizing_core.rl.policy import ActorCritic, to_tensors

    env = PalletEnv(seed=0)
    obs, _ = env.reset(spec())
    model = ActorCritic(obs_spec(env.state).hm_shape)
    logits, value = model(to_tensors([obs, obs]))
    assert logits.shape == (2, N_CANDS) and value.shape == (2,)
    probs = torch.softmax(logits, -1)
    assert torch.all(probs[:, ~torch.as_tensor(obs["mask"])] == 0)

    s = spec(2)
    res = run_episode(Config(), s.types, s.counts, s.sequence, RLPlanner(model))
    assert validate(res.state) == []
    assert res.state.placed
