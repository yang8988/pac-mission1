import dataclasses
from collections import Counter

import numpy as np

from palletizing_core import constraints as C
from palletizing_core.candidates import generate
from palletizing_core.config import Config
from palletizing_core.generator import counts_for_fill, make_sequence, random_box_types
from palletizing_core.lookahead import LookaheadPlanner, sample_scenarios
from palletizing_core.model import BoxType, PalletState
from palletizing_core.planner import PLACE
from palletizing_core.scoring import FeatureContext, features
from palletizing_core.simulate import run_episode, validate


def small_cfg(**lookahead):
    cfg = Config()
    la = dict(k0=4, max_scenarios=4, first_round=1, depth=4, time_budget=None, max_rollouts=8)
    la.update(lookahead)
    cfg.lookahead = dataclasses.replace(cfg.lookahead, **la)
    return cfg


def episode(cfg, order="random", seed=0, n_types=4, fill=0.5):
    rng = np.random.default_rng(seed)
    types = random_box_types(rng, n_types)
    counts = counts_for_fill(rng, types, cfg, fill)
    seq = make_sequence(rng, types, counts, order, cfg)
    return types, counts, seq


def test_scenarios_keep_queue_and_respect_inventory():
    cfg = small_cfg(max_scenarios=8, depth=100, adversarial_frac=0.25)
    types = [BoxType("A", 200, 200, 200, 2.0), BoxType("B", 400, 400, 300, 8.0)]
    state = PalletState.new(cfg, types, {"A": 5, "B": 3})
    queue = ["B", "A"]
    scen = sample_scenarios(state, queue, cfg.lookahead, np.random.default_rng(0))
    assert len(scen) == 8
    for s in scen:
        assert s[:2] == queue
        assert Counter(s) == Counter({"A": 5, "B": 3})
    # Scenario 1 is the unfavourable one: after the queue, all small boxes before big ones.
    rest = scen[1][2:]
    assert rest == sorted(rest, key=lambda t: state.types[t].volume)


def test_scenarios_truncated_to_depth():
    cfg = small_cfg(depth=3)
    types = [BoxType("A", 200, 200, 200, 2.0)]
    state = PalletState.new(cfg, types, {"A": 10})
    assert all(len(s) == 3 for s in sample_scenarios(state, [], cfg.lookahead, np.random.default_rng(0)))


def test_lookahead_is_deterministic_and_respects_budget():
    cfg = small_cfg()
    types, counts, seq = episode(cfg)
    a = run_episode(cfg, types, counts, seq, LookaheadPlanner())
    b = run_episode(cfg, types, counts, seq, LookaheadPlanner())
    assert [(p.x, p.y, p.z, p.orientation) for p in a.state.placed] == [
        (p.x, p.y, p.z, p.orientation) for p in b.state.placed
    ]
    assert all(d.info.get("rollouts", 0) <= cfg.lookahead.max_rollouts for d in a.decisions)
    assert any(d.info.get("rollouts", 0) > 0 for d in a.decisions)


def test_lookahead_does_not_mutate_state():
    cfg = small_cfg()
    types, counts, seq = episode(cfg)
    state = PalletState.new(cfg, types, counts)
    for box in seq[:5]:
        state.consume(box.type_id)
        dec = LookaheadPlanner().decide(state, box)
        assert dec.action == PLACE
        state.commit(dec.best)
    H, n, inv = state.H.copy(), len(state.placed), dict(state.inventory)
    LookaheadPlanner().decide(state, seq[5])
    assert np.array_equal(H, state.H) and n == len(state.placed) and inv == state.inventory


def test_lookahead_episodes_never_violate_constraints():
    cfg = small_cfg()
    for order in ("random", "small_first"):
        types, counts, seq = episode(cfg, order)
        res = run_episode(cfg, types, counts, seq, LookaheadPlanner(), lookahead_k=2)
        assert validate(res.state) == []


def test_limited_evaluation_returns_a_subset():
    cfg = Config()
    types, counts, seq = episode(cfg, fill=0.6)
    state = PalletState.new(cfg, types, counts)
    for box in seq[:8]:
        cands, _ = C.evaluate_many(state, box, generate(state, box))
        if cands:
            state.commit(cands[0])
    box = seq[8]
    acts = generate(state, box)
    full, _ = C.evaluate_many(state, box, acts)
    limited, _ = C.evaluate_many(state, box, acts, limit=3)
    assert len(limited) <= 3
    assert {(c.i, c.j, c.o) for c in limited} <= {(c.i, c.j, c.o) for c in full}


def test_fast_roughness_matches_brute_force():
    cfg = Config()
    types, counts, seq = episode(cfg, fill=0.6, seed=3)
    state = PalletState.new(cfg, types, counts)
    for box in seq[:10]:
        cands, _ = C.evaluate_many(state, box, generate(state, box))
        if cands:
            state.commit(cands[len(cands) // 2])
    box = seq[10]
    ctx = FeatureContext(state)
    H = state.H
    nx, ny = H.shape

    def rough(a):
        return np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum()

    cands, _ = C.evaluate_many(state, box, generate(state, box))
    assert cands
    for c in cands:
        a0, a1, b0, b1 = max(0, c.i - 1), min(nx, c.i + c.ni + 1), max(0, c.j - 1), min(ny, c.j + c.nj + 1)
        before = H[a0:a1, b0:b1]
        after = before.copy()
        after[c.i - a0 : c.i - a0 + c.ni, c.j - b0 : c.j - b0 + c.nj] = c.z + c.h
        expected = (rough(after) - rough(before)) / (2 * (c.ni + c.nj) * cfg.pallet.max_height)
        assert abs(features(state, c, ctx)["roughness"] - np.clip(expected, -1, 1)) < 1e-9
