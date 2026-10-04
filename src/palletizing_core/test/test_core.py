from pathlib import Path

import numpy as np
import pytest

from palletizing_core import constraints as C
from palletizing_core.config import GRAVITY, Config, load_config
from palletizing_core.generator import ORDERS, counts_for_fill, make_sequence, random_box_types
from palletizing_core.model import Box, BoxType, PalletState
from palletizing_core.planner import PLACE, REJECT, STRATEGIES, GreedyPlanner
from palletizing_core.simulate import run_episode, validate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def make_box(w, d, h, mass=5.0, max_load=1e6, box_id="b", type_id="T"):
    return Box(box_id, type_id, w, d, h, mass, max_load)


def empty_state(cfg=None, n=100):
    cfg = cfg or Config()
    return PalletState(cfg=cfg, inventory={"T": n}, total_boxes=n)


def place(state, box, i, j, o=0):
    cand, reason = C.evaluate(state, box, i, j, o)
    assert cand is not None, reason
    return state.commit(cand)


def test_first_box_on_empty_pallet_is_on_the_deck():
    state = empty_state()
    dec = GreedyPlanner().decide(state, make_box(400, 300, 200))
    assert dec.action == PLACE
    assert dec.best.z == 0
    assert dec.best.support_ratio == 1.0


def test_out_of_bounds_is_masked():
    state = empty_state()
    cand, reason = C.evaluate(state, make_box(400, 300, 200), state.H.shape[0] - 10, 0, 0)
    assert cand is None and reason == C.OUT_OF_BOUNDS


def test_height_limit_is_masked():
    state = empty_state()
    cand, reason = C.evaluate(state, make_box(400, 300, 1600), 0, 0, 0)
    assert cand is None and reason == C.TOO_HIGH


def test_overhang_fails_support_area():
    state = empty_state()
    place(state, make_box(300, 300, 200), 0, 0)
    cand, reason = C.evaluate(state, make_box(500, 500, 200), 0, 0, 0)  # 36% supported
    assert cand is None and reason == C.SUPPORT_AREA


def test_cog_outside_support_polygon_is_masked():
    cfg = Config()
    cfg.constraints.min_support_ratio = 0.3
    state = empty_state(cfg)
    place(state, make_box(300, 300, 200), 0, 0)
    # 50% supported, but the COG (x=300) sits exactly on the support edge -> inside margin fails.
    cand, reason = C.evaluate(state, make_box(600, 300, 200), 0, 0, 0)
    assert cand is None and reason == C.SUPPORT_COG


def test_heavy_box_on_weak_box_is_masked():
    state = empty_state()
    place(state, make_box(400, 400, 200, mass=2.0, max_load=50.0), 0, 0)
    cand, reason = C.evaluate(state, make_box(400, 400, 200, mass=20.0), 0, 0, 0)
    assert cand is None and reason == C.OVERLOAD
    cand, _ = C.evaluate(state, make_box(400, 400, 200, mass=3.0), 0, 0, 0)
    assert cand is not None


def test_load_propagates_through_the_stack():
    state = empty_state()
    place(state, make_box(400, 400, 200, mass=10.0), 0, 0)
    place(state, make_box(400, 400, 200, mass=7.0), 0, 0)
    place(state, make_box(400, 400, 200, mass=4.0), 0, 0)
    assert state.placed[0].load == pytest.approx((7.0 + 4.0) * GRAVITY)
    assert state.placed[1].load == pytest.approx(4.0 * GRAVITY)
    assert state.placed[2].load == 0.0


def test_load_splits_by_contact_area():
    state = empty_state()
    place(state, make_box(300, 400, 200), 0, 0)
    place(state, make_box(300, 400, 200), 30, 0)
    place(state, make_box(400, 400, 200, mass=10.0), 10, 0)  # 200 mm on A, 200 mm on B
    assert state.placed[0].load == pytest.approx(5.0 * GRAVITY)
    assert state.placed[1].load == pytest.approx(5.0 * GRAVITY)


def test_gripper_overhang_collision_is_masked():
    state = empty_state()
    place(state, make_box(300, 300, 600), 0, 0)  # tall neighbour
    cand, reason = C.evaluate(state, make_box(100, 100, 100), 30, 0, 0)  # pad (300x200) overhangs it
    assert cand is None and reason == C.GRIPPER_CLEARANCE


def test_unreachable_is_masked():
    cfg = Config()
    cfg.robot.reach = 500.0
    state = empty_state(cfg)
    cand, reason = C.evaluate(state, make_box(200, 200, 200), 100, 80, 0)
    assert cand is None and reason == C.UNREACHABLE


def test_commit_updates_heightmap_and_extreme_points():
    state = empty_state()
    place(state, make_box(400, 300, 250), 0, 0)
    assert state.H[:40, :30].max() == 250 and state.H[40:, :].max() == 0
    assert (40, 0) in state.ep and (0, 30) in state.ep


def test_decide_does_not_mutate_state():
    state = empty_state()
    place(state, make_box(400, 300, 250), 0, 0)
    H, ep, n = state.H.copy(), set(state.ep), len(state.placed)
    GreedyPlanner().decide(state, make_box(300, 300, 200))
    assert np.array_equal(H, state.H) and ep == state.ep and n == len(state.placed)


def test_damaged_and_overweight_boxes_are_rejected():
    state = empty_state()
    bad = make_box(300, 300, 200)
    bad.damaged = True
    assert GreedyPlanner().decide(state, bad).reason == "damaged"
    heavy = make_box(300, 300, 200, mass=state.cfg.robot.payload + 1)
    dec = GreedyPlanner().decide(state, heavy)
    assert dec.action == REJECT and dec.reason == "payload"


def test_convex_hull_and_margin():
    hull = C.convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    assert len(hull) == 4
    assert C.point_in_convex(hull, (5, 5), margin=4.9)
    assert not C.point_in_convex(hull, (5, 5), margin=5.1)
    assert not C.point_in_convex(hull, (11, 5))


def test_box_orientations():
    t = BoxType("T", 400, 300, 200, 5.0, orientations=(0, 1, 2))
    b = Box.from_type("b", t)
    assert b.dims(0) == (400, 300, 200)
    assert b.dims(1) == (300, 400, 200)
    assert b.dims(2) == (400, 200, 300)


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(CONFIG_DIR / "default.yaml") == Config()


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("order", ORDERS)
def test_episodes_never_violate_constraints(strategy, order):
    cfg = Config()
    for seed in range(2):
        rng = np.random.default_rng(seed)
        types = random_box_types(rng, 5)
        counts = counts_for_fill(rng, types, cfg, 0.6)
        seq = make_sequence(rng, types, counts, order, cfg)
        res = run_episode(cfg, types, counts, seq, GreedyPlanner(strategy))
        assert validate(res.state) == []
        assert len(res.state.placed) + len(res.state.rejected) == len(seq)
        assert res.state.placed, "expected at least one placement"
