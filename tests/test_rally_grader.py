"""tools/rally_grader.py: EarthRally's grader (shape.lua) run offline on a terrain block, for earth_writers.py --rally (no Earth tree needed:
a synthetic block and a flat frame).

    python -m pytest tests
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("lupa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import rally_grader  # noqa: E402

M_PER_DEG = 111320.0


class Flat:
    """A frame whose metres are degrees times M_PER_DEG (latitude north, longitude east): enough for a few hundred metres."""
    def geodetic_to_enu(self, la, lo, h):
        return np.asarray(lo, np.float64) * M_PER_DEG, np.asarray(la, np.float64) * M_PER_DEG, 0.0


N, X0, Y0 = 256, -128.0, -128.0
BUMP = (0.0, 0.0)


def block(bump_h=4.0):
    """A plane that rises 2 cm a metre east, with a bump of bump_h (sigma 8 m) on the road at BUMP."""
    x, y = np.meshgrid(X0 + np.arange(N), Y0 + np.arange(N))
    return 100 + 0.02 * x + bump_h * np.exp(-((x - BUMP[0]) ** 2 + (y - BUMP[1]) ** 2) / 128.0)


def geometry(beside=None):
    """A stage along y = 0 from x = -100 to 100, a point every 10 m (lat, lon, h)."""
    pts = [[0.0, x / M_PER_DEG, 100 + 0.02 * x] for x in range(-100, 101, 10)]
    g = {"id": "ss1", "points": pts}
    if beside:
        g["tunnelsBeside"] = [beside]
    return g


def moved(res, before):
    return np.abs(res["heights"] - before) >= 0.02


def test_a_spike_on_the_line_is_graded_and_only_round_the_road():
    h = block()
    res = rally_grader.run(h, X0, Y0, Flat(), {"ss1": geometry()}, [])
    m = moved(res, h)
    assert m.any()
    j, i = np.nonzero(m)
    assert np.abs(Y0 + j).max() <= 5.5 + 6 + 1          # (the road's core and the taper beside it: CORE_M + BLEND_M)
    at = (int(-Y0), int(-X0))
    assert res["heights"][at] < h[at] - 2                # (the bump is taken down at the line)
    assert any("stretch" in m for m in res["logs"])


def test_the_points_beside_a_tunnel_are_kept_as_earth_made_them():
    h = block()
    res = rally_grader.run(h, X0, Y0, Flat(), {"ss1": geometry(beside={"from": 70, "to": 130, "way": 1})}, [])
    assert not moved(res, h).any()
    line = res["lines"][0]
    s = np.asarray(line["s"])
    kept = np.asarray(line["kept"])
    assert kept[(s >= 70) & (s <= 130)].all() and not kept[(s < 70) | (s > 130)].any()


def test_the_heights_are_given_back_in_the_block_s_layout():
    h = block(bump_h=0.0)
    res = rally_grader.run(h, X0, Y0, Flat(), {"ss1": geometry()}, [])
    assert res["heights"].shape == h.shape
    assert np.array_equal(res["heights"], h)            # (nothing to fix: not a sample moved, none shuffled)
    assert math.isclose(res["lines"][0]["x"][0], -100.0, abs_tol=1e-6)
