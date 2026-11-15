"""Offline tests for tools/make_pacenotes.py: the game's shaping of the ground it measures, and the notes it makes of a special on
a ground given here (Earth's own ground is the tool's run, not the tests': it needs Earth's server and its cache).

BeamNG.Earth's geodesy.lua is found at $BEAMNG_EARTH_DIR, else the sibling folder ../BeamNG.Earth.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

lupa = pytest.importorskip("lupa")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import make_pacenotes as mp  # noqa: E402

EARTH = os.environ.get("BEAMNG_EARTH_DIR") or os.path.join(os.path.dirname(ROOT), "BeamNG.Earth")
if not os.path.isdir(os.path.join(EARTH, "mod", "beamng_earth", "lua", "ge", "extensions")):
    pytest.skip("BeamNG.Earth not found (set BEAMNG_EARTH_DIR)", allow_module_level=True)


@pytest.fixture(scope="module")
def rt():
    return mp.lua_runtime(EARTH)


@pytest.fixture(scope="module")
def mods(rt):
    return tuple(rt.eval("(require('%s'))" % m) for m in
                 ("beamngEarthRally/curate", "beamngEarthRally/pacenotes", "beamngEarthRally/legs", "beamngEarth/geodesy"))


def test_the_shaping_takes_a_spike_out_but_not_beside_a_tunnel(rt, mods):
    curate = mods[0]
    s = np.arange(0.0, 400.0, 10.0)
    G = 100.0 + 0.05 * s
    G[20] += 3.5                                              # a 3.5 m roof in the data, 200 m along (under MAX_CHANGE_M)
    held = np.zeros(len(s), bool)
    fixed = mp.shaped(curate, rt, G, s, held, 2.0)
    assert abs(fixed[20] - (100.0 + 0.05 * 200)) < 0.01     # laid on the grade ...
    assert np.allclose(fixed[:18], G[:18]) and np.allclose(fixed[23:], G[23:])     # ... and nothing beyond its taper
    held[15:26] = True                                        # (Earth's ground beside a tunnel: kept as it is)
    assert np.allclose(mp.shaped(curate, rt, G, s, held, 2.0), G)
    dip = 100.0 + 0.05 * s
    dip[20] -= 5.0                                            # a dip is never a spike (bumps only)
    assert np.allclose(mp.shaped(curate, rt, dip, s, np.zeros(len(s), bool), 2.0), dip)


class BumpGround:
    """A level ground with a bump `top` m high, `sigma` m wide, at (lat, lon)."""

    def __init__(self, lat, lon, top, sigma):
        self.lat, self.lon, self.top, self.sigma = lat, lon, top, sigma

    def at(self, la, lo):
        mlat = 111132.0
        mlon = 111320.0 * math.cos(math.radians(self.lat))
        d = np.hypot((np.asarray(la) - self.lat) * mlat, (np.asarray(lo) - self.lon) * mlon)
        return 250.0 + self.top * np.exp(-0.5 * (d / self.sigma) ** 2)


def popoli():
    with open(os.path.join(ROOT, "editions", "svolte-di-popoli", "geometry", "ss1.json"), encoding="utf-8") as f:
        geo = json.load(f)
    geo.pop("pacenotes", None)
    return geo


def at_along(points, metres):
    acc = 0.0
    for a, b in zip(points, points[1:]):
        d = math.hypot((b[0] - a[0]) * 111132.0, (b[1] - a[1]) * 111320.0 * math.cos(math.radians(a[0])))
        if acc + d >= metres:
            f = (metres - acc) / d
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        acc += d
    return points[-1][0], points[-1][1]


def test_a_bump_the_car_reaches_fast_is_called_a_jump_and_every_note_is_stored_as_places(rt, mods):
    geo = popoli()
    lat, lon = at_along(geo["points"], 270.0)
    notes = mp.special_notes(rt, mods, BumpGround(lat, lon, 6.0, 12.0), geo, 3.0, log=lambda *_: None)
    jumps = [n for n in notes if n["items"].get("2", {}).get("type") == "jump" and "4" not in n["items"]]
    assert len(jumps) == 1
    cr = jumps[0]["earthrally"]["crest"]
    assert abs(cr["atM"] - 270) < 30 and cr["unload"] >= 1.0 and cr["speedKmh"] > cr["liftKmh"]
    corners = [n for n in notes if "4" in n["items"]]
    assert len(corners) > 30 and all(n["items"]["4"]["type"] == "corner" for n in corners)
    for n in notes:
        for k in ("start", "startAhead", "end", "endAhead"):
            assert len(n[k]) == 3 and 42.1 < n[k][0] < 42.3 and 13.7 < n[k][1] < 13.9
        # the ahead point is AHEAD_M along the road from its waypoint
        d = math.hypot((n["startAhead"][0] - n["start"][0]) * 111132.0,
                       (n["startAhead"][1] - n["start"][1]) * 111320.0 * math.cos(math.radians(42.17)))
        assert abs(d - mp.AHEAD_M) < 0.5
    # the same ground without the bump: no jump
    flat = mp.special_notes(rt, mods, BumpGround(lat, lon, 0.0, 12.0), geo, 3.0, log=lambda *_: None)
    assert not [n for n in flat if "crest" in n["earthrally"]]
    assert len([n for n in flat if "4" in n["items"]]) == len(corners)
