"""The ground shaping along a liaison: the road the arrows lead the player over (guide.lua, the guidance in core.lua) has its ground
shaped like a special stage's (shape.lua), round the car only (guide.shapeWindow), and no other road is.

The extension with the stubbed game and Earth (test_core), Earth's terrain block (test_stage_curation) and the game's ground
markers planner (test_guide), on the Algarve testbed's liaison road1 (18.6 km).
"""
import pytest

lupa = pytest.importorskip("lupa")

from test_core import EARTHRALLY_EXT, lua, logs, launched  # noqa: E402,F401  (the stubbed game and Earth)
from test_guide import STUBS  # noqa: E402
from test_stage_curation import EARTH_SIDE, run_frames  # noqa: E402


# --- guide.shapeWindow on its own ------------------------------------------------------------------------------------------

@pytest.fixture()
def g():
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("package.path = package.path .. ';%s/?.lua'" % EARTHRALLY_EXT.replace("\\", "/"))
    rt.execute("guide = require('beamngEarthRally/guide')")
    rt.execute("LINE = guide.line({ {0, 0, 10}, {10000, 0, 20} })")      # 10 km east
    return rt


def window(rt, s0):
    w = rt.eval("guide.shapeWindow(LINE, %r)" % s0)
    return w["from"], w["to"], [w.s[i] for i in range(1, len(w.s) + 1)], [w.pts[i][1] for i in range(1, len(w.pts) + 1)]


def test_the_shaped_stretch_runs_behind_and_ahead_of_the_car_a_point_every_ten_metres(g):
    frm, to, s, xs = window(g, 2000.0)
    behind, ahead = g.eval("guide.SHAPE_BEHIND_M"), g.eval("guide.SHAPE_AHEAD_M")
    assert frm <= 2000 - behind and to >= 2000 + ahead
    assert s[0] == frm and s[-1] == to and all(b - a == pytest.approx(g.eval("guide.SHAPE_STEP_M")) for a, b in zip(s, s[1:]))
    assert xs == pytest.approx(s)                                          # (on the line: x is the metres along here)
    # (not the whole liaison: the work is round the car)
    assert to - frm <= behind + ahead + g.eval("guide.SHAPE_SHIFT_M") + 1e-6


def test_the_stretch_moves_on_in_steps_on_a_fixed_grid_and_ends_with_the_line(g):
    shift = g.eval("guide.SHAPE_SHIFT_M")
    a, b = window(g, shift + 100.0), window(g, 2 * shift - 1)
    assert a[0] == b[0] and a[1] == b[1]                                   # (within a step: the same stretch, no new job)
    c = window(g, 2 * shift)
    assert c[0] == a[0] + shift
    common = sorted(set(a[2]) & set(c[2]))
    assert len(common) > 100 and all(abs(round(v / 10.0) * 10.0 - v) < 1e-9 for v in common)   # (the same points where they overlap)
    assert g.eval("guide.shapeFrom(LINE, %r)" % (shift + 100.0)) == a[0]
    near_end = window(g, 9900.0)
    assert near_end[1] == pytest.approx(10000.0) and near_end[2][-1] == pytest.approx(10000.0)
    start = window(g, 0.0)
    assert start[0] == 0


# --- in the game: the liaison the arrows lead over is shaped round the car -------------------------------------------------

@pytest.fixture()
def run(lua):
    launched(lua)
    lua.execute(EARTH_SIDE)
    lua.execute(STUBS)
    lua.execute("core.onAnyMissionChanged('started', MISSION)")         # (the loop runs: its scenery and shaping are on)
    return lua


def block_on_liaison(rt, s, spikes, off=0.0):
    """The terrain block centred on the liaison s metres along, its ground a plane with a 4 m spike of the data at each of `spikes`
    (metres along the line, moved `off` m to the side of it)."""
    x, y, z = (rt.eval("at(%r)[%d]" % (s, k)) for k in (1, 2, 3))
    pts = []
    for at_s in spikes:
        px, py = rt.eval("at(%r)[1]" % at_s), rt.eval("at(%r)[2]" % at_s)
        pts.append("{ %r, %r }" % (px, py + off))
    rt.execute("SPIKES_AT = { %s }; GC = { x = %r, y = %r, z = %r }" % (", ".join(pts), x, y, z))
    # (a block's grid is its height above the block's base, which lies under its ground)
    rt.execute("TB = newBlock(%r, %r, %r, true); SNAP = snapshot(TB)" % (x - 128.0, y - 128.0, z - 30.0))
    return x, y


def swap_and_settle(rt):
    rt.execute("package.loaded['beamngEarth/terrain'].swap('b.ter')")
    run_frames(rt, 120)


def spike_left(rt, s):
    """How far the block's ground at the liaison s metres along stands over the plane (the spike's 4 m when it is left)."""
    x, y = rt.eval("at(%r)[1]" % s), rt.eval("at(%r)[2]" % s)
    return rt.eval("worldAt(TB, %r, %r)" % (x, y)) - rt.eval("GC.z + 0.05 * (%r - GC.x) + 0.03 * (%r - GC.y)" % (round(x), round(y)))


def test_the_liaison_the_arrows_lead_over_has_its_ground_shaped_where_the_car_is(run):
    run.execute("coverLine(0, 6000)")
    block_on_liaison(run, 800.0, [780.0, 840.0])
    assert spike_left(run, 780.0) > 3.5                                # (the data's spike: a defect the road does not have)
    run.execute("gameRoutes(700)")                                      # (the game's route: the guidance of road1 begins)
    swap_and_settle(run)
    assert run.eval("TB.sets") > 100
    assert abs(spike_left(run, 780.0)) < 0.5 and abs(spike_left(run, 840.0)) < 0.5
    assert logs(run, "liaison road1 guidance")


def test_without_guidance_the_liaison_is_left_as_earth_made_it(run):
    block_on_liaison(run, 800.0, [780.0, 840.0])
    swap_and_settle(run)
    assert run.eval("sameAs(TB, SNAP)") is True


def test_a_road_beside_the_liaison_is_not_shaped(run):
    # (only the road the player is led over: a spike 60 m to the side of it, on another road or a field, is not ours)
    run.execute("coverLine(0, 6000)")
    block_on_liaison(run, 800.0, [780.0], off=60.0)
    run.execute("gameRoutes(700)")
    swap_and_settle(run)
    assert run.eval("sameAs(TB, SNAP)") is True


def test_the_liaison_far_ahead_is_shaped_only_once_the_car_comes_near(run):
    run.execute("coverLine(0, 18000)")
    run.execute("gameRoutes(0)")
    far = run.eval("guide.SHAPE_SHIFT_M + guide.SHAPE_AHEAD_M") + 400.0     # (beyond the stretch a car at the start has)
    block_on_liaison(run, far, [far])
    swap_and_settle(run)
    assert spike_left(run, far) > 3.5                                     # not yet
    run.execute("driveTo(%r)" % (far - 1500.0))                           # (the car comes within the stretch: it moves on)
    swap_and_settle(run)
    assert abs(spike_left(run, far)) < 0.5
