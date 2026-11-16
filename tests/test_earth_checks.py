"""tools/earth_checks.py: vehicles hitting what isn't theirs, meshes cutting each other at crossings of different levels; built
scenes, no Earth tree needed.

    python -m pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import earth_checks as ck  # noqa: E402
import earth_view as ev  # noqa: E402

X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
CLASSES = ev.CHECK_CLASSES


def slab(cx, cy, z0, z1, along, L, W):
    """A closed box (12, 3, 3) from z0 to z1, L long along `along`, W wide, round (cx, cy)."""
    return ev.box_triangles((cx, cy, z0), along, L, W, z1 - z0)


def truck(x, y, z, along=X, h=4.0):
    return ev.box_triangles((x, y, z), along, 12.0, 2.55, h)


def flat_ground(half=40, z=0.0):
    zz = np.full((2 * half + 1, 2 * half + 1), z)
    return ev.terrain_mesh(zz, -half, -half, (-half, -half, half, half))


# ---- vehicles -------------------------------------------------------------------------------------------------------------
def test_the_body_starts_over_the_wheels():
    c = ck.vehicle_bodies(truck(0, 0, 10.0))
    assert np.allclose(c[0, :4, 2], 10.0 + ck.BODY_BASE_M) and np.allclose(c[0, 4:, 2], 14.0)


def test_a_truck_under_a_low_deck_hits_it_and_one_under_a_high_deck_doesnt():
    low = slab(0, 0, 3.0, 4.5, Y, 30, 10)
    high = slab(0, 0, 5.0, 6.5, Y, 30, 10)
    assert ck.vehicle_hits(truck(0, 0, 0.0), low).tolist() == [True]
    assert ck.vehicle_hits(truck(0, 0, 0.0), high).tolist() == [False]


def test_a_truck_on_its_own_deck_and_its_own_road_never_counts():
    deck = slab(0, 0, 4.0, 6.0, X, 40, 10)
    road = flat_ground()[0]
    assert ck.vehicle_hits(truck(0, 0, 6.0), deck).tolist() == [False]
    assert ck.vehicle_hits(truck(0, 0, 0.0), road).tolist() == [False]
    assert ck.vehicle_hits(truck(0, 0, -0.3), road).tolist() == [False]       # (sunk 0.3 m: still under the body)


def test_a_wagon_through_a_deck_hits_it():
    wagon = ev.box_triangles((0, 0, 0.0), X, 20.0, 3.0, 4.3)
    deck = slab(0, 0, 2.0, 3.0, Y, 30, 8)
    assert ck.vehicle_hits(wagon, deck).tolist() == [True]


def test_a_small_mesh_wholly_inside_the_body_counts():
    inside = np.array([[[0, 0, 2.0], [0.5, 0, 2.0], [0, 0.5, 2.0]]])
    assert ck.vehicle_hits(truck(0, 0, 0.0), inside).tolist() == [True]


def test_hits_are_per_vehicle():
    deck = slab(0, 0, 3.0, 4.5, Y, 10, 10)
    vt = np.concatenate([truck(0, 0, 0.0), truck(0, 30, 0.0)])
    assert ck.vehicle_hits(vt, deck).tolist() == [True, False]


# ---- crossings ------------------------------------------------------------------------------------------------------------
def scene(deck_z0, rail_line_kind="bridge", road_kind=None, rest=0.0):
    """A railway north-south over a road east-west at (0, 0): the road painted on flat ground, the railway on a deck from deck_z0
    (1 m thick) with its track on it (sunk rest m into it)."""
    g, gc = flat_ground()
    deck = slab(0, -15, deck_z0, deck_z0 + 1.0, Y, 30, 6)
    rail = slab(0, -15, deck_z0 + 1.0 - rest, deck_z0 + 1.5, Y, 30, 3)
    tris = np.concatenate([g, deck, rail])
    cls = np.concatenate([gc, np.full(12, ev.DECK, np.int8), np.full(12, ev.RAIL, np.int8)])
    r = [(0.0, 1.0, rail_line_kind)] if rail_line_kind != "ground" else []
    lines = {"road": [(np.array([[-40.0, 0.0], [40.0, 0.0]]), [(0.0, 1.0, road_kind)] if road_kind else [])],
             "rail": [(np.array([[0.0, -40.0], [0.0, 40.0]]), r)]}
    return tris, cls, lines


def test_a_crossing_at_different_levels_is_found_once():
    _t, _c, lines = scene(5.0)
    x = ck.level_crossings(lines)
    assert len(x) == 1 and abs(x[0][0]) < 1e-9 and abs(x[0][1]) < 1e-9
    assert x[0][2][0] == "bridge" and x[0][2][3] and x[0][3][0] == "ground"      # the railway's bridge over the ground road


def test_a_level_crossing_at_grade_isnt_a_crossing_of_levels():
    _t, _c, lines = scene(5.0, rail_line_kind="ground")
    assert ck.level_crossings(lines) == []


def test_a_deck_high_over_the_road_doesnt_cut_it():
    tris, cls, lines = scene(5.0)
    hit, xs = ck.penetrations(tris, cls, lines, CLASSES)
    assert not hit.any() and xs == [(0.0, 0.0, False)] and ck.count_cut(xs) == 0


def test_a_deck_sunk_into_the_road_cuts_it_both_sides_red():
    tris, cls, lines = scene(-0.6)
    hit, xs = ck.penetrations(tris, cls, lines, CLASSES)
    assert ck.count_cut(xs) == 1
    assert hit[cls == ev.DECK].any() and hit[cls == ev.GROUND].any()
    far = np.abs(tris.mean(axis=1)[:, 1]) > ck.GROUND_CORRIDOR_M + 1
    assert not hit[(cls == ev.GROUND) & far].any()          # only the road's own ground


def test_a_track_resting_on_its_deck_is_no_cut():
    tris, cls, lines = scene(5.0, rest=0.1)
    hit, xs = ck.penetrations(tris, cls, lines, CLASSES)
    assert not hit.any()


def test_a_deck_resting_on_the_ground_within_rest_m_is_no_cut():
    tris, cls, lines = scene(-0.2)
    hit, _xs = ck.penetrations(tris, cls, lines, CLASSES)
    assert not hit.any()


def test_nearby_cuts_count_as_one_crossing():
    assert ck.count_cut([(0, 0, True), (3, 0, True), (50, 0, True), (80, 0, False)]) == 2
    assert ck.count_cut([(0, 0, True), (50, 0, True)], box=(-10, -10, 10, 10)) == 1


def test_a_short_road_tunnel_is_its_ground_a_long_one_its_tube():
    g, gc = flat_ground(z=0.0)
    tube = slab(0, 0, -6.0, 0.4, X, 40, 8)                  # a tube whose roof cuts the ground over it
    tris = np.concatenate([g, tube])
    cls = np.concatenate([gc, np.full(12, ev.TUNNEL, np.int8)])
    rail = (np.array([[0.0, -40.0], [0.0, 40.0]]), [])
    long_ = {"road": [(np.array([[-200.0, 0.0], [200.0, 0.0]]), [(0.0, 1.0, "tunnel")])], "rail": [rail]}
    hit, xs = ck.penetrations(tris, cls, long_, CLASSES)
    assert ck.count_cut(xs) == 0        # (the railway's surface is its track: none here; the tube is the road's own)
    rail_track = slab(0, 0, 0.0, 0.5, Y, 80, 3)
    tris2 = np.concatenate([tris, rail_track])
    cls2 = np.concatenate([cls, np.full(12, ev.RAIL, np.int8)])
    hit, xs = ck.penetrations(tris2, cls2, long_, CLASSES)
    assert ck.count_cut(xs) == 1 and hit[cls2 == ev.TUNNEL].any()     # the tube's roof through the track over it


# ---- the eval's counts ----------------------------------------------------------------------------------------------------
def test_the_eval_reports_a_check_that_grew_as_worse(tmp_path):
    import earth_eval as ee
    r = ee.compare_checks({"vehicles_hit": 2, "crossings_cut": 1}, {"vehicles_hit": 3, "crossings_cut": 0})
    assert r["worse"] == ["vehicles_hit"] and r["checks"]["crossings_cut"] == [1, 0]
    assert ee.compare_checks({}, {"vehicles_hit": 5})["worse"] == []         # (a side without counts is never worse)

    class A:
        before, after, out = "b", "a", str(tmp_path)
    tris = np.concatenate([ev.box_triangles((0, 0, 0), X, 4, 4, 0.01)])
    g = ev.surface_grid(tris, np.full(12, ev.GROUND, np.int8), (-2, -2, 2, 2), step=0.5)
    res = {"name": "x", **ee.compare(g, g), **r}
    table = ee.write_report([res], A)
    assert "| x | **worse**, same |" in table and "**2 → 3**" in table and "| 1 → 0 |" in table
    assert ee.verdict_line(res).startswith("WORSE, same")
