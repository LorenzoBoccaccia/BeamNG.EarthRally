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


def test_both_eval_sides_wait_for_the_lidar(tmp_path, monkeypatch):
    """A cold place's lidar fetched during one side's build (Boston's Bowker, 2026-11-16): both sides wait for it."""
    import earth_eval as ee
    seen = []

    monkeypatch.setattr(ee, "run_watched", lambda cmd, log_path, env, cwd, watch=None: seen.append(env) or (0, None))

    class A:
        out, image, views, reuse, online, no_buildings, cache = str(tmp_path), "10x10", "top", False, False, False, None
    for side in ee.SIDES:
        ee.build_side({"name": "j", "at": [1.0, 2.0]}, side, "tree", A, lambda m: None)
    assert [e["BNE_LIDAR_GAPS_WAIT"] for e in seen] == [ee.LIDAR_WAIT_S] * 2


# ---- placement on both sides, terrain hits --------------------------------------------------------------------------------
def test_a_two_lane_road_gets_a_truck_each_side_and_a_service_road_one():
    xy = np.array([[-40.0, 0.0], [40.0, 0.0]])
    sides = ev.side_lines([(xy, []), (xy + (0, 50), [])], ["secondary", "service"])
    assert len(sides) == 3
    np.testing.assert_allclose(sides[0][0][:, 1], ev.SIDE_M["secondary"])
    np.testing.assert_allclose(sides[1][0][:, 1], -ev.SIDE_M["secondary"])
    np.testing.assert_allclose(sides[2][0][:, 1], 50.0)
    g = flat_ground()[0]
    surf = ev.Surfaces(g)
    ground = lambda x, y: 0.0       # noqa: E731
    got = ev.vehicles_on_lines(sides[:2], surf, ev.TRUCK_EVERY_M, ground=ground, box=(-40, -40, 40, 40))
    ys = sorted({round(float(b[1]), 2) for b, _a in got})
    assert ys == [-ev.SIDE_M["secondary"], ev.SIDE_M["secondary"]], "both sides kept, not merged as one road's copies"
    twice = ev.vehicles_on_lines(sides[:1] * 2, surf, ev.TRUCK_EVERY_M, ground=ground, box=(-40, -40, 40, 40))
    assert len(twice) == len([b for b, _a in got if b[1] > 0]), "(a copy of the same side: once)"


def test_a_truck_keeps_within_earths_carriageway():
    # (Compiegne: a 5.5 m secondary; its side 1.75 m out put a 2.55 m truck 0.28 m past the edge, into a track's fill
    # standing beside the road under the bridge's end)
    xy = np.array([[-40.0, 0.0], [40.0, 0.0]])
    sides = ev.side_lines([(xy, []), (xy + (0, 50), []), (xy + (0, 100), [])], ["secondary"] * 3, [2.75, 4.0, None])
    assert len(sides) == 6
    np.testing.assert_allclose(sides[0][0][:, 1], 2.75 - ev.TRUCK_LWH[1] / 2)
    np.testing.assert_allclose(sides[1][0][:, 1], -(2.75 - ev.TRUCK_LWH[1] / 2))
    np.testing.assert_allclose(sides[2][0][:, 1], 50.0 + ev.SIDE_M["secondary"])     # (wide enough: as before)
    np.testing.assert_allclose(sides[4][0][:, 1], 100.0 + ev.SIDE_M["secondary"])    # (not known: as before)


def test_terrain_rising_into_a_body_is_a_hit_but_its_own_road_is_not():
    road = flat_ground()
    mound = ev.terrain_mesh(np.where(np.abs(np.arange(-40, 41))[None, :] < 100, 0.0, 0.0) +
                            np.where((np.abs(np.arange(-40, 41)) <= 3)[:, None], 0.0, 2.0), -40, -40, (-40, -40, 40, 40))[0]
    t = truck(0, 0, 0.0)
    assert ck.vehicle_hits(t, road[0]).tolist() == [False]
    assert ck.vehicle_hits(truck(0, 2.0, 0.0), mound).tolist() == [True], "the slope 1 m from its side reaches the body"
    assert ev.GROUND in ev.HIT_BY and ev.MOVED in ev.HIT_BY


def test_the_report_names_only_the_junctions_with_yellow_or_red_first():
    import earth_eval as ee
    res = [{"name": "clean", "checks": {"vehicles_hit": [0, 0], "crossings_cut": [0, 0]}},
           {"name": "few", "checks": {"vehicles_hit": [3, 1]}},
           {"name": "many", "checks": {"vehicles_hit": [9, 9], "crossings_cut": [0, 2]}},
           {"name": "old", "checks": {"vehicles_hit": [4, None]}}]
    assert ee.flagged_junctions(res) == [("many", [("yellow", 9), ("red", 2)]), ("few", [("yellow", 1)])]


def test_a_rigid_truck_on_a_bend_keeps_to_the_line():
    """Daikoku's service loop (radius 25-30 m, vertices 10 m apart): headed along its one segment, a truck's far end stood
    1.3-1.5 m off the line; along the chord its ends stay within 0.6 m."""
    ang = np.radians(np.arange(0, 181, 22.0))
    line = 27.5 * np.stack([np.cos(ang), np.sin(ang)], 1)
    for s in np.arange(6.0, ev._length(line) - 6, 1.7):          # (half a truck from the ends)
        c, h = ev.chord_pose(line, s, 6.0)
        for end in (c + 6.0 * h, c - 6.0 * h):
            d = min(np.hypot(*(ev.point_at(line, q)[0] - end)) for q in np.linspace(0, ev._length(line), 2000))
            assert d < 0.6, (s, d)      # (a 12 m chord of a 27.5 m bend: 0.65 m sagitta, about half each side)


def test_japan_drives_on_the_left_in_the_views_build(monkeypatch):
    import types
    nav = types.SimpleNamespace(border_country=lambda la, lo: "JP", road_rules=lambda c: {"rightHandDrive": c == "JP"})
    monkeypatch.setitem(sys.modules, "navgraph", nav)
    assert ev.drives_left(35.46, 139.68, lambda m: None) is True
    nav.border_country = lambda la, lo: None
    assert ev.drives_left(48.8, 2.3, lambda m: None) is False


def test_the_hit_report_says_where_and_how_deep():
    low = slab(0, 0, 3.0, 4.5, Y, 30, 10)
    vt = np.concatenate([truck(0, 0, 0.0), truck(0, 30, 0.0)])
    cls = np.full(len(low), ev.DECK, np.int8)
    hit = ck.vehicle_hits(vt, low)
    rep = ck.hit_report(vt, ["truck", "truck"], low, cls, hit, {ev.DECK: "bridge deck"})
    assert len(rep) == 1 and rep[0]["kind"] == "truck" and abs(rep[0]["x"]) < 0.1 and abs(rep[0]["y"]) < 0.1
    assert rep[0]["hits"] == {"bridge deck": 1.0}         # (its underside 3 m up, the 4 m truck's roof: 1 m too low)


def test_the_hits_table_lists_the_worst_first_and_every_hit():
    import earth_hits as eh
    rs = [{"name": "a", "error": None, "vehicles": 10, "vehicles_hit": 0, "crossings": 2, "crossings_cut": 0,
           "overview_crossings_cut": 0, "hits": []},
          {"name": "b", "error": None, "vehicles": 20, "vehicles_hit": 2, "crossings": 3, "crossings_cut": 1,
           "overview_crossings_cut": 2, "hits": [{"kind": "truck", "x": 1.0, "y": 2.0, "z": 3.0, "heading": 90,
                                                  "hits": {"ground": 1.2}}]}]
    t = eh.table(rs)
    assert t.index("| b |") < t.index("| a |") and "truck at x 1.0 y 2.0 z 3.0 heading 90: hits ground by 1.2 m" in t


def test_struck_names_the_triangles_a_body_cuts_not_the_ones_beside_it():
    base = np.array([0.0, 0.0, 0.0])
    vt = ev.box_triangles(base, np.array([1.0, 0.0, 0.0]), 12.0, 2.55, 4.0)
    inside = np.array([[[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]])        # (a slab through the body)
    beside = inside + np.array([0.0, 10.0, 0.0])                                       # (10 m to its side)
    got = ck.struck(vt, np.concatenate([inside, beside]))
    assert got.tolist() == [True, False]


def test_a_hit_is_named_by_where_it_is_what_it_stands_on_and_what_it_strikes():
    class Frame:
        def enu_to_geodetic(self, x, y, z):
            return 45.0 + np.asarray(y) / 111000.0, 7.0 + np.asarray(x) / 78000.0, z
    map_l = {"road": [(np.array([[-50.0, 0.0], [50.0, 0.0]]), []), (np.array([[-50.0, 6.0], [50.0, 6.0]]), [(0.0, 1.0, "bridge")])],
             "road_id": ["own-id", "deck-id"], "road_cls": ["primary", "motorway"], "rail": [], "rail_id": [], "rail_cls": []}
    hits = [{"kind": "truck", "x": 0.0, "y": 1.5, "z": 0.0, "body_base": 0.5, "body_top": 4.5, "heading": 90.0,
             "hits": {"bridge deck": 0.3, "ground": 0.2},
             "struck": {"bridge deck": {"n": 2, "x": 0.0, "y": 5.0, "z_lo": 0.6, "z_hi": 1.0},
                        "ground": {"n": 1, "x": 1.0, "y": 0.5, "z_lo": 0.4, "z_hi": 0.7}}}]
    line, = ev.named_hits(hits, map_l, Frame())
    assert "truck at 45.0000" in line and "on road own-id (primary, ground)" in line
    assert "bridge deck at x 0.0 y 5.0 z 0.6..1.0" in line and "of road deck-id (motorway, bridge)" in line
    assert "ground at x 1.0 y 0.5" in line and "graded by nearest road own-id" in line


def test_a_rail_tunnel_the_tree_does_not_draw_has_no_surface_to_cut():
    # (Vienna: a tram tunnel under a busway "cut" by the road tunnels' and the cuttings' walls round it, Earth drawing no tube)
    g, gc = flat_ground()
    wall = slab(0, 0, -3.0, 1.0, Y, 6, 1)            # a tunnel-class wall standing through the road's ground
    tris = np.concatenate([g, wall])
    cls = np.concatenate([gc, np.full(12, ev.TUNNEL, np.int8)])
    lines = {"road": [(np.array([[-40.0, 0.0], [40.0, 0.0]]), [])],
             "rail": [(np.array([[0.0, -40.0], [0.0, 40.0]]), [(0.0, 1.0, "tunnel")])]}
    _hit, drawn = ck.penetrations(tris, cls, lines, dict(CLASSES, rail_tunnels=True))
    _hit, undrawn = ck.penetrations(tris, cls, lines, dict(CLASSES, rail_tunnels=False))
    assert ck.count_cut(drawn) == 1, "a drawn tube through the road's ground still cuts"
    assert ck.count_cut(undrawn) == 0


# ---- road steps and road ends ---------------------------------------------------------------------------------------------
def heightfield(xs, ys, f):
    """Terrain triangles (n, 3, 3), classed GROUND, over the grid xs x ys at heights f(x, y) (two per cell, facing up)."""
    T = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            a, b, c, d = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            v = lambda p: (p[0], p[1], f(*p))      # noqa: E731
            T += [(v(a), v(b), v(c)), (v(a), v(c), v(d))]
    return np.array(T, np.float64)


def plate(x0, x1, y0, y1, z):
    """A deck's top: two triangles (2, 3, 3) facing up over the rectangle, at z."""
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return np.array([(a, b, c), (a, c, d)], np.float64)


def road_scene(terrain, decks=()):
    tris = np.concatenate([terrain] + list(decks))
    cls = np.concatenate([np.full(len(terrain), ev.GROUND, np.int8)] + [np.full(len(d), ev.DECK, np.int8) for d in decks])
    return tris, cls


def line(pts, ranges=(), half=3.0, rid="r"):
    return np.array(pts, np.float64), list(ranges), half, rid


def lines_of(*ls):
    return {"road": [(xy, r) for xy, r, _h, _i in ls], "road_half": [h for *_a, h, _i in ls], "road_id": [i for *_a, i in ls]}


def test_a_step_of_0_1_m_beside_a_road_counts_and_one_of_0_09_does_not():
    road = lines_of(line([(-30, 0), (30, 0)]))
    for dz, counted in ((0.1, True), (0.09, False)):
        tris, cls = road_scene(heightfield([-40.0, 40.0], [-40.0, 3.5, 3.51, 40.0], lambda x, y: dz if y > 3.505 else 0.0))
        rc = ck.road_checks(tris, cls, road, CLASSES)
        assert (ck.count_steps(rc["steps"]) > 0) == counted
        if counted:
            assert 55 <= ck.count_steps(rc["steps"]) <= 61 and all(s["y"] > 0 for s in rc["steps"])      # (one side, the length)


def test_a_slope_beside_a_road_is_no_step_but_a_cliff_is():
    road = lines_of(line([(-30, 0), (30, 0)]))
    tris, cls = road_scene(heightfield([-40.0, 40.0], [-40.0, 40.0], lambda x, y: 0.2 * y))
    assert ck.count_steps(ck.road_checks(tris, cls, road, CLASSES)["steps"]) == 0
    tris, cls = road_scene(heightfield([-40.0, 40.0], [-40.0, 3.5, 3.51, 40.0], lambda x, y: -5.0 if y > 3.505 else 0.0))
    runs = ck.step_runs(ck.road_checks(tris, cls, road, CLASSES)["steps"])
    assert len(runs) == 1 and runs[0]["step"] == 5.0 and runs[0]["kind"] == "ground"


def _overpass(approach_z):
    """A deck at z 3 over a ground road at z 0 in a trench, its approaches at approach_z; the ground road ends at both its ends."""
    terrain = heightfield([-40.0, 40.0], [-40.0, -10.01, -10.0, 10.0, 10.01, 40.0],
                          lambda x, y: approach_z if abs(y) > 10.005 else 0.0)
    deck = plate(-4.0, 4.0, -10.0, 10.0, 3.0)
    tris, cls = road_scene(terrain, [deck])
    lines = lines_of(line([(-30, 0), (30, 0)], rid="road"), line([(0, -40), (0, 40)], [(0.375, 0.625, "bridge")], 4.0, "over"))
    return ck.road_checks(tris, cls, lines, CLASSES)


def test_a_deck_over_a_ground_road_and_its_approach_meeting_that_roads_level_count_nothing():
    rc = _overpass(3.0)
    assert rc["steps"] == []
    assert not [e for e in rc["ends"] if e["kind"] in ("broken", "missing")]
    assert [e["kind"] for e in rc["ends"] if e["what"] == "bridge"] == ["joined", "joined"]


def test_a_deck_ending_above_its_approach_is_a_step_and_a_broken_join():
    rc = _overpass(2.0)
    assert ck.count_steps(rc["steps"]) > 0 and {s["kind"] for s in rc["steps"]} == {"bridge"}
    assert ck.count_ends(rc["ends"], "broken") == 2
    assert ck.count_ends(rc["ends"], "missing") == 0


def test_a_road_end_with_nothing_abutting_is_missing_where_its_node_is_shared_and_dead_where_it_is_not():
    tris, cls = road_scene(heightfield([-40.0, 40.0], [-40.0, 40.0], lambda x, y: 0.0))
    for nodes, kind in (([(1, 2), (2, 3)], "missing"), ([(1, 2), (4, 3)], "dead")):
        # (two ground roads 20 m apart whose ends are not near: only the OSM node says they should join)
        lines = lines_of(line([(-30, 0), (-15, 0)], rid="a"), line([(15, 0), (30, 0)], rid="b"))
        lines["road_nodes"] = nodes
        rc = ck.road_checks(tris, cls, lines, CLASSES)
        assert [e["kind"] for e in rc["ends"] if e["node"] == 2] == [kind] * (2 if kind == "missing" else 1)


def test_ends_that_touch_at_another_height_are_a_broken_join_whatever_the_graph_says():
    # two decks end to end, the second 0.3 m higher; no node ids at all (nothing is "connected")
    d1 = plate(-4.0, 4.0, -20.0, 0.0, 3.0)
    d2 = plate(-4.0, 4.0, 0.0, 20.0, 3.3)
    tris, cls = road_scene(heightfield([-40.0, 40.0], [-40.0, 40.0], lambda x, y: 0.0), [d1, d2])
    lines = lines_of(line([(0, -20), (0, 0)], [(0.0, 1.0, "bridge")], 4.0, "a"), line([(0, 0), (0, 20)], [(0.0, 1.0, "bridge")], 4.0, "b"))
    rc = ck.road_checks(tris, cls, lines, CLASSES)
    assert ck.count_ends(rc["ends"], "broken") == 2
