"""Offline tests for pacenotes.lua: the corner scanner and the stock notebook it writes, on synthetic roads.

    python -m pytest tests

Needs only lupa; the real stage (ss1) is checked in test_legs.py.
"""
import math
import os

import pytest

lupa = pytest.importorskip("lupa")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EARTHRALLY_EXT = os.path.join(ROOT, "mod", "beamng_earthrally", "lua", "ge", "extensions")


@pytest.fixture(scope="module")
def lua():
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("math.atan2 = math.atan2 or function(y, x) return math.atan(y, x) end")
    rt.execute("package.path = package.path .. ';%s/?.lua'" % EARTHRALLY_EXT.replace("\\", "/"))
    return rt


@pytest.fixture(scope="module")
def pn(lua):
    return lua.eval("(require('beamngEarthRally/pacenotes'))")


def road(*parts, step=10.0):
    """A road from straights ('s', metres) and arcs ('l' or 'r', radius m, degrees), heading east (+x) at first, as world
    points every `step` metres at a height that climbs 1 m per 100 m."""
    x = y = 0.0
    th = 0.0
    pts = [(x, y, 0.0)]
    for part in parts:
        if part[0] == "s":
            n = max(1, round(part[1] / step))
            for _ in range(n):
                x, y = x + part[1] / n * math.cos(th), y + part[1] / n * math.sin(th)
                pts.append((x, y, 0.0))
        else:
            sign = 1 if part[0] == "l" else -1
            radius, total = part[1], math.radians(part[2])
            n = max(2, math.ceil(radius * total / step))
            for _ in range(n):
                d = total / n
                # a chord of the arc: the heading turns by d, the road by the chord
                th += sign * d / 2
                chord = 2 * radius * math.sin(d / 2)
                x, y = x + chord * math.cos(th), y + chord * math.sin(th)
                th += sign * d / 2
                pts.append((x, y, 0.0))
    out, acc = [], 0.0
    for i, p in enumerate(pts):
        if i:
            acc += math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1])
        out.append((p[0], p[1], acc / 100.0))
    return out


def scan(lua, pn, pts):
    t = lua.table()
    for i, p in enumerate(pts, 1):
        t[i] = lua.table(*p)
    corners, _ = pn.scan(t)
    return [corners[i] for i in range(1, len(corners) + 1)], t


def test_a_straight_road_has_no_corners(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 2000)))
    assert corners == []


def test_a_gentle_wobble_is_not_a_corner(lua, pn):
    # a 6 degree bend either way: under what is worth a call
    corners, _ = scan(lua, pn, road(("s", 300), ("l", 200, 6), ("s", 300), ("r", 200, 6), ("s", 300)))
    assert corners == []


def test_a_90_degree_left(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 300), ("l", 25, 90), ("s", 300)))
    assert len(corners) == 1
    c = corners[0]
    assert c["direction"] == "left"
    assert abs(c["angle"] - 90) < 8
    assert abs(c["radius"] - 25) < 6
    assert c["severity"] == 2               # a 50 m circle: the stock "two" is 35-65 m
    assert c["descriptor"] is None
    # the arc is 300 m to 339 m along the road: the corner's two ends hug it
    assert 285 <= c["s0"] <= 305 and 335 <= c["s1"] <= 355


def test_a_hairpin_to_the_right(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 300), ("r", 12, 180), ("s", 300)))
    assert len(corners) == 1
    c = corners[0]
    assert c["direction"] == "right"
    assert c["severity"] == 1
    assert c["descriptor"] == "hairpin"
    assert abs(c["angle"] - 180) < 12


def test_a_wide_sweeping_bend_is_not_a_hairpin(lua, pn):
    # 150 degrees on a 60 m circle (the first bend of the Svolte di Popoli climb): a long number, not a name
    corners, _ = scan(lua, pn, road(("s", 300), ("r", 60, 150), ("s", 300)))
    assert len(corners) == 1
    c = corners[0]
    assert c["descriptor"] is None and c["severity"] == 4
    assert abs(c["angle"] - 150) < 12 and 50 < c["radius"] < 70


def test_the_hairpin_names_are_earned_by_the_radius(lua, pn):
    def descriptor(radius, degrees=170):
        corners, _ = scan(lua, pn, road(("s", 300), ("l", radius, degrees), ("s", 300)))
        assert len(corners) == 1
        return corners[0]["descriptor"]
    assert descriptor(6) == "tightHairpin"
    assert descriptor(12) == "hairpin"
    assert descriptor(21) == "openHairpin"
    assert descriptor(30) is None and descriptor(60) is None      # as turned, but a wide circle
    assert descriptor(12, 100) is None or descriptor(12, 100) == "square"     # a small circle, but not a U-turn


def test_the_radius_is_the_circle_the_game_fits(lua, pn):
    # the game's circle through the corner's two ends and its middle: the radius of the arc, on any of them
    for radius, degrees in ((15, 120), (40, 90), (90, 60), (150, 45)):
        corners, _ = scan(lua, pn, road(("s", 300), ("l", radius, degrees), ("s", 300)))
        assert abs(corners[0]["radius"] - radius) < 0.1 * radius, (radius, degrees, corners[0]["radius"])


def test_the_summary_names_the_corners_the_notebook_calls(lua, pn):
    pts = road(("s", 200), ("l", 25, 90), ("s", 100), ("r", 12, 180), ("s", 200))
    corners, _ = scan(lua, pn, pts)
    line = pn.summary(lua.table(*corners), 1000)
    assert line.startswith("2 corners called") and "1 by name" in line
    assert pn.summary(lua.table(), 1000) == "no corners"


def test_severity_follows_the_radius(lua, pn):
    def severity(radius, degrees):
        corners, _ = scan(lua, pn, road(("s", 300), ("l", radius, degrees), ("s", 300)))
        assert len(corners) == 1
        return corners[0]["severity"]
    order = [severity(10, 100), severity(25, 90), severity(45, 70), severity(80, 60), severity(120, 50), severity(150, 45)]
    assert order == sorted(order) and len(set(order)) == 6      # tighter is a lower number
    assert order[0] == 1 and order[-1] == 6


def test_a_corner_to_the_right_mirrors_one_to_the_left(lua, pn):
    for radius, degrees in ((25, 90), (60, 70), (120, 50)):
        left, _ = scan(lua, pn, road(("s", 200), ("l", radius, degrees), ("s", 200)))
        right, _ = scan(lua, pn, road(("s", 200), ("r", radius, degrees), ("s", 200)))
        assert left[0]["direction"] == "left" and right[0]["direction"] == "right"
        assert left[0]["severity"] == right[0]["severity"]
        assert abs(left[0]["s0"] - right[0]["s0"]) < 1e-6


def test_an_s_bend_is_two_corners_in_order_and_apart(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 200), ("l", 40, 60), ("r", 40, 60), ("s", 200)))
    assert [c["direction"] for c in corners] == ["left", "right"]
    assert corners[0]["s1"] + pn.MIN_GAP_M - 1e-6 <= corners[1]["s0"]


def test_two_bends_the_same_way_close_together_are_one_corner(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 200), ("l", 60, 40), ("s", 10), ("l", 60, 40), ("s", 200)))
    assert len(corners) == 1 and corners[0]["direction"] == "left"
    far, _ = scan(lua, pn, road(("s", 200), ("l", 60, 40), ("s", 100), ("l", 60, 40), ("s", 200)))
    assert len(far) == 2


def test_a_corner_that_tightens(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 200), ("r", 120, 30), ("r", 40, 60), ("s", 200)))
    assert len(corners) == 1
    assert corners[0]["shape"] == "tightens"
    corners, _ = scan(lua, pn, road(("s", 200), ("r", 40, 60), ("r", 120, 30), ("s", 200)))
    assert corners[0]["shape"] == "opens"


def test_nothing_before_the_start_or_after_the_end(lua, pn):
    # a corner right at the start line and one right at the stop: kept off the ends (the route builder drops what lies past)
    corners, t = scan(lua, pn, road(("l", 30, 60), ("s", 200), ("r", 30, 60)))
    total = sum(math.hypot(t[i + 1][1] - t[i][1], t[i + 1][2] - t[i][2]) for i in range(1, len(t)))
    for c in corners:
        assert c["s0"] >= pn.START_CLEAR_M - 1e-9 and c["s1"] <= total - pn.END_CLEAR_M + 1e-9


def test_notebook_is_the_stock_structure(lua, pn):
    pts = road(("s", 200), ("l", 25, 90), ("s", 100), ("r", 12, 180), ("s", 200))
    t = lua.table()
    for i, p in enumerate(pts, 1):
        t[i] = lua.table(*p)
    nb, corners = pn.notebook(t)
    assert nb["name"] == "structured" and nb["audioMode"] == 4 and nb["version"] == "4"
    notes = [nb["pacenotes"][i] for i in range(1, len(nb["pacenotes"]) + 1)]
    assert len(notes) == len(corners) == 2
    ids = []
    for i, note in enumerate(notes, 1):
        assert note["name"] == "Pacenote %d" % i
        wps = {note["pacenoteWaypoints"][k]["waypointType"]: note["pacenoteWaypoints"][k] for k in (1, 2)}
        cs, ce = wps["cornerStart"], wps["cornerEnd"]
        assert cs["name"] == "corner start" and ce["name"] == "corner end"
        for w in (cs, ce):
            assert w["radius"] == 8
            assert abs(math.sqrt(sum(w["normal"][k] ** 2 for k in (1, 2, 3))) - 1) < 1e-6
            ids.append(w["oldId"])
        ids.append(note["oldId"])
        assert note["structured"]["schemaVersion"] == 3
        assert note["structured"]["items"]["4"]["type"] == "corner"
        # corner start before corner end, along the road (heading east at first, so the first note: x grows)
    assert len(set(ids)) == len(ids)
    assert notes[0]["structured"]["items"]["4"]["descriptor"] is None
    assert notes[1]["structured"]["items"]["4"]["descriptor"] == "hairpin"
    # the waypoints lie on the road: heights come from the route's (1 m per 100 m)
    z0 = notes[0]["pacenoteWaypoints"][1]["pos"][3]
    assert z0 > 1.5


def lua_list(lua, rows):
    t = lua.table()
    for i, r in enumerate(rows, 1):
        t[i] = lua.table_from(r) if isinstance(r, dict) else lua.table(*r)
    return t


def notebook(lua, pn, pts, crests=(), dont_cut=(), line_length=None):
    opts = lua.table_from({"crests": lua_list(lua, crests), "dontCut": lua_list(lua, dont_cut)})
    if line_length:
        opts["lineLength"] = line_length
    nb, corners = pn.notebook(lua_list(lua, pts), opts)
    notes = [nb["pacenotes"][i] for i in range(1, len(nb["pacenotes"]) + 1)]
    return notes, corners


def items(note):
    it = note["structured"]["items"]
    return {k: it[k]["type"] for k in it}


def crest(s, lift_ms, grade_out=0.0, grade_in=0.08):
    return {"s": s, "liftMs": lift_ms, "curvature": 9.81 / lift_ms ** 2, "gradeIn": grade_in, "gradeOut": grade_out,
            "turn": grade_in - grade_out}


def test_the_speed_is_the_corners_reached_by_braking_and_accelerating(lua, pn):
    corners, _ = scan(lua, pn, road(("s", 600), ("r", 12, 180), ("s", 600)))
    t = lua.table()
    for i, c in enumerate(corners, 1):
        t[i] = c
    c = corners[0]
    v_corner = math.sqrt(pn.MU * pn.G * c["radius"])
    assert abs(pn.speedAt(t, (c["s0"] + c["s1"]) / 2) - v_corner) < 1e-6
    # 50 m before it: braking down to it; 50 m after: speeding up from it; far off: the top speed
    assert abs(pn.speedAt(t, c["s0"] - 50) - math.sqrt(v_corner ** 2 + 2 * pn.BRAKE_MS2 * 50)) < 1e-6
    assert abs(pn.speedAt(t, c["s1"] + 50) - math.sqrt(v_corner ** 2 + 2 * pn.ACCEL_MS2 * 50)) < 1e-6
    assert pn.speedAt(t, c["s1"] + 450) == pn.TOP_SPEED_MS
    # from a standstill at the start
    assert abs(pn.speedAt(lua.table(), 20) - math.sqrt(2 * pn.ACCEL_MS2 * 20)) < 1e-6


def test_a_crest_is_called_by_how_near_the_car_comes_to_leaving_the_ground(lua, pn):
    top = pn.TOP_SPEED_MS
    for u, word in ((1.1, "jump"), (0.8, "bigCrest"), (0.6, "crest"), (0.4, "smallCrest")):
        call, got = pn.crestCall(top / math.sqrt(u), top)
        assert call == word and abs(got - u) < 1e-9
    assert pn.crestCall(top / math.sqrt(0.2), top)[0] is None          # (a gentle brow: the car hardly feels it)


def test_a_brow_on_a_straight_is_a_note_of_its_own(lua, pn):
    pts = road(("s", 1500))
    lift = pn.TOP_SPEED_MS / math.sqrt(0.8)                             # at the top speed, 80 % of the car's weight off it
    notes, corners = notebook(lua, pn, pts, crests=[crest(800, lift)])
    assert len(notes) == 1
    assert items(notes[0]) == {"2": "bigCrest"}
    meta = notes[0]["metadata"]["earthrally"]["crest"]
    assert meta["atM"] == 800 and meta["liftKmh"] == round(lift * 3.6) and meta["speedKmh"] == round(pn.TOP_SPEED_MS * 3.6)
    wps = [notes[0]["pacenoteWaypoints"][k] for k in (1, 2)]
    assert [w["waypointType"] for w in wps] == ["cornerStart", "cornerEnd"]
    assert abs(wps[0]["pos"][1] - (800 - pn.CREST_NOTE_M / 2)) < 0.5 and abs(wps[1]["pos"][1] - (800 + pn.CREST_NOTE_M / 2)) < 0.5


def test_a_climb_that_only_eases_and_a_gentle_brow_are_not_called(lua, pn):
    pts = road(("s", 1500))
    lift = pn.TOP_SPEED_MS / math.sqrt(0.8)
    notes, _ = notebook(lua, pn, pts, crests=[crest(600, lift, grade_out=0.10), crest(1000, pn.TOP_SPEED_MS * 3)])
    assert notes == []
    # ... nor a kink of the elevation data: sharp enough for the car's speed, but it turns the road over by a percent or two
    notes, _ = notebook(lua, pn, pts, crests=[crest(800, lift, grade_in=0.01, grade_out=-0.01)])
    assert notes == []
    notes, _ = notebook(lua, pn, pts, crests=[crest(800, lift, grade_in=0.03, grade_out=-0.03)])
    assert len(notes) == 1


def test_a_crest_in_a_corner_is_over_crest_and_before_one_it_is_called_first(lua, pn):
    pts = road(("s", 600), ("l", 60, 60), ("s", 600))
    corners, _ = scan(lua, pn, pts)
    c = corners[0]
    v_in = math.sqrt(pn.MU * pn.G * c["radius"])
    mid = (c["s0"] + c["s1"]) / 2
    notes, _ = notebook(lua, pn, pts, crests=[crest(mid, v_in / math.sqrt(0.6))])
    assert items(notes[0]) == {"4": "corner", "5": "overCrest"}
    # 20 m before the corner, near flying at the speed it is braked from: "brake over crest" before the number
    before = c["s0"] - 20
    v = math.sqrt(v_in ** 2 + 2 * pn.BRAKE_MS2 * 20)
    notes, _ = notebook(lua, pn, pts, crests=[crest(before, v / math.sqrt(0.9))])
    assert items(notes[0]) == {"2": "brakeOverCrest", "4": "corner"}
    assert notes[0]["metadata"]["earthrally"]["crestBefore"]["call"] == "bigCrest"


def test_dont_cut_on_a_corner_over_a_bridge_or_along_sidewalks_and_not_elsewhere(lua, pn):
    pts = road(("s", 300), ("l", 25, 90), ("s", 400), ("r", 25, 90), ("s", 300))
    corners, _ = scan(lua, pn, pts)
    a, b = corners
    # spans measured on a line twice as long as this route (the curated line's metres): scaled to the route
    length = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))
    span = {"from": 2 * (a["s0"] + 2), "to": 2 * (a["s1"] - 2), "why": "sidewalk"}
    notes, _ = notebook(lua, pn, pts, dont_cut=[span], line_length=2 * length)
    assert items(notes[0]) == {"4": "corner", "5": "dontCut"} and notes[0]["metadata"]["earthrally"]["dontCut"] == "sidewalk"
    assert items(notes[1]) == {"4": "corner"}
    # with a crest over the same corner: "over crest", then "don't cut"
    v_in = math.sqrt(pn.MU * pn.G * a["radius"])
    notes, _ = notebook(lua, pn, pts, crests=[crest((a["s0"] + a["s1"]) / 2, v_in / math.sqrt(0.6))],
                        dont_cut=[{"from": a["s0"], "to": a["s1"], "why": "bridge"}])
    assert items(notes[0]) == {"4": "corner", "5": "overCrest", "6": "dontCut"}


def test_the_notes_stay_in_order_of_travel_with_a_crest_between_two_corners(lua, pn):
    pts = road(("s", 300), ("l", 25, 90), ("s", 700), ("r", 25, 90), ("s", 300))
    corners, _ = scan(lua, pn, pts)
    a, b = corners
    at = (a["s1"] + b["s0"]) / 2
    notes, got = notebook(lua, pn, pts, crests=[crest(at, pn.speedAt(got_table(lua, corners), at) / math.sqrt(0.6))])
    assert [items(n) for n in notes] == [{"4": "corner"}, {"2": "crest"}, {"4": "corner"}]
    assert [n["name"] for n in notes] == ["Pacenote 1", "Pacenote 2", "Pacenote 3"]
    assert pn.summary(got, length_of(pts)).endswith("1 crest, 0 don't cut")


def got_table(lua, rows):
    t = lua.table()
    for i, r in enumerate(rows, 1):
        t[i] = r
    return t


def length_of(pts):
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))
