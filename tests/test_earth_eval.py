"""tools/earth_eval.py and earth_view's surface grid: the before / after numbers, the diff picture, the junction set's arguments;
no Earth tree needed.

    python -m pytest tests
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import earth_eval as ee  # noqa: E402
import earth_view as ev  # noqa: E402


def quad(x0, y0, x1, y1, z):
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return np.array([[a, b, c], [a, c, d]], np.float64)


def scene(deck_z=4.0, road_z=0.0):
    tris = np.concatenate([quad(-10, -10, 10, 10, 0.0), quad(-2, -10, 2, 10, road_z), quad(-2, 5, 2, 8, deck_z)])
    return tris, np.array([ev.GROUND] * 2 + [ev.ROAD] * 2 + [ev.DECK] * 2, np.int8)


def test_the_surface_grid_sees_the_deck_over_the_road_over_the_ground_south_first():
    tris, cls = scene()
    c, z = ev.surface_grid(tris, cls, (-10, -10, 10, 10), step=0.5)
    assert c.shape == (40, 40)
    assert c[2, 2] == ev.GROUND and abs(z[2, 2]) < 1e-6           # (south west corner)
    assert c[2, 20] == ev.ROAD and abs(z[2, 20]) < 1e-6           # the road wins over the ground it lies on
    assert c[30, 20] == ev.DECK and abs(z[30, 20] - 4.0) < 1e-6   # y = 5.25: the deck, at its own height
    assert c[30, 2] == ev.GROUND


def test_nothing_drawn_is_class_minus_one_and_nan():
    c, z = ev.surface_grid(quad(0, 0, 5, 5, 1.0), np.array([ev.ROAD] * 2, np.int8), (-5, -5, 5, 5), step=1.0)
    assert c[0, 0] == -1 and np.isnan(z[0, 0]) and c[7, 7] == ev.ROAD


def grids(**kw):
    tris, cls = scene(**kw)
    return ev.surface_grid(tris, cls, (-10, -10, 10, 10), step=0.5)


def test_the_same_tree_is_the_same():
    r = ee.compare(grids(), grids())
    assert not r["changed"] and r["kind_changed"] == 0.0 and r["ground"]["max"] == 0.0 and r["deck"]["n"] > 0


def test_a_deck_lowered_is_changed_and_measured():
    r = ee.compare(grids(), grids(deck_z=2.5))
    assert r["changed"] and r["kind_changed"] == 0.0
    assert abs(r["deck"]["max"] - 1.5) < 1e-6 and r["road"]["max"] < 1e-6


def test_a_small_move_is_minor_not_changed():
    r = ee.compare(grids(), grids(deck_z=4.1))
    assert r["verdict"] == "minor" and not r["changed"]
    assert ee.compare(grids(), grids())["verdict"] == "same"


def test_a_deck_gone_changes_what_the_surface_is():
    tris, cls = scene()
    after = ev.surface_grid(tris[:4], cls[:4], (-10, -10, 10, 10), step=0.5)
    r = ee.compare(grids(), after)
    assert r["changed"] and r["kind_changed"] > 0.0


def test_the_diff_image_is_white_where_same_red_where_risen_grey_where_changed():
    b = grids()
    a = grids(deck_z=6.0)
    img = ee.diff_image(b, a, 80, 80, ee.SAME_M)
    assert img.shape == (80, 80, 3)
    assert (img[75, 5] == 1.0).all()                       # south west: the ground, the same (north up: last rows south)
    red = img[80 - 31 * 2, 40]                             # the deck (row 30 of the grid, south first, 2 px a sample)
    assert red[0] > 0.8 and red[1] < 0.6 and red[2] < 0.6
    tris, cls = scene()
    gone = ee.diff_image(b, ev.surface_grid(tris[:4], cls[:4], (-10, -10, 10, 10), step=0.5), 80, 80)
    assert np.allclose(gone[80 - 31 * 2, 40], ee.KIND_RGB)


def test_junction_args_anchor_point_and_lines():
    j = {"name": "p", "at": [-31.889125, 115.91902], "radius": 80}
    args = ee.junction_args(j)
    assert "--anchor=-31.89,115.92" in args and "--at=-31.889125,115.91902" in args
    assert args[args.index("--aid") + 1] == "9" and args[args.index("--radius") + 1] == "80.0"
    j = {"name": "q", "anchor": [42.17, 13.83], "aid": 2, "q": 0.015625, "xy": [-446, 880],
         "geometry": ["editions/svolte-di-popoli/geometry/ss1.json"]}
    args = ee.junction_args(j)
    assert "--xy=-446,880" in args and args[args.index("--q") + 1] == "0.015625"
    assert os.path.isfile(args[args.index("--geometry") + 1])


def test_the_set_is_well_formed():
    js = ee.load_set(ee.DEFAULT_SET)
    names = [j["name"] for j in js]
    assert len(names) == len(set(names)) >= 5
    for j in js:
        assert j.get("what") and ("at" in j or ("xy" in j and "anchor" in j))
        ee.junction_args(j)
    assert any(abs(j.get("at", [0, 0])[0] - 35.463744) < 1e-6 for j in js)       # (Yokohama)


def test_the_report_names_every_junction(tmp_path):
    class A:
        before, after, out = "b", "a", str(tmp_path)
    r1 = {"name": "x", **ee.compare(grids(), grids(deck_z=2.0))}
    r2 = {"name": "y", "error": "earth_view exited 1"}
    table = ee.write_report([r1, r2], A)
    assert "| x | **changed** |" in table and "| y | error" in table
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert [j["name"] for j in data["junctions"]] == ["x", "y"]
