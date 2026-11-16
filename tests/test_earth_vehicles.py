"""tools/earth_view.py's reference vehicles: trucks along the roads, wagons along the railways, a truck sized to a height limit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import earth_view as ev  # noqa: E402


def strip(x0, x1, y0, y1, z=0.0, step=1.0):
    """A flat strip of 1 m quads (two triangles each) from x0 to x1, y0 to y1."""
    out = []
    for x in np.arange(x0, x1, step):
        for y in np.arange(y0, y1, step):
            a, b, c, d = (x, y, z), (x + step, y, z), (x + step, y + step, z), (x, y + step, z)
            out += [(a, b, c), (a, c, d)]
    return np.array(out, np.float64)


def boxes(tris):
    return tris.reshape(-1, 12, 3, 3)


def test_a_box_stands_on_its_base_along_its_axis():
    t = ev.box_triangles((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 12.0, 2.5, 4.0)
    assert t.shape == (12, 3, 3)
    assert np.isclose(t[..., 2].min(), 1.0) and np.isclose(t[..., 2].max(), 5.0)
    assert np.isclose(np.ptp(t[..., 1]), 12.0) and np.isclose(np.ptp(t[..., 0]), 2.5)


def test_trucks_along_a_road_every_truck_spacing_in_the_middle_heading_along_it():
    road = strip(0, 100, -3.5, 3.5)
    t, c = ev.reference_vehicles(road, np.full(len(road), ev.ROAD, np.int8))
    b = boxes(t)
    assert (c == ev.TRUCK).all() and 4 <= len(b) <= 6
    mid = b.mean(axis=(1, 2))
    assert np.all(np.abs(mid[:, 1]) < 0.5)                                   # in the middle of the road
    assert np.all(np.isclose(np.ptp(b[..., 0], axis=(1, 2)), ev.TRUCK_LWH[0], atol=0.3))     # along x
    xs = np.sort(mid[:, 0])
    assert np.all(np.diff(xs) >= ev.TRUCK_EVERY_M * 0.75)
    assert np.allclose(b[..., 2].max(axis=(1, 2)), ev.TRUCK_LWH[2])


def test_a_road_on_two_levels_gets_trucks_on_both():
    road = np.concatenate([strip(0, 40, -3.5, 3.5, 0.0), strip(14, 21, -20, 20, 7.0)[:, :, [1, 0, 2]] * (1, 1, 1)])
    cls = np.full(len(road), ev.ROAD, np.int8)
    t, c = ev.reference_vehicles(road, cls)
    bottoms = boxes(t)[..., 2].min(axis=(1, 2))
    assert (np.isclose(bottoms, 0.0)).any() and (np.isclose(bottoms, 7.0)).any()


def test_wagons_on_the_rails_heads_and_none_on_a_rail_deck():
    rail = strip(0, 60, -1.5, 1.5, 0.5)
    deck = strip(0, 60, -3.0, 3.0, 0.0)
    tris = np.concatenate([rail, deck])
    cls = np.array([ev.RAIL] * len(rail) + [ev.DECK] * len(deck), np.int8)
    t, c = ev.reference_vehicles(tris, cls)
    assert (c == ev.WAGON).all() and len(boxes(t)) >= 2                      # (no truck on the track's deck)
    assert np.allclose(boxes(t)[..., 2].min(axis=(1, 2)), 0.5)


def test_walls_do_not_turn_a_wagon_across_its_track():
    rail = strip(0, 60, -1.5, 1.5, 0.5)
    wall = np.array([[(30, -6, 0), (30, 6, 0), (30, 6, 5)], [(30, -6, 0), (30, 6, 5), (30, -6, 5)]] * 20, np.float64)
    tris = np.concatenate([rail, wall])
    cls = np.full(len(tris), ev.RAIL, np.int8)
    b = boxes(ev.reference_vehicles(tris, cls)[0])
    assert np.all(np.ptp(b[..., 0], axis=(1, 2)) > np.ptp(b[..., 1], axis=(1, 2)))


def test_a_signed_road_gets_a_vehicle_as_high_as_the_sign_and_a_pickup_under_2_5_m():
    road = strip(0, 30, -3.5, 3.5)
    cls = np.full(len(road), ev.ROAD, np.int8)
    line = np.array([[-5.0, 0.0], [40.0, 0.0]])
    for h, length in ((3.3, ev.TRUCK_LWH[0]), (1.9, ev.PICKUP_LWH[0])):
        b = boxes(ev.reference_vehicles(road, cls, [(line, h)])[0])
        assert np.allclose(b[..., 2].max(axis=(1, 2)), h)
        assert np.allclose(np.ptp(b[..., 0], axis=(1, 2)), length, atol=0.3)
    far = np.array([[-5.0, 50.0], [40.0, 50.0]])
    b = boxes(ev.reference_vehicles(road, cls, [(far, 1.9)])[0])
    assert np.allclose(b[..., 2].max(axis=(1, 2)), ev.TRUCK_LWH[2])         # (a sign elsewhere)


def test_limit_at_takes_the_lowest_near():
    lines = [(np.array([[0.0, 0.0], [10.0, 0.0]]), 3.3), (np.array([[0.0, 2.0], [10.0, 2.0]]), 2.5),
             (np.array([[0.0, 50.0], [10.0, 50.0]]), 1.0)]
    assert ev.limit_at(lines, (5.0, 1.0)) == 2.5 and ev.limit_at(lines, (5.0, 20.0)) is None
