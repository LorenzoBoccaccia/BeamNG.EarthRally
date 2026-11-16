"""The 1 km overview (tools/earth_view.py) and the eval's 3 x 2 picture (tools/earth_eval.py); no Earth tree needed.

    python -m pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import earth_eval as ee  # noqa: E402
import earth_view as ev  # noqa: E402

W, H = 300, 200


def block(n=1100, x0=-550.0, y0=-550.0, slope=0.01):
    """A height lattice n x n from (x0, y0), rising to the east."""
    z = np.tile(100.0 + slope * (x0 + np.arange(n)), (n, 1))
    return z, x0, y0


def test_the_overview_shows_1_km_round_the_point_north_up():
    cam = ev.overview_camera(0.0, 0.0, W, H)
    assert 2 * ev.OVERVIEW_M * cam.scale <= min(W, H)          # the whole square fits
    assert 2 * ev.OVERVIEW_M * cam.scale >= 0.95 * min(W, H)   # ... and fills the picture's height
    sx, sy, _ = cam.project(np.array([[0.0, 100.0, 0.0], [100.0, 0.0, 0.0]]))
    assert sy[0] < H / 2 and abs(sx[0] - W / 2) < 1e-6          # north is up
    assert sx[1] > W / 2                                        # east is right


def test_the_overview_heights_are_the_block_inside_the_square_and_none_outside():
    z, x0, y0 = block()
    cam = ev.overview_camera(0.0, 0.0, W, H)
    zp = ev.overview_heights(z, x0, y0, cam)
    assert zp.shape == (H, W)
    assert np.isnan(zp[H // 2, 5]) and np.isnan(zp[H // 2, W - 5])         # left and right of the square
    assert abs(zp[H // 2, W // 2] - 100.0) < 0.05                           # the point
    assert zp[H // 2, W // 2 + 50] > zp[H // 2, W // 2 - 50]                # rising to the east


def test_the_overview_off_the_block_has_no_ground():
    z, x0, y0 = block(n=700, x0=-350.0, y0=-350.0)
    cam = ev.overview_camera(0.0, 0.0, W, H)
    zp = ev.overview_heights(z, x0, y0, cam)
    xs = cam.target[0] + (np.arange(W) + 0.5 - W / 2.0) / cam.scale
    assert np.isnan(zp[H // 2, np.abs(xs) > 360]).all() and np.isfinite(zp[H // 2, np.abs(xs) < 340]).all()


def test_the_overview_draws_the_map_lines_by_level_and_the_close_up_box():
    z, x0, y0 = block(slope=0.0)
    cam = ev.overview_camera(0.0, 0.0, W, H)
    road = np.array([[-400.0, -200.0], [400.0, -200.0]])         # a bridge in its middle third
    rail = np.array([[-200.0, -400.0], [-200.0, 400.0]])
    lines = {"road": [(road, [(1 / 3, 2 / 3, "bridge")])], "rail": [(rail, [])]}
    img, zp = ev.overview_picture(z, x0, y0, cam, lines, (-40, -40, 40, 40), (0.0, 0.0))
    def px(x, y):
        sx, sy, _ = cam.project(np.array([x, y, 0.0]))
        return tuple(int(v) for v in (img[int(round(float(sy))), int(round(float(sx)))] * 255).round())

    def rgb(k):
        return tuple(int(v * 255) for v in ev.CLASS_RGB[k])
    assert px(-300, -200) == rgb(ev.ROAD)
    assert px(0, -200) == rgb(ev.DECK)
    assert px(-200, 300) == rgb(ev.RAIL)
    assert px(40, 0) == ev.OVERVIEW_BOX_RGB                                       # the close-up's box
    assert px(0, 0) == ev.OVERVIEW_BOX_RGB                                        # the point
    assert px(200, 200) != ev.OVERVIEW_BOX_RGB                                    # the ground


def test_the_overview_lines_stop_at_the_square():
    z, x0, y0 = block(slope=0.0)
    cam = ev.overview_camera(0.0, 0.0, W, H)
    road = np.array([[-2000.0, 0.0], [2000.0, 0.0]])
    img, _ = ev.overview_picture(z, x0, y0, cam, {"road": [(road, [])], "rail": []}, (-40, -40, 40, 40), (0.0, 0.0))
    assert np.allclose(img[H // 2, 3], ev.BACKGROUND, atol=1 / 255)


def test_the_after_overview_is_tinted_where_the_ground_moved():
    img = np.full((4, 4, 3), 128, np.uint8)
    zb = np.zeros((4, 4))
    za = zb.copy()
    za[0, 0], za[1, 1], za[2, 2], zb[3, 3] = 1.0, -1.0, 0.1, np.nan
    out = ee.overview_change(img, zb, za)
    assert out[0, 0, 0] > out[0, 0, 2]                  # rose: red
    assert out[1, 1, 2] > out[1, 1, 0]                  # fell: blue
    assert np.allclose(out[2, 2], 128 / 255) and np.allclose(out[3, 3], 128 / 255)     # less than MINOR_M; none before


def test_the_eval_picture_is_3_x_2_with_the_overviews(tmp_path):
    PW, PH = 120, 80
    cam = ev.overview_camera(0.0, 0.0, PW, PH)
    for side, dz in (("before", 0.0), ("after", 2.0)):
        d = tmp_path / "j" / side
        d.mkdir(parents=True)
        z = np.zeros((PH, PW), np.float32)
        z[40:50, :10] = dz
        np.savez_compressed(d / "view_overview.npz", img=np.full((PH, PW, 3), 128, np.uint8), z=z,
                            scale=np.float64(cam.scale), label=np.array(f"{side}: overview"))
        from PIL import Image
        for v in ("top", "from_SW"):
            Image.new("RGB", (PW, PH), (200, 200, 200)).save(d / f"view_panel_{v}.png")

    class A:
        before, after, out, image, views = "b", "a", str(tmp_path), f"{PW}x{PH}", "overview,top,SW"
    res = {"name": "j", **ee.compare(grids(), grids(deck_z=2.0))}
    res["_diff"] = np.ones((PH, PW, 3))
    path = ee.eval_picture({"name": "j", "what": "a test"}, A, res, lambda m: None)
    from PIL import Image
    pic = np.asarray(Image.open(path).convert("RGB"))
    assert pic.shape[1] == 3 * PW + 2 * ev.PANEL_GAP                           # three columns
    top = 62
    after_ov = pic[top + PH + ev.PANEL_GAP + 45, 5]                            # the after overview's corner: it rose there
    before_ov = pic[top + 45, 5]
    assert after_ov[0] > after_ov[2] and tuple(before_ov) == (128, 128, 128)
    assert pic[top + 2 * PH + ev.PANEL_GAP + 5, 0].tolist() == [255, 255, 255]  # two rows, then the legend
    assert (tmp_path / "j_diff.png").is_file()


def grids(deck_z=4.0):
    tris = np.concatenate([quad(-10, -10, 10, 10, 0.0), quad(-2, 5, 2, 8, deck_z)])
    return ev.surface_grid(tris, np.array([ev.GROUND] * 2 + [ev.DECK] * 2, np.int8), (-10, -10, 10, 10), step=0.5)


def quad(x0, y0, x1, y1, z):
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return np.array([[a, b, c], [a, c, d]], np.float64)
