#!/usr/bin/env python3
"""A 3D debug viewer for BeamNG.Earth's terrain and meshes around a point: shaded pictures in debug colours.

For ground and structure problems (a road under the ground, a deck without its approach, a tunnel hole with no tube, a building
on a road): Earth's own code builds, offline and unchanged (the Earth tree is only imported, never modified), the terrain block
that holds the point and the anchor's road tiles that cover the radius, and the tool draws what they hold, every surface lit by
one directional (Lambert) light and a little ambient light, in flat colours:

  brown       the ground (the block's 1 m height lattice, cropped to the radius; the .ter's tunnel holes are left open)
  magenta     ground the EarthRally grader moves (--rally; the picture then shows the graded ground)
  pink        roads (the tile's road mesh); a deeper pink: bridge decks (its deck mesh)
  yellow      railways (rail mesh: ballast, sleepers, rails)
  white       buildings
  blue-grey   tunnel tubes and portals
  green       sidewalks;  orange: median barriers   (--kinds adds the rest: signs, wires, the collision meshes)
  dark line   the stage line (--geometry, repeatable); blue needle: the point

The meshes are the ones Earth writes for the game: each tile's .cdae files are read back (Earth's cdae.read), so what is drawn is what
the game would load, closed solids included. A picture is an orthographic camera, rasterised by a numpy z-buffer (headless, no GPU):

  <name>_top.png            straight down, north up
  <name>_from_<dir>.png     oblique (--elev degrees above the horizon), the camera on the compass side <dir> looking at the point
                            (--views top,SW,SE,N by default: any of N NE E SE S SW W NW or a bearing in degrees)

Each has a title (the point, the radius, the Earth tree), a legend of the colours in it, a scale bar and a north arrow.

  python tools/earth_view.py --earth ../BeamNG.Earth-fork --anchor 42.17,13.83 --aid 2 --xy -446,880 \
      --geometry editions/svolte-di-popoli/geometry/ss1.json --radius 120 --out view-out
  python tools/earth_view.py --earth ../BeamNG.Earth-fork --anchor 42.17,13.83 --aid 2 --at 42.1743502,13.8243183 --out view-out --rally \
      --geometry editions/svolte-di-popoli/geometry/ss1.json

A block takes minutes to build (a 1024 m square, as the server makes it), a road tile ~15 s: --reuse draws again from the data a
former run left in <out>/data (a new camera, radius, kinds) without building. Left out: the props (trees, poles, signs), the wires,
a geotile's far meshes, the game's runtime colliders; a building is the tile's own mesh, as the game draws it near the car.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HM_ZOOM = 16
BLOCK_STEP = 128            # a block's centre is a multiple of this (m)
TILE = 256.0                # roads.TILE: an anchor road tile's side (m)

# ---- classes: what a triangle is, its debug colour, its name in the legend ---------------------------------------------------
GROUND, MOVED, ROAD, DECK, RAIL, BUILDING, TUNNEL, WALK, BARRIER, OTHER = range(10)
CLASS_NAME = ["ground", "ground moved by the rally grader", "road", "bridge deck", "rail", "building", "tunnel tube / portal",
              "sidewalk", "median barrier", "other mesh (--kinds)"]
CLASS_RGB = np.array([
    (0.55, 0.38, 0.22),     # ground: brown
    (1.00, 0.00, 1.00),     # moved by the grader: magenta
    (1.00, 0.45, 0.72),     # road: pink
    (0.90, 0.22, 0.52),     # deck: deeper pink
    (1.00, 0.85, 0.10),     # rail: yellow
    (1.00, 1.00, 1.00),     # building: white
    (0.42, 0.62, 0.80),     # tunnel: blue-grey
    (0.50, 0.80, 0.42),     # sidewalk: green
    (1.00, 0.55, 0.10),     # barrier: orange
    (0.70, 0.55, 0.90),     # anything else: lilac
])
KIND_CLASS = {"road": ROAD, "deck": DECK, "rail": RAIL, "bld": BUILDING, "b3dmr": BUILDING, "tunnel": TUNNEL, "walk": WALK,
              "barrier": BARRIER}
DEFAULT_KINDS = ("road", "deck", "tunnel", "rail", "bld", "b3dmr", "walk", "barrier")
BACKGROUND = (0.84, 0.87, 0.91)
LINE_RGB = (0.05, 0.05, 0.08)
NEEDLE_RGB = (0.0, 0.25, 1.0)
AMBIENT = 0.30
DEPTH_BIAS_M = 0.03         # m: everything but the ground is drawn this much nearer (the road mesh lies on the ground it follows)
COMPASS = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}


# ======================================================================================================================
# pure parts: the ground mesh, the camera, the rasteriser, the shading
# ======================================================================================================================
def terrain_mesh(z, x0, y0, box, holes=None, moved=None):
    """The ground triangles of the cells of the height lattice z (row j is y0 + j, column i is x0 + i, 1 m apart) inside box
    (xmin, ymin, xmax, ymax). Returns (tris (n, 3, 3) world vertices, cls (n,) GROUND or MOVED). A cell with a hole (holes[j, i], the
    .ter's tunnel mouths) has no triangles; a cell with a moved corner (moved[j, i] bool) is MOVED."""
    ny, nx = z.shape
    i0 = int(max(0, math.floor(box[0] - x0)))
    i1 = int(min(nx - 1, math.ceil(box[2] - x0)))
    j0 = int(max(0, math.floor(box[1] - y0)))
    j1 = int(min(ny - 1, math.ceil(box[3] - y0)))
    if i1 <= i0 or j1 <= j0:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int8)
    zz = z[j0:j1 + 1, i0:i1 + 1].astype(np.float64)
    h, w = zz.shape
    X = x0 + i0 + np.arange(w, dtype=np.float64)
    Y = y0 + j0 + np.arange(h, dtype=np.float64)
    V = np.stack(np.broadcast_arrays(X[None, :], Y[:, None], zz), axis=-1)        # (h, w, 3)
    a, b, c, d = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]                   # the cell's corners, counter-clockwise from below
    keep = np.isfinite(a[..., 2]) & np.isfinite(b[..., 2]) & np.isfinite(c[..., 2]) & np.isfinite(d[..., 2])
    if holes is not None:
        keep &= ~holes[j0:j1, i0:i1]
    cell_cls = np.full(keep.shape, GROUND, np.int8)
    if moved is not None:
        m = moved[j0:j1 + 1, i0:i1 + 1]
        cell_cls[m[:-1, :-1] | m[:-1, 1:] | m[1:, 1:] | m[1:, :-1]] = MOVED
    tris = np.concatenate([np.stack([a[keep], b[keep], c[keep]], axis=1), np.stack([a[keep], c[keep], d[keep]], axis=1)])
    cls = np.concatenate([cell_cls[keep], cell_cls[keep]])
    return tris, cls


def face_normals(tris):
    """Unit normals of triangles (n, 3, 3); (0, 0, 1) for a degenerate one."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n[ok] /= ln[ok, None]
    n[~ok] = (0.0, 0.0, 1.0)
    return n


class Camera:
    """An orthographic camera looking along fwd at target; scale pixels per metre, the image W x H with target at its centre
    (shifted by the offset (m along right, m along up))."""

    def __init__(self, target, fwd, W, H, scale=1.0, offset=(0.0, 0.0)):
        f = np.asarray(fwd, np.float64)
        self.fwd = f / np.linalg.norm(f)
        hint = np.array([0.0, 0.0, 1.0]) if abs(self.fwd[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
        r = np.cross(self.fwd, hint)
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)
        self.target = np.asarray(target, np.float64)
        self.W, self.H, self.scale = int(W), int(H), float(scale)
        self.offset = (float(offset[0]), float(offset[1]))

    def project(self, P):
        """World points (..., 3) -> (sx, sy pixels with y down, depth along the view: larger is farther)."""
        d = np.asarray(P, np.float64) - self.target
        sx = self.W / 2.0 + (d @ self.right - self.offset[0]) * self.scale
        sy = self.H / 2.0 - (d @ self.up - self.offset[1]) * self.scale
        return sx, sy, d @ self.fwd

    def light(self):
        """The direction towards the light: from over the camera's left shoulder and above (so relief shows in every view)."""
        L = -0.55 * self.fwd - 0.45 * self.right + 0.45 * self.up
        return L / np.linalg.norm(L)


def view_camera(target, bearing_deg, elev_deg, W, H, scale=1.0, offset=(0.0, 0.0)):
    """The camera standing on the compass side bearing_deg of target (0 = north; looking back at it), elev_deg above the horizon
    (90: straight down, north up)."""
    el = math.radians(elev_deg)
    if elev_deg >= 89.9:
        fwd = (0.0, 0.0, -1.0)
    else:
        b = math.radians(bearing_deg)
        fwd = (-math.sin(b) * math.cos(el), -math.cos(b) * math.cos(el), -math.sin(el))
    return Camera(target, fwd, W, H, scale, offset)


def fit_camera(cam, points):
    """Scale and offset of cam that frame the world points (n, 3) with a 4 % margin in a W x H image."""
    d = np.asarray(points, np.float64) - cam.target
    u, v = d @ cam.right, d @ cam.up
    wm, hm = max(u.max() - u.min(), 1e-6), max(v.max() - v.min(), 1e-6)
    cam.scale = min(cam.W / (wm * 1.08), cam.H / (hm * 1.08))
    cam.offset = ((u.max() + u.min()) / 2.0, (v.max() + v.min()) / 2.0)
    return cam


def shade(normals, cam, ambient=AMBIENT):
    """Lambert shade (ambient .. 1) of triangles with the given normals, each normal turned to face the camera (two-sided)."""
    n = normals * np.where(normals @ cam.fwd > 0, -1.0, 1.0)[:, None]
    return ambient + (1.0 - ambient) * np.clip(n @ cam.light(), 0.0, 1.0)


def _split_long(scr, dep, ids, W, H, kmax, rounds=40):
    """Triangles (scr (n, 3, 2) pixels, dep (n, 3)) with no side of their pixel box over kmax: the long ones cut at the middle of
    their longest edge again and again; pieces outside the image are dropped."""
    done = []
    for _ in range(rounds):
        if not len(scr):
            break
        x, y = scr[..., 0], scr[..., 1]
        inside = (x.max(1) >= 0) & (x.min(1) < W) & (y.max(1) >= 0) & (y.min(1) < H)
        scr, dep, ids = scr[inside], dep[inside], ids[inside]
        if not len(scr):
            break
        x, y = scr[..., 0], scr[..., 1]
        big = np.maximum(x.max(1) - x.min(1), y.max(1) - y.min(1)) > kmax
        if (~big).any():
            done.append((scr[~big], dep[~big], ids[~big]))
        scr, dep, ids = scr[big], dep[big], ids[big]
        if not len(scr):
            break
        e = np.stack([np.hypot(*(scr[:, (k + 1) % 3] - scr[:, k]).T) for k in range(3)], axis=1)
        k = e.argmax(axis=1)
        r = np.arange(len(scr))
        i1, i2 = (k + 1) % 3, (k + 2) % 3
        p0, p1, p2 = scr[r, k], scr[r, i1], scr[r, i2]
        d0, d1, d2 = dep[r, k], dep[r, i1], dep[r, i2]
        pm, dm = (p0 + p1) / 2.0, (d0 + d1) / 2.0
        scr = np.concatenate([np.stack([p0, pm, p2], 1), np.stack([pm, p1, p2], 1)])
        dep = np.concatenate([np.stack([d0, dm, d2], 1), np.stack([dm, d1, d2], 1)])
        ids = np.concatenate([ids, ids])
    if not done:
        return np.zeros((0, 3, 2)), np.zeros((0, 3)), np.zeros(0, np.int64)
    return (np.concatenate([d[0] for d in done]), np.concatenate([d[1] for d in done]), np.concatenate([d[2] for d in done]))


def rasterize(scr, dep, W, H, kmax=32, budget=2_000_000):
    """Z-buffer of triangles: scr (n, 3, 2) pixel coordinates (x right, y down; a pixel's centre is at +0.5), dep (n, 3) depth
    (smaller is nearer; linear in the screen, an orthographic camera's). Returns (tri (H, W) int64 index of the nearest triangle or -1,
    depth (H, W) float64, inf where none). Batched in numpy: the triangles are bucketed by the size of their pixel box (cut when
    large), every batch tests all the pixel centres of its boxes at once and the nearest fragment per pixel is a minimum over a
    packed (depth, triangle) key."""
    scr = np.asarray(scr, np.float64)
    dep = np.asarray(dep, np.float64)
    n = len(scr)
    none = np.full(W * H, np.iinfo(np.uint64).max, np.uint64)
    if n == 0:
        return np.full((H, W), -1, np.int64), np.full((H, W), np.inf)
    ok = np.isfinite(scr).all(axis=(1, 2)) & np.isfinite(dep).all(axis=1)
    ids = np.arange(n, dtype=np.int64)[ok]
    scr, dep = scr[ok], dep[ok]
    if not len(scr):
        return np.full((H, W), -1, np.int64), np.full((H, W), np.inf)
    dmin, dmax = float(dep.min()), float(dep.max())
    q = (2.0 ** 32 - 2.0) / max(dmax - dmin, 1e-9)
    scr, dep, ids = _split_long(scr, dep, ids, W, H, kmax)
    if len(scr):
        x, y = scr[..., 0], scr[..., 1]
        x0 = np.maximum(np.ceil(x.min(1) - 0.5), 0).astype(np.int64)
        x1 = np.minimum(np.floor(x.max(1) - 0.5), W - 1).astype(np.int64)
        y0 = np.maximum(np.ceil(y.min(1) - 0.5), 0).astype(np.int64)
        y1 = np.minimum(np.floor(y.max(1) - 0.5), H - 1).astype(np.int64)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        live = (w > 0) & (h > 0)
        scr, dep, ids = scr[live], dep[live], ids[live]
        x0, y0, x1, y1, w, h = x0[live], y0[live], x1[live], y1[live], w[live], h[live]
        kw = (1 << np.ceil(np.log2(np.maximum(w, 1))).astype(np.int64)).astype(np.int64)
        kh = (1 << np.ceil(np.log2(np.maximum(h, 1))).astype(np.int64)).astype(np.int64)
        for a_, b_ in set(zip(kw.tolist(), kh.tolist())):
            sel = np.nonzero((kw == a_) & (kh == b_))[0]
            step = max(1, budget // (a_ * b_))
            for s in range(0, len(sel), step):
                t = sel[s:s + step]
                v, dz, ti = scr[t], dep[t], ids[t]
                px = x0[t][:, None, None] + np.arange(a_)[None, None, :]
                py = y0[t][:, None, None] + np.arange(b_)[None, :, None]
                cx, cy = px + 0.5, py + 0.5
                ax, ay, bx, by, qx, qy = (v[:, 0, 0, None, None], v[:, 0, 1, None, None], v[:, 1, 0, None, None],
                                          v[:, 1, 1, None, None], v[:, 2, 0, None, None], v[:, 2, 1, None, None])
                area = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
                good = np.abs(area) > 1e-12
                area = np.where(good, area, 1.0)
                wa = ((qx - bx) * (cy - by) - (qy - by) * (cx - bx)) / area
                wb = ((ax - qx) * (cy - qy) - (ay - qy) * (cx - qx)) / area
                wc = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / area
                inside = good & (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9) & (px <= x1[t][:, None, None]) & \
                    (py <= y1[t][:, None, None])
                if not inside.any():
                    continue
                z = (wa * dz[:, 0, None, None] + wb * dz[:, 1, None, None] + wc * dz[:, 2, None, None])[inside]
                tri = np.broadcast_to(ti[:, None, None], inside.shape)[inside].astype(np.uint64)
                pix = (py * W + px)[inside]
                key = (np.clip((z - dmin) * q, 0, 2.0 ** 32 - 2.0).astype(np.uint64) << np.uint64(32)) | tri
                np.minimum.at(none, pix, key)
    hit = none != np.iinfo(np.uint64).max
    tri = np.where(hit, (none & np.uint64(0xFFFFFFFF)).astype(np.int64), -1).reshape(H, W)
    depth = np.where(hit, (none >> np.uint64(32)).astype(np.float64) / q + dmin, np.inf).reshape(H, W)
    return tri, depth


def sample_polyline(P, step=0.5):
    """Points every step m along the 3D polyline P (n, 3), its vertices included."""
    P = np.asarray(P, np.float64)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.unique(np.concatenate([np.arange(0.0, s[-1], step), s]))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def splat(img, sx, sy, rgb, radius=1, alpha=None):
    """Draw pixels at (sx, sy) in img (H, W, 3): a square of half-size radius, alpha (per point, or a number) over what is there."""
    H, W = img.shape[:2]
    rgb = np.asarray(rgb, np.float64)
    al = np.ones(len(sx)) if alpha is None else np.broadcast_to(np.asarray(alpha, np.float64), (len(sx),))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = np.floor(sx).astype(np.int64) + dx
            y = np.floor(sy).astype(np.int64) + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[ok], x[ok]] = img[y[ok], x[ok]] * (1.0 - al[ok, None]) + rgb * al[ok, None]


def render(tris, cls, cam, lines=(), needle=None, ss=2, bias=DEPTH_BIAS_M, line_tol=0.6):
    """The picture (H, W, 3 floats 0..1) of triangles tris (n, 3, 3) with classes cls (n,) seen by cam, the polylines (3D point
    arrays) drawn over it as a thin dark line (full where nothing hides it by more than line_tol m, faint where it is behind
    something), needle = (x, y, z0, z1) a blue vertical marker. Rendered ss times larger and averaged down. Returns (image, used
    classes: sorted unique class numbers drawn)."""
    cs = Camera(cam.target, cam.fwd, cam.W * ss, cam.H * ss, cam.scale * ss, cam.offset)
    cls = np.asarray(cls)
    img = np.empty((cs.H, cs.W, 3))
    img[:] = BACKGROUND
    used = np.zeros(0, np.int64)
    zbuf = np.full((cs.H, cs.W), np.inf)
    if len(tris):
        sx, sy, d = cs.project(tris)
        scr = np.stack([sx, sy], axis=-1)
        d = d - np.where(cls <= MOVED, 0.0, bias)[:, None]
        tri, zbuf = rasterize(scr, d, cs.W, cs.H)
        col = CLASS_RGB[cls] * shade(face_normals(tris), cs)[:, None]
        hit = tri >= 0
        img[hit] = col[tri[hit]]
        used = np.unique(cls[np.unique(tri[hit])]) if hit.any() else used
    for P in lines:
        S = sample_polyline(P, 0.5 / cs.scale)
        sx, sy, d = cs.project(S)
        xi, yi = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        ok = (xi >= 0) & (xi < cs.W) & (yi >= 0) & (yi < cs.H)
        xi, yi, sx, sy, d = xi[ok], yi[ok], sx[ok], sy[ok], d[ok]
        seen = d <= zbuf[yi, xi] + line_tol
        splat(img, sx, sy, LINE_RGB, radius=max(1, ss // 2), alpha=np.where(seen, 1.0, 0.3))
    if needle is not None:
        x, y, z0, z1 = needle
        S = sample_polyline(np.array([[x, y, z0], [x, y, z1]]), 0.25)
        sx, sy, _ = cs.project(S)
        splat(img, sx, sy, NEEDLE_RGB, radius=max(1, ss))
        cx, cy, _ = cs.project(np.array([[x, y, z0]]))
        t = np.linspace(0, 2 * np.pi, 90)
        splat(img, cx[0] + 4.0 * ss * np.cos(t), cy[0] + 4.0 * ss * np.sin(t), NEEDLE_RGB, radius=max(1, ss // 2))
    out = img.reshape(cam.H, ss, cam.W, ss, 3).mean(axis=(1, 3)) if ss > 1 else img
    return out, used


# ======================================================================================================================
# the tiles' meshes, the picture's frame (title, legend, scale bar, north arrow)
# ======================================================================================================================
def kind_of(key):
    """The mesh kind of a tile JSON field (bld2 -> bld, railcol3 -> railcol)."""
    return re.sub(r"\d+$", "", key)


def tile_entries(data):
    """[(kind, shape file name, pos)] of the meshes a road tile's JSON names ({"shape": "/vfs/x.cdae", "pos": [x, y, z]} fields, and
    b3dmr's list)."""
    out = []
    for key, v in data.items():
        items = v if isinstance(v, list) else [v]
        for e in items:
            if isinstance(e, dict) and isinstance(e.get("shape"), str) and e.get("pos") is not None \
                    and e["shape"].lower().endswith(".cdae"):
                out.append((kind_of(key), e["shape"].replace("\\", "/").rsplit("/", 1)[-1], [float(c) for c in e["pos"]]))
    return out


def shape_triangles(shape, pos):
    """Triangles (n, 3, 3) of a decoded cdae shape (cdae.read's dict) in world coordinates: every standard mesh's vertices at pos."""
    import cdae
    out = []
    for m in shape["meshes"]:
        if m.get("type") != 0:
            continue
        a = cdae.mesh_arrays(m)
        idx = a["indices"].astype(np.int64)
        idx = idx[:len(idx) // 3 * 3].reshape(-1, 3)
        if not len(idx) or not len(a["verts"]):
            continue
        out.append(a["verts"].astype(np.float64)[idx] + np.asarray(pos, np.float64))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def crop_triangles(tris, box):
    """The triangles whose x/y box meets box (xmin, ymin, xmax, ymax)."""
    if not len(tris):
        return tris
    x, y = tris[..., 0], tris[..., 1]
    k = (x.max(1) >= box[0]) & (x.min(1) <= box[2]) & (y.max(1) >= box[1]) & (y.min(1) <= box[3])
    return tris[k]


def parse_bearing(s):
    s = s.strip().upper()
    if s in COMPASS:
        return COMPASS[s]
    return float(s) % 360.0


def view_name(s):
    return "top" if s.strip().lower() == "top" else f"from_{s.strip().upper()}"


def _font(size):
    from PIL import ImageFont
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def frame_picture(img, cam, title, subtitle, used, has_line, has_needle, extra_legend=()):
    """The map image (H, W, 3 floats) with a title bar above, the scale bar and north arrow on it, the legend below; a PIL image."""
    from PIL import Image, ImageDraw
    pic = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(pic)
    W, H = pic.size
    f = _font(15)
    # scale bar: along the picture's horizontal, whatever the camera (orthographic: the same everywhere in it)
    for L in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if L * cam.scale > W * 0.12:
            break
    x0, y0 = 20, H - 22
    d.rectangle([x0 - 6, y0 - 22, x0 + L * cam.scale + 6, y0 + 10], fill=(255, 255, 255))
    d.line([x0, y0, x0 + L * cam.scale, y0], fill=(0, 0, 0), width=3)
    d.line([x0, y0 - 5, x0, y0 + 5], fill=(0, 0, 0), width=2)
    d.line([x0 + L * cam.scale, y0 - 5, x0 + L * cam.scale, y0 + 5], fill=(0, 0, 0), width=2)
    d.text((x0, y0 - 20), f"{L} m (across the picture)", fill=(0, 0, 0), font=f)
    # north arrow
    nx, ny = float(np.array([0, 1, 0]) @ cam.right), float(np.array([0, 1, 0]) @ cam.up)
    ln = math.hypot(nx, ny) or 1.0
    cx, cy = W - 50, 50
    d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], fill=(255, 255, 255), outline=(90, 90, 90))
    ux, uy = nx / ln, -ny / ln
    reach = 26 * max(ln, 0.35)
    d.line([cx - ux * 6, cy - uy * 6, cx + ux * reach, cy + uy * reach], fill=(200, 0, 0), width=4)
    d.text((cx + ux * (reach + 6) - 5, cy + uy * (reach + 6) - 9), "N", fill=(200, 0, 0), font=f)
    # legend
    entries = [(tuple(int(c * 255) for c in CLASS_RGB[c]), CLASS_NAME[c]) for c in sorted(set(int(u) for u in used))]
    if has_line:
        entries.append((tuple(int(c * 255) for c in LINE_RGB), "stage line (--geometry): dark; faint where hidden"))
    if has_needle:
        entries.append((tuple(int(c * 255) for c in NEEDLE_RGB), "the point (needle up from the ground, ring on it)"))
    entries += list(extra_legend)
    cols = 3
    rows = (len(entries) + cols - 1) // cols
    row_h, tf = 24, _font(18)
    head = 62
    canvas = Image.new("RGB", (W, head + H + 12 + rows * row_h + 10), (255, 255, 255))
    canvas.paste(pic, (0, head))
    cd = ImageDraw.Draw(canvas)
    cd.text((12, 6), title, fill=(0, 0, 0), font=tf)
    cd.text((12, 34), subtitle, fill=(60, 60, 60), font=f)
    for k, (rgb, name) in enumerate(entries):
        cx_ = 12 + (k % cols) * (W // cols)
        cy_ = head + H + 12 + (k // cols) * row_h
        cd.rectangle([cx_, cy_ + 2, cx_ + 28, cy_ + 18], fill=rgb, outline=(0, 0, 0))
        cd.text((cx_ + 36, cy_ + 1), name, fill=(0, 0, 0), font=f)
    return canvas


# ======================================================================================================================
# Earth: the block and the tiles
# ======================================================================================================================
def parse_pair(s):
    a, b = s.split(",")
    return float(a), float(b)


def block_centre(x, y):
    """The block centre (a multiple of BLOCK_STEP) nearest the point: the point lies near the middle of its 1024 m block."""
    return float(round(x / BLOCK_STEP) * BLOCK_STEP), float(round(y / BLOCK_STEP) * BLOCK_STEP)


def tiles_covering(x, y, radius):
    """(tx, ty) of the anchor road tiles meeting the square of half-side radius around (x, y)."""
    return [(tx, ty) for tx in range(int(math.floor((x - radius) / TILE)), int(math.floor((x + radius) / TILE)) + 1)
            for ty in range(int(math.floor((y - radius) / TILE)), int(math.floor((y + radius) / TILE)) + 1)]


def earth_version(earth):
    try:
        with open(os.path.join(earth, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "?"


def load_block_files(data_dir, aid, cx, cy):
    """(heights (n, n) by row = y, holes (n, n) bool or None, info dict) of a block's files in data_dir."""
    from PIL import Image
    info_path = os.path.join(data_dir, f"hm_{aid}_{int(cx)}_{int(cy)}.info.json")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    png = np.asarray(Image.open(os.path.join(data_dir, f"hm_{aid}_{int(cx)}_{int(cy)}.png")), np.float64)[::-1]
    z = png * float(info["q"]) + float(info.get("zBase") or 0.0)
    holes = None
    terp = os.path.join(data_dir, f"hm_{aid}_{int(cx)}_{int(cy)}_view.ter")
    if os.path.isfile(terp):
        with open(terp, "rb") as f:
            raw = f.read()
        n = int.from_bytes(raw[1:5], "little")
        lay = np.frombuffer(raw, np.uint8, n * n, 5 + 2 * n * n).reshape(n, n)
        holes = lay == 255
        holes[0, :] = holes[-1, :] = holes[:, 0] = holes[:, -1] = False     # (the edge ring: always holes)
    return z, holes, info


def bilinear(z, x0, y0, x, y):
    fx, fy = np.asarray(x, np.float64) - x0, np.asarray(y, np.float64) - y0
    i = np.clip(np.floor(fx).astype(int), 0, z.shape[1] - 2)
    j = np.clip(np.floor(fy).astype(int), 0, z.shape[0] - 2)
    u, v = np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1)
    return z[j, i] * (1 - u) * (1 - v) + z[j, i + 1] * u * (1 - v) + z[j + 1, i] * (1 - u) * v + z[j + 1, i + 1] * u * v


def earth_state(a, log):
    """Earth's worker state (what blockworker builds from: the disk cache, the map data), set up once; the quantum a.q is read from
    the server log when not given."""
    import blockworker
    import earth_writers
    if a.q is None:
        a.q = earth_writers.log_quantum(a.server_log, a.aid) or 1 / 32
        log(f"height quantum {a.q:g} m (the server log's)")
    if getattr(blockworker, "_s", None) is None:
        blockworker._s = blockworker._State(a.cache, not a.online, True, True, True, True, airports=not a.no_airports,
                                            sea=not a.no_sea)
        blockworker._s.anchor(a.aid, *a.anchor)
    return blockworker


def build_earth(a, frame, px, py, log):
    """Build (or with --reuse read) the block and the road tiles of the point with Earth's own code. Returns (heights, holes, info,
    x0, y0, block (cx, cy), [(tx, ty)], tile data dir)."""
    data_dir = os.path.join(a.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    cx, cy = a.block if a.block else block_centre(px, py)
    tiles = tiles_covering(px, py, a.radius)
    stem = f"hm_{a.aid}_{int(cx)}_{int(cy)}"
    if not (a.reuse and os.path.isfile(os.path.join(data_dir, stem + ".info.json"))):
        blockworker = earth_state(a, log)
        lat0, lon0 = a.anchor
        log(f"building block ({cx:g}, {cy:g}), {a.block_size} samples: this takes minutes ...")
        info = blockworker.build(a.aid, lat0, lon0, HM_ZOOM, a.q, cx, cy, a.block_size, data_dir, stem + ".png",
                                 ter_name=stem + "_view.ter")
        info = {k: v for k, v in dict(info).items() if isinstance(v, (int, float, str, bool, type(None)))}
        info["q"] = a.q
        info["size"] = a.block_size
        with open(os.path.join(data_dir, stem + ".info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)
    else:
        log(f"reusing the data in {data_dir}")
    for tx, ty in tiles:
        if os.path.isfile(os.path.join(data_dir, f"roads_{a.aid}_{tx}_{ty}.json")) and a.reuse:
            continue
        blockworker = earth_state(a, log)
        log(f"building road tile ({tx}, {ty}) ...")
        blockworker.roads_tile(a.aid, a.anchor[0], a.anchor[1], HM_ZOOM, tx, ty, data_dir, "/vfs", True, 0.0,
                               buildings=not a.no_buildings, left_hand=False, mph=False, q=a.q, gas_stations=True)
    z, holes, info = load_block_files(data_dir, a.aid, cx, cy)
    n = z.shape[0]
    return z, holes, info, cx - n / 2.0, cy - n / 2.0, (cx, cy), tiles, data_dir


def read_tiles(data_dir, aid, tiles, kinds, log):
    """{kind: triangles (n, 3, 3)} of the road tiles' meshes (the .cdae files read back with Earth's cdae)."""
    import cdae
    got = {}
    for tx, ty in tiles:
        p = os.path.join(data_dir, f"roads_{aid}_{tx}_{ty}.json")
        if not os.path.isfile(p):
            log(f"no tile file {os.path.basename(p)}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for kind, shape, pos in tile_entries(data):
            if kind not in kinds:
                continue
            sp = os.path.join(data_dir, shape)
            if not os.path.isfile(sp):
                continue
            try:
                t = shape_triangles(cdae.read(sp), pos)
            except Exception as e:      # (a shape this reader does not know: left out, said)
                log(f"{shape}: not read ({type(e).__name__}: {e})")
                continue
            got.setdefault(kind, []).append(t)
    return {k: np.concatenate(v) for k, v in got.items() if v}


def rally_moved(a, frame, z, x0, y0, log):
    """EarthRally's grader (rally_grader.run, as earth_writers.rally_pass calls it) on z: (heights after, moved mask)."""
    import rally_grader
    earth_state(a, log)             # (the grader reads the tunnels from Earth's map data)
    geoms = rally_grader.load_geometries(a.geometry)
    if not geoms:
        log("--rally needs --geometry (the stage lines the grader shapes along): nothing moved")
        return z, np.zeros(z.shape, bool)
    n = z.shape[0]
    tunnels = rally_grader.tunnel_points(frame, (x0, y0, x0 + z.shape[1], y0 + n))
    log(f"rally grader: {len(geoms)} stage lines, {len(tunnels)} tunnel points, running shape.lua ...")
    shaping = None
    if a.rally_edition:
        with open(a.rally_edition, encoding="utf-8") as f:
            shaping = json.load(f).get("shaping")
    res = rally_grader.run(z, x0, y0, frame, geoms, tunnels, shaping)
    after = res["heights"]
    moved = np.abs(after - z) >= 0.02
    log(f"rally grader: {int(moved.sum())} samples moved 0.02 m or more, largest {np.abs(after - z).max():.2f} m")
    return after, moved


def stage_lines(paths, frame, z, x0, y0, box, mode):
    """The stage lines' 3D polylines (anchor frame) near box. mode 'geometry': the geometry's own heights (ENU of the anchor);
    'ground': the block's ground."""
    out = []
    for gp in paths:
        with open(gp, encoding="utf-8") as f:
            pts = np.asarray(json.load(f)["points"], np.float64)
        e, n, u = frame.geodetic_to_enu(pts[:, 0], pts[:, 1], pts[:, 2] if pts.shape[1] > 2 else 0.0)
        e, n, u = np.asarray(e), np.asarray(n), np.asarray(u)
        if mode == "ground":
            u = bilinear(z, x0, y0, e, n) + 0.2
        near = (e >= box[0] - 30) & (e <= box[2] + 30) & (n >= box[1] - 30) & (n <= box[3] + 30)
        if near.sum() >= 2:
            lo, hi = np.nonzero(near)[0][[0, -1]]
            out.append(np.stack([e[lo:hi + 1], n[lo:hi + 1], u[lo:hi + 1]], axis=1))
    return out


def joined_pairs(argv, names=("--anchor", "--at", "--xy", "--block")):
    """argv with "--xy -446,880" turned into "--xy=-446,880" (argparse takes a value starting with a minus for an option)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in names and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--earth", default=os.environ.get("BEAMNG_EARTH_DIR"), help="Earth source tree (its server/); read only")
    ap.add_argument("--anchor", required=True, type=parse_pair, help="lat,lon of the anchor")
    ap.add_argument("--aid", type=int, default=2, help="the anchor's id")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--at", type=parse_pair, help="lat,lon of the point")
    where.add_argument("--xy", type=parse_pair, help="x,y of the point in the anchor frame (m, x east, y north)")
    ap.add_argument("--radius", type=float, default=120.0, help="m around the point drawn (default 120)")
    ap.add_argument("--out", default="earth-view")
    ap.add_argument("--name", default="earth_view", help="the pictures' file name stem")
    ap.add_argument("--geometry", action="append", default=[], help="a EarthRally stage geometry json drawn as a thin dark line; repeatable")
    ap.add_argument("--line-z", choices=("geometry", "ground"), default="geometry",
                    help="the stage line at its own heights (default), or 0.2 m over the block's ground")
    ap.add_argument("--rally", action="store_true", help="run EarthRally's ground grader (rally_grader.py) on the block along the "
                                                         "--geometry stages and colour the ground it moves magenta")
    ap.add_argument("--rally-edition", help="the edition's legs.json, for its shaping limits")
    ap.add_argument("--views", default="top,SW,SE,N", help="comma list: top, compass points or bearings (camera side); default top,SW,SE,N")
    ap.add_argument("--elev", type=float, default=35.0, help="the oblique cameras' elevation above the horizon (deg)")
    ap.add_argument("--image", default="1600x1000", help="the map's pixels, WxH (a title and legend are added around it)")
    ap.add_argument("--ss", type=int, default=2, help="supersampling per axis (antialiasing)")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS),
                    help="mesh kinds drawn: " + ", ".join(DEFAULT_KINDS) + "; also wire, signs, bldcol, railcol, walkcol, buscol, ...")
    ap.add_argument("--no-buildings", action="store_true", help="build the road tiles without buildings (faster)")
    ap.add_argument("--no-holes", action="store_true", help="draw the ground over the .ter's tunnel holes too")
    ap.add_argument("--bias", type=float, default=DEPTH_BIAS_M, help="m the meshes lying on the ground are drawn nearer than it")
    ap.add_argument("--reuse", action="store_true", help="draw from the data a former run left in <out>/data, building nothing")
    ap.add_argument("--block", type=parse_pair, help="cx,cy of the block (a multiple of 128; default: the one nearest the point)")
    ap.add_argument("--block-size", type=int, default=1024, help="the block's samples per edge (the server's layout)")
    ap.add_argument("--q", type=float, help="the anchor's height quantum (default: the server log's, else 1/32)")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth"))
    ap.add_argument("--server-log", default=os.path.join(os.environ.get("LOCALAPPDATA", ""), "beamng-earth", "logs", "server.log"))
    ap.add_argument("--online", action="store_true", help="fetch the map data the cache lacks")
    ap.add_argument("--no-airports", action="store_true")
    ap.add_argument("--no-sea", action="store_true")
    a = ap.parse_args(joined_pairs(sys.argv[1:]))
    if not a.earth:
        sys.exit("--earth (or BEAMNG_EARTH_DIR) is needed")
    a.earth = os.path.abspath(a.earth)
    a.out = os.path.abspath(a.out)
    a.geometry = [os.path.abspath(g) for g in a.geometry]
    os.makedirs(a.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    os.environ["BNGE_BLOCKCACHE"] = "off"       # (a block from Earth's disk cache of graded blocks would skip the build)
    import earth_trace
    earth_trace.load_earth(a.earth)
    import geodesy
    lat0, lon0 = a.anchor
    frame = geodesy.EnuFrame(lat0, lon0)
    if a.at:
        e, n, _ = frame.geodetic_to_enu(a.at[0], a.at[1], 0.0)
        px, py = float(np.asarray(e)), float(np.asarray(n))
        la, lo = a.at
    else:
        px, py = a.xy
        la, lo, _ = frame.enu_to_geodetic(px, py, 0.0)
        la, lo = float(np.asarray(la)), float(np.asarray(lo))
    log(f"point ({px:.1f}, {py:.1f}) = {la:.7f}, {lo:.7f}; radius {a.radius:g} m")

    z, holes, info, x0, y0, blk, tiles, data_dir = build_earth(a, frame, px, py, log)
    box = (px - a.radius, py - a.radius, px + a.radius, py + a.radius)
    if not (x0 + 1 < box[0] and box[2] < x0 + z.shape[1] - 2 and y0 + 1 < box[1] and box[3] < y0 + z.shape[0] - 2):
        log(f"warning: the radius leaves the block {blk} ({x0:g}..{x0 + z.shape[1] - 1:g}, {y0:g}..{y0 + z.shape[0] - 1:g}): the ground is cut there")
    moved = None
    if a.rally:
        z, moved = rally_moved(a, frame, z, x0, y0, log)
    tris, cls = terrain_mesh(z, x0, y0, box, None if a.no_holes else holes, moved)
    kinds = tuple(k.strip() for k in a.kinds.split(",") if k.strip())
    meshes = read_tiles(data_dir, a.aid, tiles, kinds, log)
    T, C = [tris], [cls]
    for kind, t in sorted(meshes.items()):
        t = crop_triangles(t, box)
        log(f"  {kind}: {len(t)} triangles in the box")
        T.append(t)
        C.append(np.full(len(t), KIND_CLASS.get(kind, OTHER), np.int8))
    tris, cls = np.concatenate(T), np.concatenate(C)
    log(f"  ground: {int((cls <= MOVED).sum())} triangles; total {len(tris)}")
    pz = float(bilinear(z, x0, y0, px, py))
    lines = stage_lines(a.geometry, frame, z, x0, y0, box, a.line_z)
    W, H = (int(v) for v in a.image.lower().split("x"))
    ver = earth_version(a.earth)
    tree = os.path.basename(a.earth.rstrip("/\\"))
    title = f"point {la:.7f}, {lo:.7f}  (x {px:.1f}, y {py:.1f}, ground z {pz:.1f})  radius {a.radius:g} m   Earth: {tree} ({ver})"
    sub = f"anchor {a.aid} at {lat0}, {lon0}; block ({blk[0]:g}, {blk[1]:g}); tiles " + ", ".join(f"{tx}_{ty}" for tx, ty in tiles) + \
        ("; rally grader applied" if a.rally else "")
    corners = np.array([[box[i], box[j], zz] for i in (0, 2) for j in (1, 3)
                        for zz in (float(tris[:, :, 2].min()), float(tris[:, :, 2].max()))])
    target = np.array([px, py, pz])
    paths = []
    for v in (s for s in a.views.split(",") if s.strip()):
        top = v.strip().lower() == "top"
        cam = view_camera(target, 0.0 if top else parse_bearing(v), 90.0 if top else a.elev, W, H)
        fit_camera(cam, corners)
        img, used = render(tris, cls, cam, lines=lines, needle=(px, py, pz, pz + 0.25 * a.radius), ss=a.ss, bias=a.bias)
        suffix = "top-down, north up" if top else f"camera on the {v.strip().upper()} side, {a.elev:g} deg above the horizon"
        pic = frame_picture(img, cam, title, f"{suffix}.  " + sub, used, bool(lines), True)
        path = os.path.join(a.out, f"{a.name}_{view_name(v)}.png")
        pic.save(path)
        paths.append(path)
        log(f"wrote {path}")


if __name__ == "__main__":
    main()
