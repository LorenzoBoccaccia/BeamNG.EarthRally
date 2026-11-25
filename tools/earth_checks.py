"""Checks of Earth's drawn geometry for the debug pictures and the eval, made from the meshes and the map's lines alone (not
from Earth's grading, nor from where the viewer put things):

  vehicle_hits      a reference truck or wagon whose body (from BODY_BASE_M over its wheels to its roof) cuts a road deck, a
                    track, a tunnel tube or a road's ground: the clearance it doesn't have. Its own road or track is under
                    its wheels, under the body, so it never counts.
  penetrations      where two of the map's lines cross at different levels (one on a bridge or in a tunnel, the other not),
                    the meshes of one cutting through the other's: a deck through a track, a track through a road's ground,
                    a tube through the road over it. A surface resting on another (a track on its deck, a deck's end on its
                    embankment) is no penetration: within REST_M of it.

  road_steps        a gap in z, STEP_M or more, where two surfaces should touch, along an edge: a ground road's ribbon (its road
                    mesh's boundary edges in the road's corridor) against the terrain at the same xy, a deck's or a long tunnel's
                    end edge against the surface that continues it. Not a deck's sides over open air. Counted in metres of edge.
  road_ends         at every end of every drivable surface (a deck's, a long tunnel's, a ground road's): its edge's vertices
                    against the surface of a road that goes on from it (one with an end there, or the same road's next stretch)
                    at the same xy: a gap of STEP_M or more is a broken join; where nothing continues it and the raw OSM node of
                    the end is shared with another drivable line, a missing join (a node used by one line is a dead end: fine).
                    Only surfaces meant to be continuous at one level are compared, by the map's own levels (ground, bridge,
                    tunnel): a deck over or under another road is neither a step nor a join.

The first two work on triangles (n, 3, 3) and are exact (edge / triangle intersections), so a count is the same whoever looks at
it. The road checks read the heights of the meshes and the map's lines (where a road is, how wide, the OSM node of its ends),
never Earth's road graph: nodes that were not recognised as connected still show.
"""
import math

import numpy as np

BODY_BASE_M = 0.5           # m over its wheels a vehicle's body starts (its own road under it never counts)
REST_M = 0.3                # m one surface may sink into another and still rest on it (a track on its deck)
FLAT_NZ = 0.7               # a triangle this near horizontal is a surface something rests on
EDGE_EPS_M = 0.02           # m from its ends an edge must cross a triangle (an edge ending on a surface only touches it)
CROSS_NEAR_M = 12.0         # m round a crossing of the map's lines its meshes are compared
GROUND_CORRIDOR_M = 4.0     # m each side of a ground road's line its ground is that road's surface (Earth paints it there)
SHORT_TUNNEL_M = 60.0       # m: a tunnel this short is opened by Earth (gradesep): its road is painted on the ground
STEP_M = 0.1                # m: a gap in z where two surfaces should touch (a road's edge and the terrain, a deck's end and what
                            # continues it): 0.1 m is already a discontinuity that can ruin a driver's day, and there should
                            # be none (not BODY_BASE_M: the clearance of a body is not the smoothness of a road)
SAMPLE_M = 1.0              # m between the samples across a ground road's end
END_SEEK_M = 10.0           # m either side of the end the map's line puts there the mesh's end edge is looked for
LEVEL_M = 2.0               # m: a deck or tube is the same level as an end if it lies within this of it at the same xy (a road passing
                            # over or under it is 4 m or more off); a gap of this or more is another level, not a step
DRIVE_NZ = 0.5              # a face drives if its normal's z is at least this (a slope up to 60 degrees: not a slab's end or side face, which leans a hair
                            # outward at most, nor its underside)
SLAB_M = 1.5                # m: a deck's underside lies this near under its top at most (a deck stacked over another is farther)
NODE_TOL_M = 0.05           # m: a line's end is the OSM node whose connector lies this near it (Overture's lie on the ends)


# ---- exact tests ---------------------------------------------------------------------------------------------------------
def _aabb(t):
    return t.min(axis=1), t.max(axis=1)


def _pairs(A, B, pad=0.0, chunk=4000):
    """(i, j) of the triangles of A and B whose boxes overlap (pad m apart at most)."""
    if not len(A) or not len(B):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    alo, ahi = _aabb(A)
    blo, bhi = _aabb(B)
    I, J = [], []
    for s in range(0, len(A), chunk):
        lo, hi = alo[s:s + chunk, None], ahi[s:s + chunk, None]
        ok = ((lo <= bhi[None] + pad) & (hi >= blo[None] - pad)).all(axis=2)
        i, j = np.nonzero(ok)
        I.append(i + s)
        J.append(j)
    return np.concatenate(I), np.concatenate(J)


def _edges_cross(E0, E1, T, eps=EDGE_EPS_M):
    """Bool (n,): the segment E0 -> E1 crosses triangle T (n, 3, 3) inside it or on its edge, away from the segment's ends
    (Moller-Trumbore)."""
    d = E1 - E0
    L = np.linalg.norm(d, axis=1)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = E0 - a
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (d * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    m = np.where(L > 0, eps / np.maximum(L, 1e-12), 1.0)
    return ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > m) & (t < 1 - m)


def tri_tri_cross(A, B):
    """Bool (n,): triangles A[k] and B[k] (n, 3, 3 each) cut through each other: an edge of one crosses the other inside it."""
    hit = np.zeros(len(A), bool)
    for e in ((0, 1), (1, 2), (2, 0)):
        hit |= _edges_cross(A[:, e[0]], A[:, e[1]], B)
        hit |= _edges_cross(B[:, e[0]], B[:, e[1]], A)
    return hit


def _normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1)
    return n / np.where(ln > 1e-12, ln, 1.0)[:, None]


def _rests_on(A, B, rest=REST_M):
    """Bool (n,): A[k] rests on B[k]: B is a flat-ish surface and no corner of A lies more than rest under B's plane."""
    n = _normals(B)
    flat = np.abs(n[:, 2]) >= FLAT_NZ
    nz = np.where(flat, n[:, 2], 1.0)
    # B's plane's height under each corner of A
    zb = B[:, None, 0, 2] - ((A[:, :, :2] - B[:, None, 0, :2]) * n[:, None, :2]).sum(2) / nz[:, None]
    return flat & ((A[:, :, 2] - zb).min(axis=1) >= -rest)


def cutting(A, B, rest=REST_M):
    """(hit A (len A,), hit B (len B,)): the triangles of A and of B that cut through one of the other set, not resting on it."""
    i, j = _pairs(A, B)
    ha, hb = np.zeros(len(A), bool), np.zeros(len(B), bool)
    if not len(i):
        return ha, hb
    x = tri_tri_cross(A[i], B[j])
    x &= ~(_rests_on(A[i], B[j], rest) | _rests_on(B[j], A[i], rest))
    ha[i[x]] = True
    hb[j[x]] = True
    return ha, hb


# ---- vehicles -------------------------------------------------------------------------------------------------------------
def vehicle_bodies(vt, base=BODY_BASE_M):
    """The bodies of the vehicles drawn as box_triangles (12 triangles each, the bottom face first, then the top): (n, 8, 3)
    corners, the bottom raised base m towards the top (the body over the wheels). Corners: v000 v100 v110 v010 v001 v101 v111
    v011 (length, side, up)."""
    n = len(vt) // 12
    if not n:
        return np.zeros((0, 8, 3))
    t = vt[:n * 12].reshape(n, 12, 3, 3)
    bot = np.stack([t[:, 0, 0], t[:, 0, 1], t[:, 0, 2], t[:, 1, 2]], axis=1)      # v000 v100 v110 v010
    top = np.stack([t[:, 2, 0], t[:, 2, 1], t[:, 2, 2], t[:, 3, 2]], axis=1)      # v001 v101 v111 v011
    h = np.linalg.norm(top[:, 0] - bot[:, 0], axis=1)
    f = np.clip(base / np.maximum(h, 1e-9), 0.0, 0.9)[:, None, None]
    return np.concatenate([bot + f * (top - bot), top], axis=1)


def _body_triangles(c):
    """The 12 triangles (12, 3, 3) of a body's corners c (8, 3)."""
    q = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    return np.array([[c[a], c[b], c[d]] for a, b, cc, d in q for (a, b, d) in ((a, b, cc), (a, cc, d))])


def _inside_body(P, c):
    """Bool (k,): points P (k, 3) inside the body with corners c (8, 3)."""
    o = c[0]
    ax = [c[1] - o, c[3] - o, c[4] - o]
    ok = np.ones(len(P), bool)
    for v in ax:
        s = ((P - o) @ v) / max(float(v @ v), 1e-12)
        ok &= (s > 1e-6) & (s < 1 - 1e-6)
    return ok


def vehicle_hits(vt, others, base=BODY_BASE_M):
    """Bool per vehicle (vt: their triangles, 12 each, box_triangles' order): its body cuts one of others (n, 3, 3) or holds
    one's corner."""
    bodies = vehicle_bodies(vt, base)
    out = np.zeros(len(bodies), bool)
    if not len(others):
        return out
    olo, ohi = _aabb(others)
    for k, c in enumerate(bodies):
        lo, hi = c.min(0), c.max(0)
        near = np.nonzero(((olo <= hi) & (ohi >= lo)).all(axis=1))[0]
        if not len(near):
            continue
        O = others[near]
        if _inside_body(O.reshape(-1, 3), c).any():
            out[k] = True
            continue
        bt = _body_triangles(c)
        i, j = np.meshgrid(np.arange(len(bt)), np.arange(len(O)), indexing="ij")
        out[k] = bool(tri_tri_cross(bt[i.ravel()], O[j.ravel()]).any())
    return out


def struck(vt1, others, base=BODY_BASE_M):
    """Bool (len others,): which of others (n, 3, 3) one vehicle's body (vt1: its 12 triangles) cuts or holds a corner of."""
    out = np.zeros(len(others), bool)
    if not len(others):
        return out
    c = vehicle_bodies(vt1, base)[0]
    olo, ohi = _aabb(others)
    lo, hi = c.min(0), c.max(0)
    near = np.nonzero(((olo <= hi) & (ohi >= lo)).all(axis=1))[0]
    if not len(near):
        return out
    O = others[near]
    hit = _inside_body(O.reshape(-1, 3), c).reshape(-1, 3).any(axis=1)
    bt = _body_triangles(c)
    i, j = np.meshgrid(np.arange(len(bt)), np.arange(len(O)), indexing="ij")
    x = tri_tri_cross(bt[i.ravel()], O[j.ravel()]).reshape(len(bt), len(O)).any(axis=0)
    out[near] = hit | x
    return out


def _depth_in(O, c, n_long=9, n_wide=5):
    """m the surfaces O (n, 3, 3) reach into the body c (8, 3 corners) at most: over a grid of its footprint, each surface's
    height there inside the body's span, the lesser of its height over the body's base and under its top (the shorter move
    that clears it: a deck 1 m too low, a slope 2 m into it). 0 when none is over the footprint inside the span."""
    o, ax, ay = c[0], c[1] - c[0], c[3] - c[0]
    u, v = np.meshgrid(np.linspace(0.02, 0.98, n_long), np.linspace(0.02, 0.98, n_wide))
    S = o[:2] + u.ravel()[:, None] * ax[:2] + v.ravel()[:, None] * ay[:2]
    base = o[2] + u.ravel() * ax[2] + v.ravel() * ay[2]
    top = base + (c[4, 2] - c[0, 2])
    a, b, d = O[:, 0], O[:, 1], O[:, 2]
    e1, e2 = b[:, :2] - a[:, :2], d[:, :2] - a[:, :2]
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(den) > 1e-12
    if not ok.any():
        return 0.0
    a, b, d, e1, e2, den = a[ok], b[ok], d[ok], e1[ok], e2[ok], den[ok]
    w = S[:, None, :] - a[None, :, :2]
    l1 = (w[..., 0] * e2[:, 1] - w[..., 1] * e2[:, 0]) / den
    l2 = (e1[:, 0] * w[..., 1] - e1[:, 1] * w[..., 0]) / den
    inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    z = a[:, 2] + l1 * (b[:, 2] - a[:, 2]) + l2 * (d[:, 2] - a[:, 2])
    pen = np.minimum(z - base[:, None], top[:, None] - z)
    pen = np.where(inside & (pen > 0), pen, 0.0)
    return float(pen.max()) if pen.size else 0.0


def hit_report(vt, kinds, tris, cls, hit, names, base=BODY_BASE_M):
    """[{kind, x, y, z, heading, hits: {class name: how deep m}}] of the vehicles hit marks (vt: 12 triangles each, kinds: each
    one's name, tris / cls: the scene and its classes, names: class -> name): what each one cuts, and how far into its body
    the deepest corner of it reaches (0 for an edge crossing alone). The cases an analysis starts from."""
    bodies = vehicle_bodies(vt, base)
    out = []
    for k in np.flatnonzero(hit):
        c = bodies[k]
        what, parts = {}, {}
        for cl in sorted(set(int(v) for v in np.unique(cls))):
            O = tris[cls == cl]
            if not len(O):
                continue
            s = struck(vt[12 * k:12 * (k + 1)], O, base)
            if not s.any():
                continue
            name = names.get(cl, str(cl))
            what[name] = round(_depth_in(O, c), 2)
            P = O[s]
            parts[name] = {"n": int(s.sum()), "x": round(float(P[:, :, 0].mean()), 1), "y": round(float(P[:, :, 1].mean()), 1),
                           "z_lo": round(float(P[:, :, 2].min()), 2), "z_hi": round(float(P[:, :, 2].max()), 2)}
        ax = c[1] - c[0]
        mid = c.mean(axis=0)
        out.append({"kind": kinds[k], "x": round(float(mid[0]), 1), "y": round(float(mid[1]), 1),
                    "z": round(float(c[:4, 2].mean() - base), 2), "body_base": round(float(c[:4, 2].min()), 2),
                    "body_top": round(float(c[4:, 2].max()), 2),
                    "heading": round(float(np.degrees(np.arctan2(ax[0], ax[1])) % 360), 0), "hits": what, "struck": parts})
    return out


# ---- crossings at different levels ------------------------------------------------------------------------------------------
def _level(ranges, f):
    for a, b, k in ranges:
        if a <= f <= b:
            return k
    return "ground"


def _length(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) >= 2 else 0.0


RANK = {"bridge": 2, "ground": 1, "tunnel": 0}


def level_crossings(lines):
    """[(x, y, upper, lower)] where two of the map's lines cross at different levels; upper / lower: (kind, xy, ranges, rail,
    length): kind "bridge", "ground" or "tunnel" there (the map's, a short tunnel included). lines: {"road": [(xy, ranges)],
    "rail": [...]} (earth_view.map_lines)."""
    L = [(np.asarray(xy, np.float64), r, kind == "rail") for kind in ("road", "rail") for xy, r in (lines or {}).get(kind, [])]
    out = []
    for p in range(len(L)):
        A, ra, rail_a = L[p]
        if len(A) < 2:
            continue
        cum_a = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(A, axis=0).T))])
        for q in range(p + 1, len(L)):
            B, rb, rail_b = L[q]
            if len(B) < 2:
                continue
            cum_b = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(B, axis=0).T))])
            a0, d1 = A[:-1], np.diff(A, axis=0)
            b0, d2 = B[:-1], np.diff(B, axis=0)
            for i in range(len(a0)):
                den = d1[i, 0] * d2[:, 1] - d1[i, 1] * d2[:, 0]
                ok = np.abs(den) > 1e-12
                den = np.where(ok, den, 1.0)
                w = b0 - a0[i]
                ta = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / den
                tb = (w[:, 0] * d1[i, 1] - w[:, 1] * d1[i, 0]) / den
                for j in np.nonzero(ok & (ta > 0) & (ta < 1) & (tb > 0) & (tb < 1))[0]:
                    fa = (cum_a[i] + ta[j] * (cum_a[i + 1] - cum_a[i])) / cum_a[-1]
                    fb = (cum_b[j] + tb[j] * (cum_b[j + 1] - cum_b[j])) / cum_b[-1]
                    ka, kb = _level(ra, fa), _level(rb, fb)
                    if ka == kb:
                        continue
                    x, y = a0[i] + ta[j] * d1[i]
                    la = (ka, A, ra, rail_a, cum_a[-1])
                    lb = (kb, B, rb, rail_b, cum_b[-1])
                    up, lo = (la, lb) if RANK[ka] > RANK[kb] else (lb, la)
                    out.append((float(x), float(y), up, lo))
    return out


def _dist_xy(P, line):
    """Distances (k,) in plan from points P (k, 2) to the polyline line (m, 2)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-12)[None], 0.0, 1.0)
    return np.hypot(*(a[None] + ab[None] * t[..., None] - P[:, None]).transpose(2, 0, 1)).min(axis=1)


def _surface(side, tris, cls, C, near, classes):
    """Indices of the triangles that are this line's surface at the crossing (near: the triangles round it)."""
    kind, xy, ranges, rail, length = side
    G, D, R, T = classes["ground"], classes["deck"], classes["rail"], classes["tunnel"]
    if kind == "bridge":
        want = (cls == D) | (cls == R) if rail else (cls == D)
        return near[want[near]]
    if kind == "tunnel" and not rail:
        # (a short road tunnel is opened by Earth: its road painted on the ground, like a ground road)
        tun = [(a, b) for a, b, k in ranges if k == "tunnel"]
        if not any((b - a) * length <= SHORT_TUNNEL_M for a, b in tun):
            return near[cls[near] == T]
        kind = "ground"
    if kind == "tunnel":
        if rail and not classes.get("rail_tunnels", True):
            # (the Earth tree under test draws no tube for a railway in a tunnel, railways.TUNNELS_ON off: the tunnel-class
            # triangles round the crossing are the road tunnels' and the cuttings' walls, not this railway's; counted as
            # its surface, Vienna's tram and subway tunnels under a busway and a trench "cut" 8 and 12 times there)
            return near[:0]
        return near[cls[near] == T]
    if rail:
        return near[cls[near] == R]
    g = near[np.isin(cls[near], G)]
    return g[_dist_xy(C[g, :2], xy) <= GROUND_CORRIDOR_M]


def penetrations(tris, cls, lines, classes, near_m=CROSS_NEAR_M, xings=None):
    """(hit (len tris,) bool, crossings [(x, y, cut)]): at every crossing of the map's lines at different levels, the triangles
    of the upper line's surface and of the lower's cutting through each other (cutting). classes: {"ground": (ids,), "deck":
    id, "rail": id, "tunnel": id, "rail_tunnels": whether the tree draws a railway's tunnel tube (default True)} (the viewer's
    classes). A line's surface: a road bridge its deck, a railway its track (and its
    deck on a bridge), a long tunnel its tube, a ground road its ground GROUND_CORRIDOR_M round its line. xings: those crossings
    only (level_crossings' items), else all of lines'."""
    hit = np.zeros(len(tris), bool)
    out = []
    if not len(tris):
        return hit, out
    C = tris.mean(axis=1)
    lo, hi = tris[:, :, :2].min(axis=1), tris[:, :, :2].max(axis=1)
    for x, y, up, low in (level_crossings(lines) if xings is None else xings):
        near = np.nonzero((lo[:, 0] <= x + near_m) & (hi[:, 0] >= x - near_m) &
                          (lo[:, 1] <= y + near_m) & (hi[:, 1] >= y - near_m))[0]
        if not len(near):
            continue
        a = _surface(up, tris, cls, C, near, classes)
        b = _surface(low, tris, cls, C, near, classes)
        b = np.setdiff1d(b, a)
        if not len(a) or not len(b):
            continue
        ha, hb = cutting(tris[a], tris[b])
        cut = bool(ha.any())
        hit[a[ha]] = True
        hit[b[hb]] = True
        out.append((x, y, cut))
    return hit, out


def count_cut(crossings, box=None):
    """The crossings that cut, inside box (x0, y0, x1, y1) if given; crossings closer than CROSS_NEAR_M are one."""
    pts = [(x, y) for x, y, cut in crossings if cut and (box is None or (box[0] <= x <= box[2] and box[1] <= y <= box[3]))]
    kept = []
    for p in pts:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > CROSS_NEAR_M for q in kept):
            kept.append(p)
    return len(kept)


# ---- road steps and road ends -----------------------------------------------------------------------------------------------
class Tops:
    """The up-facing triangles of a mesh indexed by plan cell: the heights of what a point of the plan stands on."""

    def __init__(self, tris, cell=8.0):
        t = np.asarray(tris, np.float64).reshape(-1, 3, 3)
        if len(t):
            t = t[_normals(t)[:, 2] > 0]
        self.t, self.cell = t, float(cell)
        self.keys = self.order = np.zeros(0, np.int64)
        self.o = self.n = np.zeros(2, np.int64)
        if not len(t):
            return
        lo = np.floor(t[:, :, :2].min(axis=1) / cell).astype(np.int64)
        hi = np.floor(t[:, :, :2].max(axis=1) / cell).astype(np.int64)
        self.o = lo.min(axis=0)
        lo, hi = lo - self.o, hi - self.o
        self.n = hi.max(axis=0) + 1
        nx = hi[:, 0] - lo[:, 0] + 1
        cnt = nx * (hi[:, 1] - lo[:, 1] + 1)
        tri = np.repeat(np.arange(len(t)), cnt)
        k = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        key = (lo[tri, 0] + k % nx[tri]) * self.n[1] + lo[tri, 1] + k // nx[tri]
        o = np.argsort(key, kind="stable")
        self.keys, self.order = key[o], tri[o]

    def at(self, P):
        """(point index, height, triangle index) of every triangle holding a point of P (k, 2) in plan, at its plane there."""
        P = np.asarray(P, np.float64).reshape(-1, 2)
        if not len(self.keys) or not len(P):
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
        c = np.floor(P / self.cell).astype(np.int64) - self.o
        ok = (c >= 0).all(axis=1) & (c < self.n).all(axis=1)
        key = np.where(ok, c[:, 0] * self.n[1] + c[:, 1], -1)
        lo, hi = np.searchsorted(self.keys, key, "left"), np.searchsorted(self.keys, key, "right")
        cnt = np.where(ok, hi - lo, 0)
        pi = np.repeat(np.arange(len(P)), cnt)
        ti = self.order[np.repeat(lo, cnt) + np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
        a, b, d = self.t[ti, 0], self.t[ti, 1], self.t[ti, 2]
        e1, e2 = b[:, :2] - a[:, :2], d[:, :2] - a[:, :2]
        den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        good = np.abs(den) > 1e-12
        den = np.where(good, den, 1.0)
        w = P[pi] - a[:, :2]
        l1 = (w[:, 0] * e2[:, 1] - w[:, 1] * e2[:, 0]) / den
        l2 = (e1[:, 0] * w[:, 1] - e1[:, 1] * w[:, 0]) / den
        hit = good & (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
        z = a[:, 2] + l1 * (b[:, 2] - a[:, 2]) + l2 * (d[:, 2] - a[:, 2])
        return pi[hit], z[hit], ti[hit]

    def _weld(self):
        """The triangles' corners welded at 1 mm: ids (n, 3), and the triangles by corner id."""
        if hasattr(self, "vid"):
            return
        key = np.round(self.t.reshape(-1, 3) / 1e-3).astype(np.int64)
        u, inv = np.unique(key, axis=0, return_inverse=True) if len(key) else (np.zeros((0, 3), np.int64), np.zeros(0, np.int64))
        self.vid, self.nv = inv.reshape(-1, 3), len(u)
        self.vmap = {tuple(k): i for i, k in enumerate(u.tolist())}
        o = np.argsort(self.vid.ravel(), kind="stable")
        self.vtri, self.vsorted = o // 3, self.vid.ravel()[o]

    def boundary(self):
        """(A (e, 3), B (e, 3), triangle (e,)): the edges used by one triangle alone (the mesh's boundary: a deck's ends and sides,
        and a gap in z between two meshes that touch in plan, their corners being welded in z too), with their triangle."""
        if hasattr(self, "_bnd"):
            return self._bnd
        self._weld()
        n = len(self.t)
        if not n:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
        e = np.stack([self.vid[:, [0, 1]], self.vid[:, [1, 2]], self.vid[:, [2, 0]]], axis=1).reshape(-1, 2)
        c = np.stack([self.t[:, [0, 1]], self.t[:, [1, 2]], self.t[:, [2, 0]]], axis=1).reshape(-1, 2, 3)
        lo, hi = e.min(axis=1), e.max(axis=1)
        _u, inv, cnt = np.unique(lo * self.nv + hi, return_inverse=True, return_counts=True)
        b = (cnt[inv] == 1) & (lo != hi)
        self._bnd = (c[b, 0], c[b, 1], np.repeat(np.arange(n), 3)[b])
        return self._bnd

    def components(self):
        """(n,) the connected piece each triangle belongs to: triangles sharing a corner (welded at 1 mm) are one piece, so a deck
        and the deck passing over it are two."""
        if hasattr(self, "_comp"):
            return self._comp
        self._weld()
        n = len(self.t)
        lab = np.arange(n)
        if n:
            flat = self.vid.ravel()
            while True:
                vm = np.full(self.nv, n, np.int64)
                np.minimum.at(vm, flat, np.repeat(lab, 3))
                new = np.minimum(lab, vm[self.vid].min(axis=1))
                new = new[new]
                if (new == lab).all():
                    break
                lab = new
        self._comp = lab
        return lab

    def vertex_triangles(self, P, Z):
        """[set of triangle indices] per vertex (P (k, 2), Z (k,)): the triangles that have a corner there."""
        self._weld()
        out = []
        for p, z in zip(P, Z):
            v = self.vmap.get(tuple(np.round(np.array([p[0], p[1], z]) / 1e-3).astype(np.int64).tolist()))
            out.append(set() if v is None else set(self.vtri[np.searchsorted(self.vsorted, v, "left"):
                                                              np.searchsorted(self.vsorted, v, "right")].tolist()))
        return out

    def lowest(self, P):
        """(k,) the lowest height of what each point of P stands on, NaN where nothing does (the terrain: its one surface; a
        tube: its floor, not its roof)."""
        pi, z, _ = self.at(P)
        out = np.full(len(np.asarray(P).reshape(-1, 2)), np.inf)
        np.minimum.at(out, pi, z)
        return np.where(np.isfinite(out), out, np.nan)


def _pose(xy, cum, s):
    """Point (k, 2) and unit tangent (k, 2) of the polyline xy (cumulative length cum) at the distances s (k,)."""
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(xy) - 2)
    d = xy[i + 1] - xy[i]
    ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
    t = (s - cum[i]) / ln
    return xy[i] + d * t[:, None], d / ln[:, None]


def _structures(ranges, length):
    """([(f0, f1, kind)] the bridges and long tunnels of a line, [(f0, f1)] its ground): a short tunnel is ground, Earth opens it."""
    st = sorted((float(a), float(b), k) for a, b, k in ranges
                if b > a and (k == "bridge" or (k == "tunnel" and (b - a) * length > SHORT_TUNNEL_M)))
    ground, f = [], 0.0
    for a, b, _k in st:
        if a - f > 1e-9:
            ground.append((f, a))
        f = max(f, b)
    if 1.0 - f > 1e-9:
        ground.append((f, 1.0))
    return st, ground


def _road_lines(lines):
    """[{xy, cum, length, half, st, ground, nodes, id}] of the map's drivable roads (map_lines' "road", with its "road_half",
    "road_nodes" and "road_id" when it has them: the carriageway's half width, the OSM nodes of the line's two ends)."""
    out = []
    lines = lines or {}
    for i, (xy, ranges) in enumerate(lines.get("road", [])):
        xy = np.asarray(xy, np.float64)
        if len(xy) < 2:
            continue
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        if cum[-1] <= 0:
            continue
        half = (lines.get("road_half") or {})
        half = half[i] if i < len(half) else None
        nodes = (lines.get("road_nodes") or [])
        nodes = nodes[i] if i < len(nodes) else None
        rid = (lines.get("road_id") or [])
        rid = rid[i] if i < len(rid) else None
        st, ground = _structures(ranges, cum[-1])
        out.append(dict(xy=xy, cum=cum, length=float(cum[-1]), half=float(half) if half else GROUND_CORRIDOR_M, st=st,
                        ground=ground, nodes=tuple(nodes) if nodes else (None, None), id=rid))
    return out


def _lateral(t, half):
    """(offsets (m,), weights (m,), normal (2,)): samples across a width of 2 half along a road whose unit tangent is t, each
    standing for about SAMPLE_M of it."""
    m = max(int(math.ceil(2 * half / SAMPLE_M)), 1)
    w = 2 * half / m
    return -half + (np.arange(m) + 0.5) * w, np.full(m, w), np.array([-t[1], t[0]])


def _up_sign(tris, cls, classes):
    """+1 or -1: the sign of the z of the normals (winding) of the faces that look up in this scene, read from its road mesh (a single
    surface, all facing up: the majority of its flat faces), +1 where there is none. Earth's .cdae meshes are wound the other way
    (-1: a deck's driving surface faces down by its winding, and its underside up)."""
    if "road" not in classes:
        return 1.0
    t = tris[cls == classes["road"]]
    if not len(t):
        return 1.0
    nz = _normals(t)[:, 2]
    flat = np.abs(nz) >= 0.8
    return 1.0 if (nz[flat] > 0).sum() >= (nz[flat] < 0).sum() else -1.0


def _facing_up(t, up):
    """t (n, 3, 3) wound so the faces that look up have a normal of positive z (what Tops reads): reversed if up is -1."""
    return t[:, ::-1] if up < 0 and len(t) else t


class _Scene:
    """What the road checks read: the heights of the terrain, the decks and the tubes, and where each road's ground, bridge and
    long tunnel stretches run (their corridors), by the map's own levels."""

    def __init__(self, tris, cls, lines, classes):
        tris, cls = np.asarray(tris, np.float64), np.asarray(cls)
        up = _up_sign(tris, cls, classes)
        self.ground = Tops(tris[np.isin(cls, classes["ground"])])
        self.deck = Tops(_facing_up(tris[cls == classes["deck"]], up))
        self.tunnel = Tops(tris[cls == classes["tunnel"]])
        self.road = Tops(tris[cls == classes["road"]] if "road" in classes else tris[:0])
        self.roads = _road_lines(lines)
        self.corridors = []     # per road: {level: [(a stretch's polyline (k, 2), its bounding box)]}, level "ground", "bridge", "tunnel"
        self.bounds = []        # per road: the points where its level can change or it ends: its ends and its stretches' ends
        for r in self.roads:
            cor = {"ground": [], "bridge": [], "tunnel": []}
            spans = [(a, b, "ground") for a, b in r["ground"]] + [(a, b, k) for a, b, k in r["st"]]
            for f0, f1, kd in spans:
                a, b = f0 * r["length"], f1 * r["length"]
                s = np.concatenate([[a], r["cum"][(r["cum"] > a) & (r["cum"] < b)], [b]])
                p, _t = _pose(r["xy"], r["cum"], s)
                cor[kd].append((p, np.concatenate([p.min(axis=0), p.max(axis=0)])))
            self.corridors.append(cor)
            fs = sorted({0.0, 1.0} | {f for a, b, _k in r["st"] for f in (a, b)})
            self.bounds.append(_pose(r["xy"], r["cum"], np.array(fs) * r["length"])[0])

    def in_corridor(self, P, k, level):
        """Bool (n,): P (n, 2) lies within the road k's width of one of its stretches at this level."""
        out = np.zeros(len(P), bool)
        h = self.roads[k]["half"]
        for p, b in self.corridors[k][level]:
            m = (P[:, 0] >= b[0] - h) & (P[:, 0] <= b[2] + h) & (P[:, 1] >= b[1] - h) & (P[:, 1] <= b[3] + h)
            if m.any():
                out[m] |= _dist_xy(P[m], p) <= h
        return out

    def continuing(self, k, P0, width):
        """[road j]: the other roads with an end, or the end of a bridge or long tunnel, within width of P0 (an end of the road k):
        the roads that are meant to go on from it. A road passing over or under it has none there."""
        return [j for j in range(len(self.roads)) if j != k and len(self.bounds[j])
                and np.hypot(*(self.bounds[j] - P0).T).min() <= width]

    def surface(self, P, j, zref, ground_as=None, excl=None, level=None):
        """[(height, level)] per point of P (n, 2) of the road j's own surface there, by the map's level of its stretch under the
        point: the terrain in its ground corridor, the deck (of those under the point the nearest to zref (n,)) in its bridge's,
        the tube's floor in its tunnel's; NaN where it has none. excl: per point the deck triangles that do not count. level: the
        deck counts only within this of zref (the same level: not one passing over or under). ground_as (n,): the terrain's height is this instead (ground
        meeting ground is one surface: continuous by its making)."""
        n = len(P)
        out = np.full(n, np.nan)
        zg = self.ground.lowest(P) if ground_as is None else np.where(np.isfinite(self.ground.lowest(P)), ground_as, np.nan)
        m = self.in_corridor(P, j, "ground")
        out[m] = zg[m]
        zt = self.tunnel.lowest(P)
        m = self.in_corridor(P, j, "tunnel") & np.isfinite(zt)
        out[m] = zt[m]
        m = self.in_corridor(P, j, "bridge")
        if m.any():
            zd = _near_hit(self.deck, P, zref, excl, level)
            m &= np.isfinite(zd)        # (a corridor is round at its ends: a point just past a bridge's has no deck)
            out[m] = zd[m]
        return out


def _near_hit(tops, P, zref, excl=None, level=None):
    """(n,) the height of the triangle of tops under each point of P (n, 2) nearest zref (n,), NaN where none; excl: per point the
    triangle indices that do not count (the ones the point is a corner of); level: only those within this of zref count."""
    pi, z, ti = tops.at(P)
    out, best = np.full(len(P), np.nan), np.full(len(P), np.inf)
    for a, b, t in zip(pi, z, ti):
        if excl is not None and t in excl[a]:
            continue
        if abs(b - zref[a]) < best[a] and (level is None or abs(b - zref[a]) < level):
            best[a], out[a] = abs(b - zref[a]), b
    return out


def _gap(z, cands):
    """(n,) the smallest vertical gap between each height z (n,) and the heights cands (c, n) of what continues it at the same
    xy, inf where nothing does."""
    if not len(cands):
        return np.full(len(z), np.inf)
    return np.where(np.isfinite(cands), np.abs(cands - z[None]), np.inf).min(axis=0)


def _own_pieces(tops, r, span):
    """[(triangle indices of tops, those of the undersides under them)]: the pieces of tops (Tops.components) the line r runs along over its stretch span
    (f0, f1): those under all (0.8 of the most) of the samples of its centre line. Two or more where decks are stacked on it:
    which of them is the line's is for what continues its end to say; not a slab's underside, which lies under its top."""
    lab = tops.components()
    a, b = span[0] * r["length"], span[1] * r["length"]
    s = np.arange(a + 0.5, b - 0.4, 2.0)
    if not len(s):
        s = np.array([(a + b) / 2])
    pi, z, ti = tops.at(_pose(r["xy"], r["cum"], s)[0])
    if not len(ti):
        return []
    zs = {}
    for p_, z_, t_ in zip(pi.tolist(), z.tolist(), ti.tolist()):
        key = (p_, int(lab[t_]))
        zs[key] = max(zs.get(key, -math.inf), z_)
    cnt = {}
    for (_p, c) in zs:
        cnt[c] = cnt.get(c, 0) + 1
    top = max(cnt.values())
    keep = [c for c, v in sorted(cnt.items()) if v >= 0.8 * top]
    # (a piece lying within a slab's thickness under a piece of the line, anywhere along it, is its underside, not a surface to
    # drive on; nor does it continue the end)
    under = {c for (p_, c), z_ in zs.items() if any(0.0 < zs[(p_, c2)] - z_ < SLAB_M for c2 in keep if (p_, c2) in zs)}
    gone = np.flatnonzero(np.isin(lab, list(under)))
    return [(np.flatnonzero(lab == c), gone) for c in keep if c not in under]


def _end_edges(tops, own, P0, tg, half):
    """[(p xy, p xy, z, z, width)]: samples (SAMPLE_M apart, each standing for its width) along the end edge of the piece own
    (triangle indices of tops) of a bridge or tube end at P0 (unit tangent tg out of the structure): the piece's boundary edges
    across the road within END_SEEK_M of P0, facing out, cut to the carriageway's width (half either side of the line: not the
    edge's tips beyond it), and only where they are the piece's driving surface, not an underside or a side under it."""
    # (the driving surface alone: the piece's faces that face up, not the end and side faces of its slab, even where they lean a hair
    # outward: their lower edges are the slab's depth below the road)
    own = np.asarray(sorted(own), np.int64)
    top = Tops(tops.t[own][_normals(tops.t[own])[:, 2] >= DRIVE_NZ])
    A, B, ti = top.boundary()
    if not len(A):
        return []
    nrm = np.array([-tg[1], tg[0]])
    mid, d = (A + B) / 2, B[:, :2] - A[:, :2]
    cen = top.t[ti].mean(axis=1)[:, :2]
    ok = (np.abs((mid[:, :2] - P0) @ tg) <= END_SEEK_M) & (np.abs(d @ tg) < np.abs(d @ nrm)) & (((cen - mid[:, :2]) @ tg) < 0)
    out = []
    for e in np.flatnonzero(ok):
        la, lb = (A[e, :2] - P0) @ nrm, (B[e, :2] - P0) @ nrm
        if abs(lb - la) < 1e-9:
            t0, t1 = (0.0, 1.0) if abs(la) <= half else (1.0, 0.0)
        else:
            ta, tb = (-half - la) / (lb - la), (half - la) / (lb - la)
            t0, t1 = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
        if t1 - t0 <= 1e-6:
            continue
        a, b = A[e] + t0 * (B[e] - A[e]), A[e] + t1 * (B[e] - A[e])
        m = max(int(math.ceil(np.hypot(*(b[:2] - a[:2])) / SAMPLE_M)), 1)
        u = ((np.arange(m) + 0.5) / m)[:, None]
        Q = a[None] + u * (b - a)[None]
        pi, z, _t = top.at(Q[:, :2])
        hi = np.full(m, -np.inf)
        np.maximum.at(hi, pi, z)
        w = float(np.hypot(*(b[:2] - a[:2]))) / m
        out += [(Q[i, :2], Q[i, :2], Q[i, 2], Q[i, 2], w) for i in range(m) if Q[i, 2] >= hi[i] - 0.02]
    return out


def road_checks(tris, cls, lines, classes, box=None):
    """{"steps": [...], "ends": [...]}: the road steps and the road ends of a scene (tris (n, 3, 3), cls their classes; classes:
    {"ground": (ids,), "deck": id, "tunnel": id, "road": id}, the viewer's) round the map's lines (earth_view.map_lines), those
    whose edge or end lies in box (x0, y0, x1, y1) if given.

    The measure is one: a gap in z where two surfaces should touch, at the same xy, STEP_M or more. No slope is read, nothing
    beyond an edge. Only what is meant to be continuous at one level is compared, by the map's own levels (ground, bridge,
    tunnel): a deck passing over or under another road is neither a step nor a join, nor is a bridge's side.

    A step {x, y, z_edge, z_beyond, m, kind, run, line}: m of an edge of a mesh with a gap: the edge of a ground road's ribbon
    (the road mesh's boundary edges lying in a ground road's corridor, not resting on a deck) against the terrain at the same xy
    (z_beyond), or the end edge of a bridge's deck or a long tunnel's tube against the surface that continues it there.
    An end {x, y, z, width, kind, what, line, node}: of a bridge, a long tunnel or a ground road's line. Its edge's vertices (a
    deck's or a tube's end edge in its mesh: the boundary edges across the road within END_SEEK_M of where the line ends,
    facing out, of the line's own piece of the mesh (not a deck passing over it, nor a slab's underside), sampled across the
    carriageway's width at its driving surface; a ground road's: samples across its width on the terrain) are compared with the surface of a road that
    goes on from it (the same road's next stretch, or a road with an end there: _Scene.continuing), at the same xy: kind
    "joined" (none of its vertices has a gap), "broken" (a vertex has a gap, no surface that continuing within STEP_M),
    "missing" (nothing continues it at any vertex, and its OSM node is another drivable line's end too) or "dead" (nothing
    does, its node is its alone: a dead end). The map's graph is not read: only where lines are, how wide, which level, and
    which OSM node their ends are."""
    S = _Scene(tris, cls, lines, classes)

    def inside(p):
        return box is None or (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])

    steps, ends, shared = [], [], {}
    state = {"run": 0}

    def add_step(mid, z_edge, z_beyond, m, kind, line, new):
        if new:
            state["run"] += 1
        steps.append(dict(x=float(mid[0]), y=float(mid[1]), z_edge=float(z_edge), z_beyond=float(z_beyond), m=float(m), kind=kind,
                          run=state["run"], line=line))

    for k, r in enumerate(S.roads):
        for n in r["nodes"]:
            if n is not None:
                shared.setdefault(n, set()).add(k)
    # a ground road's ribbon: its boundary edges in the corridors of the roads' ground stretches against the terrain under them
    rd = classes.get("road")
    if rd is not None and S.roads:
        T = S.road
        A, B, ti = T.boundary()
        if len(A):
            mid = (A[:, :2] + B[:, :2]) / 2
            on_ground = np.zeros(len(A), bool)
            owner = np.full(len(A), -1)
            for k in range(len(S.roads)):
                m = S.in_corridor(mid, k, "ground") & (owner < 0)
                owner[m] = k
                on_ground |= m
            za, zb = S.ground.lowest(A[:, :2]), S.ground.lowest(B[:, :2])
            gap = np.maximum(np.abs(A[:, 2] - za), np.abs(B[:, 2] - zb))
            cand = np.flatnonzero(on_ground & np.isfinite(gap) & (gap >= STEP_M - 1e-6) &
                                  np.array([inside(q) for q in mid]))
            # (an edge of a road mesh lying on a deck over the ground stretch is the deck's road, not the ground's)
            for e in cand:
                rest = [_near_hit(S.deck, P[None], np.array([zz]))[0] for P, zz in ((A[e, :2], A[e, 2]), (B[e, :2], B[e, 2]))]
                if all(np.isfinite(v) and abs(v - zz) < STEP_M - 1e-6 for v, zz in zip(rest, (A[e, 2], B[e, 2]))):
                    continue
                w = 0 if abs(A[e, 2] - za[e]) >= abs(B[e, 2] - zb[e]) else 1
                P, zt = (A, za) if w == 0 else (B, zb)
                add_step(mid[e], P[e, 2], zt[e], np.hypot(*(B[e, :2] - A[e, :2])), "ground", S.roads[owner[e]]["id"], True)
    # the ends: of each bridge and long tunnel, and of the line's own end where it lies on the ground
    for k, r in enumerate(S.roads):
        cand = []
        for a, b, kd in r["st"]:
            if not any(abs(b2 - a) < 1e-9 for _a2, b2, _k2 in r["st"]):
                cand.append((a, -1.0, kd, (a, b)))
            if not any(abs(a2 - b) < 1e-9 for a2, _b2, _k2 in r["st"]):
                cand.append((b, 1.0, kd, (a, b)))
        if r["ground"] and r["ground"][0][0] < 1e-9:
            cand.append((0.0, -1.0, "ground", None))
        if r["ground"] and r["ground"][-1][1] > 1.0 - 1e-9:
            cand.append((1.0, 1.0, "ground", None))
        for f, sign, kd, span in cand:
            p, t = _pose(r["xy"], r["cum"], np.array([f * r["length"]]))
            P0, tg = p[0], t[0] * sign
            if not inside(P0):
                continue
            nrm = np.array([-tg[1], tg[0]])

            def judge(edges, own):
                """The edge's vertices against what continues the end at the same xy: the roads with an end within its width, and the
                line's own next stretch if it is ground. own: the triangles of the structure's own piece (they never continue it)."""
                Pv = np.array([q for e in edges for q in (e[0], e[1])])
                Zv = np.array([q for e in edges for q in (e[2], e[3])])
                ex = None if own is None else [own] * len(Pv)
                if own is not None:
                    # (a piece under the structure's own surface a metre inside its end is its underside: it never continues it)
                    T = S.deck if kd == "bridge" else S.tunnel
                    lab = T.components()
                    pi, z, ti = T.at(Pv - tg[None])
                    zt = np.full(len(Pv), -np.inf)
                    mine = np.array([t in own for t in ti.tolist()], bool)
                    np.maximum.at(zt, pi[mine], z[mine])
                    ex = [set(own) for _ in Pv]
                    for a_, z_, t_ in zip(pi.tolist(), z.tolist(), ti.tolist()):
                        if 0.0 < zt[a_] - z_ < SLAB_M and t_ not in own:
                            ex[a_] |= set(np.flatnonzero(lab == lab[t_]).tolist())
                cs = [S.surface(Pv, j, Zv, Zv if kd == "ground" else None, ex, LEVEL_M) for j in S.continuing(k, P0, r["half"])]
                if kd != "ground" and any(abs((g0 if sign > 0 else g1) - f) < 1e-9 for g0, g1 in r["ground"]):
                    cs.append(S.surface(Pv, k, Zv, None, ex, LEVEL_M))
                cands = np.array(cs) if cs else np.full((0, len(Pv)), np.nan)
                return Pv, Zv, cands, _gap(Zv, cands)

            if kd == "ground":
                off, wt, _n = _lateral(tg, r["half"])
                V = P0[None] + off[:, None] * nrm[None]
                zv = S.ground.lowest(V)
                keep = np.isfinite(zv)
                V, zv, wk = V[keep], zv[keep], wt[keep]
                edges = [(V[i], V[i], zv[i], zv[i], float(wk[i])) for i in range(len(V))]
                if not edges:
                    continue
                Pv, Zv, cands, gap = judge(edges, None)
            else:
                tops = S.deck if kd == "bridge" else S.tunnel
                best = None
                for idx, under in _own_pieces(tops, r, span):
                    edges = _end_edges(tops, idx, P0, tg, r["half"])
                    if not edges:
                        continue
                    res = judge(edges, set(idx.tolist()) | set(under.tolist()))
                    g = res[3][np.isfinite(res[3])]
                    score = (float(np.median(g)) if len(g) else math.inf, float(res[1].mean()))
                    if best is None or score < best[0]:
                        best = (score, edges, res)
                if best is None:
                    continue
                _sc, edges, (Pv, Zv, cands, gap) = best
            have = np.isfinite(gap)
            bad = have & (gap >= STEP_M - 1e-6)
            if kd != "ground":
                new = True
                for i, e in enumerate(edges):
                    if bad[2 * i] or bad[2 * i + 1]:
                        w = 2 * i if (bad[2 * i] and (not bad[2 * i + 1] or gap[2 * i] >= gap[2 * i + 1])) else 2 * i + 1
                        near = np.where(np.isfinite(cands[:, w]), np.abs(cands[:, w] - Zv[w]), np.inf).argmin() if len(cands) else 0
                        add_step((Pv[2 * i] + Pv[2 * i + 1]) / 2, Zv[w], cands[near, w], e[4], kd, r["id"], new)
                        new = False
            node = r["nodes"][0 if sign < 0 else 1] if f in (0.0, 1.0) else None
            other = node is not None and len(shared.get(node, ())) > 1
            how = "broken" if bad.any() else "joined" if have.any() else "missing" if other else "dead"
            c = Pv.mean(axis=0)
            ends.append(dict(x=float(c[0]), y=float(c[1]), z=float(Zv.mean()), width=float(sum(e[4] for e in edges)), kind=how,
                             what=kd, line=r["id"], node=node))
    return {"steps": steps, "ends": ends}


def count_steps(steps, box=None):
    """m of road edge stepping (road_checks' steps), inside box (x0, y0, x1, y1) if given."""
    return round(sum(s["m"] for s in steps if box is None or (box[0] <= s["x"] <= box[2] and box[1] <= s["y"] <= box[3])), 1)


def count_ends(ends, kind, box=None):
    """The road ends of this kind ("broken", "missing") inside box (x0, y0, x1, y1) if given."""
    return sum(1 for e in ends if e["kind"] == kind and (box is None or (box[0] <= e["x"] <= box[2] and box[1] <= e["y"] <= box[3])))


def step_runs(steps):
    """[{x, y, z_edge, z_beyond, m, step, kind, line}] one per run of steps along an edge, the longest first: where its tallest
    step is, how long the run is (m) and how tall the step."""
    runs = {}
    for s in steps:
        runs.setdefault(s["run"], []).append(s)
    out = []
    for g in runs.values():
        w = max(g, key=lambda s: abs(s["z_beyond"] - s["z_edge"]))
        out.append(dict(x=round(w["x"], 1), y=round(w["y"], 1), z_edge=round(w["z_edge"], 2), z_beyond=round(w["z_beyond"], 2),
                        m=round(sum(s["m"] for s in g), 1), step=round(abs(w["z_beyond"] - w["z_edge"]), 2), kind=w["kind"],
                        line=w["line"]))
    return sorted(out, key=lambda r: -r["m"])
