"""Checks of Earth's drawn geometry for the debug pictures and the eval, made from the meshes and the map's lines alone (not
from Earth's grading, nor from where the viewer put things):

  vehicle_hits      a reference truck or wagon whose body (from BODY_BASE_M over its wheels to its roof) cuts a road deck, a
                    track, a tunnel tube or a road's ground: the clearance it doesn't have. Its own road or track is under
                    its wheels, under the body, so it never counts.
  penetrations      where two of the map's lines cross at different levels (one on a bridge or in a tunnel, the other not),
                    the meshes of one cutting through the other's: a deck through a track, a track through a road's ground,
                    a tube through the road over it. A surface resting on another (a track on its deck, a deck's end on its
                    embankment) is no penetration: within REST_M of it.

  road_steps        along the edges of every drivable surface (a ground road's corridor against the ground beside it; a deck's and a
                    long tunnel's ends against what goes on from them: not a deck's sides over open air), where the height beyond
                    the edge jumps STEP_M or more from what the surface's own slope carries on: a wall a road drops into, a cliff
                    beside it, a deck ending over the ground. Counted in metres of edge.
  road_ends         at every end of every drivable surface (a deck's, a long tunnel's, a ground road's), whether the surface of a
                    road that goes on from it (one with an end there, or the same road's next stretch) lies beyond it: one that
                    does at another height than STEP_M is a broken join; where none does and the raw OSM node of the end is
                    shared with another drivable line, a missing join (a node used by one line is a dead end: fine).
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
STEP_M = 0.1                # m: a road's surface may not jump this much anywhere: 0.1 m is already a discontinuity that can ruin a
                            # driver's day, and there should be none (not BODY_BASE_M: the clearance of a body is not the
                            # smoothness of a road)
STEP_REACH_M = 1.0          # m beyond an edge or an end what is there is read: the terrain lattice's step (a step is a change
                            # within one cell); also the m of edge a sample stands for
MARCH_M = 0.25              # m between the probes of the march that finds a deck's or a tube's end in its mesh
END_SEEK_M = 10.0           # m either side of the end the map's line puts there the mesh's end is looked for
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
    standing for about STEP_REACH_M of it."""
    m = max(int(math.ceil(2 * half / STEP_REACH_M)), 1)
    w = 2 * half / m
    return -half + (np.arange(m) + 0.5) * w, np.full(m, w), np.array([-t[1], t[0]])


class _Scene:
    """What the road checks read: the heights of the terrain, the decks and the tubes, and where each road's ground, bridge and
    long tunnel stretches run (their corridors), by the map's own levels."""

    def __init__(self, tris, cls, lines, classes):
        tris, cls = np.asarray(tris, np.float64), np.asarray(cls)
        self.ground = Tops(tris[np.isin(cls, classes["ground"])])
        self.deck = Tops(tris[cls == classes["deck"]])
        self.tunnel = Tops(tris[cls == classes["tunnel"]])
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

    def surface(self, P, j, zref, ground_as=None):
        """[(height, level)] per point of P (n, 2) of the road j's own surface there, by the map's level of its stretch under the
        point: the terrain in its ground corridor, the deck (of those under the point the nearest to zref (n,)) in its bridge's,
        the tube's floor in its tunnel's; NaN where it has none. ground_as (n,): the terrain's height is this instead (ground
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
            zd = _near_hit(self.deck, P, zref)
            m &= np.isfinite(zd)        # (a corridor is round at its ends: a point just past a bridge's has no deck)
            out[m] = zd[m]
        return out


def _deck_end(scene, kind, P0, t, half):
    """[(position (2,), height, weight, tangent)] over the width at the end of a bridge (kind "bridge") or a long tunnel that the
    map's line puts at P0 (outward unit tangent t): where the mesh's end lies there, found by marching along t from END_SEEK_M
    inside to END_SEEK_M outside over each sample of the width, from the mesh nearest the line's end, on while it goes on within
    STEP_M of where its own slope carries it (a gap of STEP_REACH_M is bridged): a jump of STEP_M ends it. A mesh that goes
    on past the seek has no end there."""
    tops = scene.deck if kind == "bridge" else scene.tunnel
    off, wt, nrm = _lateral(t, half)
    nd = int(round(2 * END_SEEK_M / MARCH_M)) + 1
    d = -END_SEEK_M + np.arange(nd) * MARCH_M
    P = P0[None, None] + off[:, None, None] * nrm[None, None] + d[None, :, None] * t[None, None]
    pi, z, _ = tops.at(P.reshape(-1, 2))
    cov = {}
    for a, b in zip(pi, z):
        cov.setdefault(int(a), []).append(float(b))
    if kind != "bridge":
        cov = {a: [min(v)] for a, v in cov.items()}
    gap = int(round(STEP_REACH_M / MARCH_M))
    k0, out = nd // 2, []
    for j in range(len(off)):
        def zs(k):
            return cov.get(j * nd + k, [])
        start = next((k for r in range(nd) for k in (k0 - r, k0 + r) if 0 <= k < nd and zs(k)), None)
        if start is None:
            continue
        levels = []
        for z0 in sorted(zs(start)):
            if not levels or z0 - levels[-1] > MARCH_M:
                levels.append(z0)
        for z0 in levels:
            k, zc = start, z0
            back = [v for v in zs(start - 1) if abs(v - z0) <= MARCH_M]
            slope = z0 - min(back, key=lambda v: abs(v - z0)) if back else 0.0         # (m per march step)
            while k < nd - 1:
                nxt = [(kk, abs(v - zc - (kk - k) * slope), v) for kk in range(k + 1, min(k + gap, nd - 1) + 1) for v in zs(kk)
                       if abs(v - zc - (kk - k) * slope) < STEP_M - 1e-6]
                if not nxt:
                    break
                kk, _dz, v = min(nxt, key=lambda q: (q[0], q[1]))
                slope, k, zc = (v - zc) / (kk - k), kk, v
            if k < nd - 1:
                out.append((P0 + off[j] * nrm + d[k] * t, zc, float(wt[j]), t))
    return out


def _near_hit(tops, P, zref):
    """(n,) the height of the triangle of tops under each point of P (n, 2) nearest zref (n,), NaN where none."""
    pi, z, _ = tops.at(P)
    out, best = np.full(len(P), np.nan), np.full(len(P), np.inf)
    for a, b in zip(pi, z):
        if abs(b - zref[a]) < best[a]:
            best[a], out[a] = abs(b - zref[a]), b
    return out


def _kink(z_edge, z_beyond, z_inside):
    """m a surface jumps across an edge: the height beyond it less the height its own slope carries on from inside it (a slope
    is no step; the inside height is the edge's own where there is none)."""
    zi = np.where(np.isfinite(z_inside), z_inside, z_edge)
    return np.abs(z_beyond - (2 * z_edge - zi))


def road_checks(tris, cls, lines, classes, box=None):
    """{"steps": [...], "ends": [...]}: the road steps and the road ends of a scene (tris (n, 3, 3), cls their classes; classes:
    {"ground": (ids,), "deck": id, "tunnel": id}, the viewer's) round the map's lines (earth_view.map_lines), those whose
    sample or end lies in box (x0, y0, x1, y1) if given.

    Only what is meant to be continuous at one level is compared: a ground road's edge with the ground beside it, an end with
    the surface of a road that goes on from it (the same road's next stretch, or a road with an end there: _Scene.continuing),
    at the level the map's lines give that stretch (ground, bridge, tunnel). A deck passing over or under another road is
    neither a step nor a join, nor is the ground below a bridge's side.

    A step {x, y, z_edge, z_beyond, m, kind, run, line}: m of a road's edge, a ground road's (its carriageway's width either side
    of its line) or a bridge's or long tunnel's end (across its width; its sides are no edge), where the height beyond it
    (STEP_REACH_M on) jumps STEP_M or more from what the surface's own slope carries on (_kink). The deck's end is found in its
    mesh (_deck_end), the map's line says only where to look; the ground beside a bridge's end is the reference where the line
    goes on as ground there.
    An end {x, y, z, width, kind, what, line, node}: of a bridge, a long tunnel or a ground road's line, kind "joined" (a surface
    of a road that goes on from it lies beyond it within STEP_M of its height), "broken" (such surfaces lie beyond it, none
    that near), "missing" (none, and its OSM node is another drivable line's end too) or "dead" (none, its node is its alone:
    a dead end). The map's graph is not read: only where lines are, how wide, which level, and which OSM node their ends are."""
    S = _Scene(tris, cls, lines, classes)

    def inside(p):
        return box is None or (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])

    def add_steps(bad, pts, z_edge, z_beyond, w, kind, line, run):
        cur = None
        for i in range(len(bad)):
            if bad[i]:
                if cur is None:
                    run, cur = run + 1, run + 1
                steps.append(dict(x=float(pts[i][0]), y=float(pts[i][1]), z_edge=float(z_edge[i]), z_beyond=float(z_beyond[i]),
                                  m=float(w[i]), kind=kind, run=cur, line=line))
            else:
                cur = None
        return run

    steps, ends, shared, run = [], [], {}, 0
    for k, r in enumerate(S.roads):
        for n in r["nodes"]:
            if n is not None:
                shared.setdefault(n, set()).add(k)
    for k, r in enumerate(S.roads):
        # a ground road's edges: its carriageway's width either side of its line, against the ground beside it
        for f0, f1 in r["ground"]:
            s0, s1 = f0 * r["length"], f1 * r["length"]
            n = max(int(math.ceil((s1 - s0) / STEP_REACH_M)), 1)
            s = s0 + (np.arange(n) + 0.5) * ((s1 - s0) / n)
            p, t = _pose(r["xy"], r["cum"], s)
            nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
            for side in (-1.0, 1.0):
                e = p + side * r["half"] * nrm
                ze, zb = S.ground.lowest(e), S.ground.lowest(p + side * (r["half"] + STEP_REACH_M) * nrm)
                zi = S.ground.lowest(p + side * (r["half"] - STEP_REACH_M) * nrm)
                bad = np.isfinite(ze) & np.isfinite(zb) & (_kink(ze, zb, zi) >= STEP_M - 1e-6) & np.array([inside(q) for q in e])
                run = add_steps(bad, e, ze, zb, np.full(n, (s1 - s0) / n), "ground", r["id"], run)
        # the ends: of each bridge and long tunnel, and of the line's own end where it lies on the ground
        cand = []
        for a, b, kd in r["st"]:
            if not any(abs(b2 - a) < 1e-9 for _a2, b2, _k2 in r["st"]):
                cand.append((a, -1.0, kd))
            if not any(abs(a2 - b) < 1e-9 for a2, _b2, _k2 in r["st"]):
                cand.append((b, 1.0, kd))
        if r["ground"] and r["ground"][0][0] < 1e-9:
            cand.append((0.0, -1.0, "ground"))
        if r["ground"] and r["ground"][-1][1] > 1.0 - 1e-9:
            cand.append((1.0, 1.0, "ground"))
        for f, sign, kd in cand:
            p, t = _pose(r["xy"], r["cum"], np.array([f * r["length"]]))
            P0, tg = p[0], t[0] * sign
            if not inside(P0):
                continue
            if kd == "ground":
                off, wt, nrm = _lateral(tg, r["half"])
                E = P0[None] + off[:, None] * nrm[None]
                zl = S.ground.lowest(E)
                prof = [(E[j], float(zl[j]), float(wt[j]), tg) for j in range(len(off)) if np.isfinite(zl[j])]
            else:
                prof = _deck_end(S, kd, P0, tg, r["half"])
            if not prof:
                continue
            E = np.array([q[0] for q in prof])
            ze = np.array([q[1] for q in prof])
            wt = np.array([q[2] for q in prof])
            tt = np.array([q[3] for q in prof])
            B, A = E + STEP_REACH_M * tt, E - STEP_REACH_M * tt
            # the height beyond the end its own surface's slope carries on to
            zi = _near_hit(S.deck, A, ze) if kd == "bridge" else (S.tunnel if kd == "tunnel" else S.ground).lowest(A)
            zexp = ze + (ze - np.where(np.isfinite(zi), zi, ze))
            # what goes on from the end: the roads with an end within its width, and the line's own next stretch if it is ground
            cands = [S.surface(B, j, zexp, zexp if kd == "ground" else None) for j in S.continuing(k, P0, r["half"])]
            own_ground = kd != "ground" and any(abs((g0 if sign > 0 else g1) - f) < 1e-9 for g0, g1 in r["ground"])
            if own_ground:
                cands.append(S.surface(B, k, zexp))
            cands = np.array(cands) if cands else np.full((0, len(E)), np.nan)
            if kd != "ground":
                # a step: the ground where the line goes on as ground, the surfaces of the roads that go on, off by STEP_M
                zg = S.ground.lowest(B) if own_ground else np.full(len(E), np.nan)
                ref = np.vstack([cands, zg[None]]) if len(cands) else zg[None]
                have = np.isfinite(ref).any(axis=0)
                dist = np.where(np.isfinite(ref), np.abs(ref - zexp[None]), np.inf).min(axis=0)
                bad = have & (dist >= STEP_M - 1e-6)
                near = np.take_along_axis(ref, np.where(np.isfinite(ref), np.abs(ref - zexp[None]), np.inf).argmin(axis=0)[None], 0)[0]
                zb = np.where(np.isfinite(zg), zg, near)
                run = add_steps(bad, E, ze, zb, wt, kd, r["id"], run)
            touched = np.isfinite(cands).any(axis=0) if len(cands) else np.zeros(len(E), bool)
            matched = bool((np.isfinite(cands) & (np.abs(cands - zexp[None]) < STEP_M - 1e-6)).any()) if len(cands) else False
            how_broken = bool(touched.any())      # (touched, and none within STEP_M)
            node = r["nodes"][0 if sign < 0 else 1] if f in (0.0, 1.0) else None
            other = node is not None and len(shared.get(node, ())) > 1
            how = "joined" if matched else "broken" if how_broken else "missing" if other else "dead"
            c = E.mean(axis=0)
            ends.append(dict(x=float(c[0]), y=float(c[1]), z=float(ze.mean()), width=float(wt.sum()), kind=how, what=kd,
                             line=r["id"], node=node))
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
