#!/usr/bin/env python3
"""EarthRally's ground grader (shape.lua, with crossing.lua and curate.lua) run offline on a terrain block: where it moves the ground.

The game shapes the ground along the special stages at run time, on the block Earth sent (shape.lua: spikes of the line, the ground under an
overpass, the ground at a deck's ends, a road across a tunnel). Nothing of that is in Earth's own build, so earth_writers.py --rally hands
the block it built to this module, which runs the mod's own Lua (lupa, as the tests do) on it as the game would:

  * the lines: each stage's geometry (editions/<edition>/geometry/<leg>.json) curated by curate.line, in the anchor frame, with its
    bridges and tunnels skipped (curate.spanAt), its points beside a tunnel kept as Earth made them (tunnelsBeside) and its other
    roads (otherRoads), as core.lua's shapeLines;
  * the tunnels: the points the game's tunnel lights stand at (BeamNG.Earth's ambience.tunnelLights: along each tunnel box, one per 20 m,
    at the middle of its share), here along each tunnel piece of the map's roads (Earth's roads.tunnel_parts) instead of its sound boxes;
  * no colliders: the game's rays for decks and overpasses (env.deck) find nothing offline, so a deck's ends and an overpass the colliders
    show are not shaped here. The log says so.

Returns the heights after the grader, its log, and per point of each line what the grader made of it (kept beside a tunnel, a span, ...).
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT = os.path.join(ROOT, "mod", "beamng_earthrally", "lua", "ge", "extensions").replace("\\", "/")
LIGHT_M = 20.0          # (ambience.TUNNEL_LIGHT_M)

PRELUDE = r"""
math.atan2 = math.atan2 or function(y, x) return math.atan(y, x) end
package.path = package.path .. ';%(ext)s/?.lua'
curate = require('beamngEarthRally/curate')
shape = require('beamngEarthRally/shape')
T = 0
function now() T = T + 0.002; return T end
LOGS = {}
function newBlock(px, py, n, H)
  local tb = { px = px, py = py, n = n, h = H }
  function tb:getPosition() return { x = self.px, y = self.py, z = 0 } end
  function tb:getSquareSize() return 1 end
  function tb:getWorldBox() local s = self; return { getExtents = function() return { x = s.n, y = s.n, z = 100 } end } end
  function tb:getHeightGrid(gx, gy) return self.h[gx + gy * self.n] end
  function tb:setHeightGrid(gx, gy, v) self.h[gx + gy * self.n] = v end
  function tb:updateGrid(a, b) end
  return tb
end
"""


def tunnel_points(frame, box):
    """The game's tunnel light points for the tunnels whose pieces come near box (x0, y0, x1, y1, anchor frame): [(x, y, z)]."""
    import blockworker
    import roads
    S = blockworker._s
    x0, y0, x1, y1 = box
    keys = set()
    for tx in range(int(math.floor(x0 / roads.TILE)), int(math.floor(x1 / roads.TILE)) + 1):
        for ty in range(int(math.floor(y0 / roads.TILE)), int(math.floor(y1 / roads.TILE)) + 1):
            keys.update(roads.cells_of_tile(frame, tx, ty, margin=60.0))
    out, seen = [], set()
    for k in sorted(keys):
        feats = S.md.cell(k, layers=("roads",), kind="demand").layers.get("roads")
        if feats is None or not len(feats):
            continue
        specs = roads.tunnel_specs(feats)
        for i, sp in enumerate(specs):
            if sp is None:
                continue
            for piece in roads.tunnel_parts(feats, i):
                ll = np.asarray(piece, np.float64)
                if len(ll) < 2:
                    continue
                key = tuple(np.round(ll[[0, -1]].ravel(), 7))
                if key in seen:
                    continue
                seen.add(key)
                e, n, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
                xy = np.column_stack([np.asarray(e), np.asarray(n)])
                seg = np.hypot(*np.diff(xy, axis=0).T)
                arc = np.concatenate([[0.0], np.cumsum(seg)])
                L = float(arc[-1])
                if L < 1e-3 or xy[:, 0].max() < x0 - 100 or xy[:, 0].min() > x1 + 100 or xy[:, 1].max() < y0 - 100 or xy[:, 1].min() > y1 + 100:
                    continue
                m = max(1, int(math.floor(L / LIGHT_M + 0.5)))
                for q in range(m):
                    t = (q + 0.5) / m * L
                    out.append((float(np.interp(t, arc, xy[:, 0])), float(np.interp(t, arc, xy[:, 1])), 0.0))
    return out


def run(heights, x0, y0, frame, geometries, tunnels, shaping=None, enabled=True):
    """heights: (n, n) absolute heights of the block's samples (row j is y0 + j, column i is x0 + i), anchor frame z. geometries: { leg id:
    geometry json dict } of the special stages. tunnels: tunnel_points(). Returns { heights, logs, lines: [ { leg, s, x, y, kept, skip } ],
    stretches }."""
    import lupa
    n = heights.shape[0]
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    rt.execute(PRELUDE % {"ext": EXT})
    G = rt.globals()
    flat = heights.ravel()
    H = rt.table_from({int(k): float(v) for k, v in enumerate(flat)})
    tb = G.newBlock(float(x0), float(y0), n, H)
    orig = heights.copy()

    def ground(x, y):           # the game's height query: the block's own ground (bilinear)
        fx, fy = x - x0, y - y0
        i, j = int(math.floor(fx)), int(math.floor(fy))
        if i < 0 or j < 0 or i + 1 >= n or j + 1 >= n:
            return None
        u, v = fx - i, fy - j
        return float(orig[j, i] * (1 - u) * (1 - v) + orig[j, i + 1] * u * (1 - v) + orig[j + 1, i] * (1 - u) * v + orig[j + 1, i + 1] * u * v)

    lt = rt.table_from([rt.table_from([float(x), float(y), float(z)]) for x, y, z in tunnels]) if tunnels else None
    logs = []
    env = rt.table_from({
        "block": lambda: tb, "now": G.now, "clock": lambda: G.T / 1000.0, "player": lambda: None,
        "ground": ground, "tunnels": (lambda: lt) if lt is not None else None,
        "log": lambda level, msg: logs.append(f"{level} {msg}"), "vec3": lambda x, y, z: rt.table_from({"x": x, "y": y, "z": z}),
    })
    E = G.shape.new(env)
    lines, info = [], []
    for leg, g in geometries.items():
        pts = rt.table_from([rt.table_from([float(v) for v in p]) for p in g["points"]])
        opts = rt.table_from({"structures": rt.table_from([rt.table_from(s) for s in g.get("structures") or []])})
        r = G.curate.line(pts, opts)
        if isinstance(r, tuple):
            r = r[0]
        if not isinstance(r, type(pts)):
            logs.append(f"W {leg}: not curated, left out")
            continue
        rp = [[float(r.points[k][q] or 0) for q in (1, 2, 3)] for k in range(1, len(r.points) + 1)]
        s = [float(r.s[k]) for k in range(1, len(r.points) + 1)] if r.s else None
        ll = np.asarray(rp, np.float64)
        e, nn, _ = frame.geodetic_to_enu(ll[:, 0], ll[:, 1], 0.0)
        e, nn = np.asarray(e), np.asarray(nn)
        skip = {}
        for i in range(1, len(rp)):
            sp = G.curate.spanAt(r, i)
            if sp:
                skip[i] = "tagged" if sp.tagged else True
        keep = {}
        for i in range(1, len(rp) + 1):
            for t in g.get("tunnelsBeside") or []:
                if s and t["from"] <= s[i - 1] <= t["to"]:
                    keep[i] = True
                    break
        others = []
        for o in g.get("otherRoads") or []:
            op = np.asarray(o["pts"], np.float64)
            oe, on, _ = frame.geodetic_to_enu(op[:, 0], op[:, 1], 0.0)
            others.append(rt.table_from({"pts": rt.table_from([rt.table_from([float(a), float(b)]) for a, b in zip(oe, on)]),
                                         "hw": float(o.get("hw") or 3)}))
        line = {"pts": rt.table_from([rt.table_from([float(a), float(b), float(p[2])]) for a, b, p in zip(e, nn, rp)]),
                "skip": rt.table_from(skip), "others": rt.table_from(others)}
        if keep:
            line["keep"] = rt.table_from(keep)
        if shaping and shaping.get("spikeM"):
            line["spikeM"] = float(shaping["spikeM"])
        lines.append(rt.table_from(line))
        info.append({"leg": leg, "s": s, "x": e, "y": nn, "kept": [bool(keep.get(i)) for i in range(1, len(rp) + 1)],
                     "skip": [skip.get(i) for i in range(1, len(rp) + 1)]})
    E.setEnabled(E, bool(enabled))
    E.setLines(E, rt.table_from(lines))
    E.blockChanged(E, False)
    frames = 0
    while E.busy(E) and frames < 100000:
        E.update(E, 1e6)
        frames += 1
    out = np.array([H[k] for k in range(n * n)], np.float64).reshape(n, n)
    return {"heights": out, "logs": logs, "lines": info, "frames": frames}


def load_geometries(paths):
    """{ leg id: geometry } of the stage geometry files given (special stages and liaisons alike: the game shapes the specials)."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        out[g.get("id") or os.path.splitext(os.path.basename(p))[0]] = g
    return out
