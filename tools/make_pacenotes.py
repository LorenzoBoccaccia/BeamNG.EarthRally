#!/usr/bin/env python3
"""Precompute the pace notes of an edition's special stages, offline, from the stage data and the ground BeamNG.Earth builds there.

The game does no scanning of its own: it places the notes stored here (legs.materialize, pacenotes.fromStored). Only offline
is the whole stage's ground known: in the game Earth has built the terrain block round the car and no further.

For each special (legs.json, type "special") of the edition:

  1. the line     the curated line of the game (curate.line on the geometry's points and structures, through lupa: the same
                  Lua the game runs), every curate.STEP m
  2. the ground   Earth's ground under the line, every SAMPLE_M m, as its physics block computes it: the DEM, Earth's own bumps,
                  runways and sea floor, every road graded in, the ground over its tunnels and their low arches (the chain of
                  tools/earth_trace.py: Earth's own modules on its disk cache, read-only)
  3. the shaping  the spikes the game takes out of that ground along the line (shape.lua: curate.despike, bumps only, with the
                  edition's shaping.spikeM, and the points within TAPER_M of one laid on the straight line between the points
                  either side), not beside a tunnel (the geometry's tunnelsBeside: Earth's ground there is kept as it made it)
  4. the profile  that ground smoothed over SMOOTH_M (what a car's wheels and suspension feel of it), and its crests
                  (curate.crests: the convex bends, each with the speed that lifts a car off)
  5. the notes    pacenotes.notebook on the line at that ground: every corner (and "don't cut" over the geometry's bridges and
                  along its sidewalks), each crest called by how near the speed the corners allow there comes to lifting the car

and writes them into the special's geometry file as "pacenotes": { "ground": "BeamNG.Earth <version>", "made": <date>,
"sampleM", "smoothM", "notes": [ { "items" (the stock structured items, slot -> item), "earthrally" (our metadata: the corner, the crest,
"don't cut"), "start", "startAhead", "end", "endAhead" } ] }, each point [lat, lon, h] (the waypoints of the corner start and end,
and a point AHEAD_M further along the road: its direction).

    python tools/make_pacenotes.py --earth ../BeamNG.Earth editions/svolte-di-popoli [leg id ...]
    python tools/make_pacenotes.py --earth ../BeamNG.Earth --online editions/testbed-algarve

--online fetches what Earth's cache lacks (the DEM and the roads of the stage's area); without it a missing tile is an error.
Needs lupa and numpy, and Earth's Python requirements.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EARTHRALLY_EXT = os.path.join(ROOT, "mod", "beamng_earthrally", "lua", "ge", "extensions")
sys.path.insert(0, HERE)

SAMPLE_M = 2.0          # the ground is measured this often along the line
SMOOTH_M = 5.0          # ... and smoothed over this (gaussian sigma): a car's wheelbase and suspension, not the DEM's grain
CHUNK_M = 800.0         # Earth's ground is computed this much of the line at a time (the tunnels' arches of the box round it)
CHUNK_MARGIN_M = 300.0  # ... the box reaching this far beyond the line
AHEAD_M = 5.0           # a waypoint's direction: the point this far further along the road
HM_ZOOM = 16            # the zoom of Earth's physics block DEM
AID = 7                 # the first anchor id Earth's caches are keyed by (any: the heights are geodetic) ...
REANCHOR_M = 15000.0    # ... and a new one when the line is this far from it (Earth's frame is a plane: a block's reach)
TAPER_M = 15.0          # shape.lua's: the points within this of a spike are fixed with it
MAX_CHANGE_M = 4.0      # shape.lua's: a fix moves the ground no further than this (the whole fix scaled to it)
SPIKE_M = 2.0           # shape.lua's default (the edition's shaping.spikeM overrides it)


# --- Lua (the game's own modules) ---------------------------------------------------------------------------------------

def lua_runtime(earth_dir):
    import lupa
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("math.atan2 = math.atan2 or function(y, x) return math.atan(y, x) end")
    earth_ext = os.path.join(earth_dir, "mod", "beamng_earth", "lua", "ge", "extensions")
    rt.execute("package.path = package.path .. ';%s/?.lua;%s/?.lua'" % (
        EARTHRALLY_EXT.replace("\\", "/"), earth_ext.replace("\\", "/")))
    return rt


def to_lua(rt, v):
    if isinstance(v, dict):
        t = rt.table()
        for k, x in v.items():
            t[k] = to_lua(rt, x)
        return t
    if isinstance(v, (list, tuple)):
        t = rt.table()
        for i, x in enumerate(v):
            t[i + 1] = to_lua(rt, x)
        return t
    return v


def from_lua(v):
    """A Lua table as a list (keys 1..n) or a dict (string keys), all the way down."""
    if not hasattr(v, "items") or isinstance(v, (str, bytes)):
        return v
    d = {k: from_lua(x) for k, x in v.items()}
    if d and all(isinstance(k, int) for k in d) and sorted(d) == list(range(1, len(d) + 1)):
        return [d[k] for k in range(1, len(d) + 1)]
    return {str(k): x for k, x in d.items()}


def lua_list(t):
    return [t[i] for i in range(1, len(t) + 1)] if t else []


# --- Earth's ground -----------------------------------------------------------------------------------------------------

class EarthGround:
    """Earth's physics-block ground (geodetic h) at lat / lon points, as tools/earth_trace.py rebuilds it. Anchored, as the game is, near
    the points asked for: at the first of them, again whenever they are REANCHOR_M from the anchor."""

    def __init__(self, earth_dir, cache, online):
        import earth_trace
        earth_trace.load_earth(earth_dir)
        import blockworker
        import geodesy
        import grading
        import roads as roadsmod
        self.bw, self.grading, self.roads, self.geodesy = blockworker, grading, roadsmod, geodesy
        blockworker._s = blockworker._State(cache, not online, True, True, True, False, airports=True, sea=True)
        self.S = blockworker._s
        self.aid, self.frame = AID - 1, None
        self.version = earth_version(earth_dir)
        self.cell_fn = lambda key: self.S.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")  # noqa: E731

    def anchor(self, lat0, lon0):
        self.aid += 1
        self.frame = self.geodesy.EnuFrame(lat0, lon0)
        self.g, _ = self.S.anchor(self.aid, lat0, lon0)
        if self.g is None:
            raise SystemExit("Earth gave no grader (roads or grading off)")
        self.tun = self.bw.tunnel_profiles(self.aid, self.frame, self._graded, self.g)

    def _near(self, la, lo):
        if self.frame is None:
            return False
        e, n, _ = self.frame.geodetic_to_enu(la, lo, 0.0)
        return float(np.max(np.hypot(np.asarray(e), np.asarray(n)))) < REANCHOR_M

    def _graded(self, la, lo):
        z = self.S.dem.sample(la, lo, HM_ZOOM)
        z = self.bw.ground(self.frame, la, lo, z)
        return self.g.apply(la, lo, z, self.grading.ROAD_LIFT)

    def at(self, la, lo):
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        if not self._near(la, lo):
            self.anchor(float(la[0]), float(lo[0]))
        e, n, _ = self.frame.geodetic_to_enu(la, lo, 0.0)
        e, n = np.asarray(e), np.asarray(n)
        bx = (float(e.min()) - CHUNK_MARGIN_M, float(n.min()) - CHUNK_MARGIN_M,
              float(e.max()) + CHUNK_MARGIN_M, float(n.max()) + CHUNK_MARGIN_M)
        z = self._graded(la, lo)
        return np.asarray(self.roads.block_arch_raise(self.tun, self.cell_fn, self.frame, la, lo, z, bx), np.float64)


def earth_version(earth_dir):
    try:
        with open(os.path.join(earth_dir, "VERSION"), encoding="utf-8") as f:
            return "BeamNG.Earth " + f.read().strip()
    except OSError:
        return "BeamNG.Earth"


# --- the profile --------------------------------------------------------------------------------------------------------

def gaussian(h, sigma_samples):
    if sigma_samples <= 0:
        return h.copy()
    r = int(math.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(r, h[0]), h, np.full(r, h[-1])])
    return np.convolve(pad, k, mode="valid")


def shaped(curate, rt, G, s, held, spike_m):
    """The ground the game leaves along the line (shape.lua, the spikes only): G at the curated points (s along the line), the points
    held (beside a tunnel) left as they are and breaking the runs. Returns the fixed heights."""
    out = np.array(G, np.float64)
    n = len(G)
    i = 0
    while i < n:
        if held[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not held[j + 1]:
            j += 1
        if j - i >= 2:
            flagged = lua_list(curate.despike(to_lua(rt, [float(x) for x in G[i:j + 1]]),
                                              to_lua(rt, {"SPIKE_M": spike_m, "BUMPS_ONLY": True}))[0])
            core = np.zeros(n, bool)
            for f in flagged:
                q = i + int(f) - 1
                m = q
                while m >= i + 1 and s[q] - s[m] <= TAPER_M:
                    core[m] = True
                    m -= 1
                m = q + 1
                while m <= j - 1 and s[m] - s[q] <= TAPER_M:
                    core[m] = True
                    m += 1
            q = i
            while q <= j:
                if core[q]:
                    a = q
                    while q + 1 <= j and core[q + 1]:
                        q += 1
                    b = q
                    lo_, hi_ = (a - 1 if a - 1 >= i else None), (b + 1 if b + 1 <= j else None)
                    idx = np.arange(a, b + 1)
                    if lo_ is not None and hi_ is not None:
                        want = G[lo_] + (G[hi_] - G[lo_]) * (s[idx] - s[lo_]) / (s[hi_] - s[lo_])
                    else:
                        want = np.full(len(idx), G[lo_ if lo_ is not None else hi_])
                    d = want - G[idx]
                    peak = float(np.max(np.abs(d))) if len(d) else 0.0
                    if peak > MAX_CHANGE_M:
                        d *= MAX_CHANGE_M / peak
                    out[idx] = G[idx] + d
                q += 1
        i = j + 1
    return out


EARTH_R = 6371008.8


def developed(lat, lon):
    """The line unrolled into a plane: x, y (m) of each point, every segment at its true length (on the sphere) and its bearing."""
    la, lo = np.radians(lat), np.radians(lon)
    dy = np.diff(la) * EARTH_R
    dx = np.diff(lo) * EARTH_R * np.cos((la[1:] + la[:-1]) / 2)
    return np.concatenate([[0.0], np.cumsum(dx)]), np.concatenate([[0.0], np.cumsum(dy)])


def along(X, Y, cum, x, y):
    """Metres along the polyline (X, Y; cum: metres along at each point) of its point nearest (x, y)."""
    ax, ay, bx, by = X[:-1], Y[:-1], X[1:], Y[1:]
    dx, dy = bx - ax, by - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / l2, 0.0, 1.0)
    d2 = (ax + t * dx - x) ** 2 + (ay + t * dy - y) ** 2
    k = int(np.argmin(d2))
    return float(cum[k] + t[k] * (cum[k + 1] - cum[k]))


def held_points(geo, s):
    held = np.zeros(len(s), bool)
    for t in geo.get("tunnelsBeside") or []:
        held |= (s >= t["from"]) & (s <= t["to"])
    return held


# --- one special --------------------------------------------------------------------------------------------------------

def special_notes(rt, mods, ground, geo, spike_m, sample_m=SAMPLE_M, smooth_m=SMOOTH_M, log=print, cache_dir=None):
    curate, pacenotes, legs, geodesy = mods
    got = curate.line(to_lua(rt, geo["points"]), to_lua(rt, {"structures": geo.get("structures") or []}))
    r = got[0] if isinstance(got, tuple) else got
    if r is None:
        raise SystemExit("the line cannot be curated: %s" % (got[1] if isinstance(got, tuple) else "?"))
    pts = [tuple(p[k] for k in (1, 2, 3)) for p in lua_list(r.points)]
    s = np.array([float(x) for x in lua_list(r.s)])
    lat, lon = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    length = float(s[-1])
    # the ground every sample_m along the line, Earth's, computed CHUNK_M at a time
    S = np.arange(0.0, length, sample_m)
    S = np.append(S, length)
    la, lo = np.interp(S, s, lat), np.interp(S, s, lon)
    # (kept in cache_dir by the points asked for and the Earth that answered: the calls can be made again without Earth's work)
    path = None
    if cache_dir:
        key = hashlib.sha1(np.round(np.stack([la, lo]), 7).tobytes() + str(getattr(ground, "version", "")).encode()).hexdigest()
        path = os.path.join(cache_dir, "ground-%s.npy" % key[:20])
    if path and os.path.isfile(path):
        gnd = np.load(path)
    else:
        gnd = np.empty(len(S))
        step = max(1, int(CHUNK_M / sample_m))
        for a in range(0, len(S), step):
            gnd[a:a + step] = ground.at(la[a:a + step], lo[a:a + step])
            if length > 50000 and int(S[a] // 25000) != int(S[min(a + step, len(S) - 1)] // 25000):
                log("  ... %.0f of %.0f km" % (S[min(a + step, len(S) - 1)] / 1000, length / 1000))
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(path, gnd)
    # the game's shaping at the curated points: its delta spread over the samples between them
    G = np.interp(s, S, gnd)
    fixed = shaped(curate, rt, G, s, held_points(geo, s), spike_m)
    gnd = gnd + np.interp(S, s, fixed - G)
    prof = gaussian(gnd, smooth_m / sample_m)
    # the crests, outside the curated line's spans (a deck or a tunnel floor is straight)
    spans = []
    for sp in lua_list(r.spans):
        a, b = float(s[int(sp.i0) - 1]), float(s[int(sp.i1) - 1])
        spans.append({"i0": int(np.searchsorted(S, a)) + 1, "i1": int(np.searchsorted(S, b, side="right"))})
    crests = curate.crests(to_lua(rt, [float(x) for x in prof]), to_lua(rt, [float(x) for x in S]), sample_m, to_lua(rt, spans))
    # the notes, on the curated line as the race path and the driveline are (the progress bar puts the waypoints into that route)
    # (scanned on the line unrolled into a plane, each segment at its true length and bearing: one world frame for a stage of
    # hundreds of kilometres is kilometres under the ground at its far end; the notes go back to lat / lon by the metres along it)
    hs = np.array([p[2] for p in pts])
    X, Y = developed(lat, lon)
    w = to_lua(rt, [[float(x), float(y), float(h)] for x, y, h in zip(X, Y, hs)])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    opts = legs.noteOptions(to_lua(rt, geo), r)
    opts.crests = crests
    opts.lineLength = length
    nb, corners = pacenotes.notebook(w, opts)
    log("  " + pacenotes.summary(corners, float(cum[-1])))
    notes = []
    for note in lua_list(nb.pacenotes):
        wps = {wp.waypointType: wp for wp in lua_list(note.pacenoteWaypoints)}

        def geo_of(wp, ahead=0.0):
            at = along(X, Y, cum, wp.pos[1], wp.pos[2]) + ahead
            return [round(float(np.interp(at, cum, lat)), 7), round(float(np.interp(at, cum, lon)), 7),
                    round(float(np.interp(at, cum, hs)), 2)]
        items = from_lua(note.structured["items"])
        notes.append({
            "items": {str(k): v for k, v in sorted(items.items())} if isinstance(items, dict) else
            {str(k + 1): v for k, v in enumerate(items)},
            "earthrally": from_lua(note.metadata.earthrally) or {},
            "start": geo_of(wps["cornerStart"]), "startAhead": geo_of(wps["cornerStart"], AHEAD_M),
            "end": geo_of(wps["cornerEnd"]), "endAhead": geo_of(wps["cornerEnd"], AHEAD_M),
        })
    for c in lua_list(crests):
        log("    crest at %6.0f m: %.4f 1/m, lifts at %3.0f km/h, grade %+.0f%% -> %+.0f%%" % (
            c.s, c.curvature, c.liftMs * 3.6, c.gradeIn * 100, c.gradeOut * 100))
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edition", help="the edition's folder (legs.json, geometry/)")
    ap.add_argument("legs", nargs="*", help="the specials to do (default: all)")
    ap.add_argument("--earth", default=os.environ.get("BEAMNG_EARTH_DIR") or os.path.join(os.path.dirname(ROOT), "BeamNG.Earth"),
                    help="BeamNG.Earth's source tree (its server/ and mod/)")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth"))
    ap.add_argument("--online", action="store_true", help="fetch what Earth's cache lacks")
    ap.add_argument("--dry-run", action="store_true", help="say what would be written, write nothing")
    ap.add_argument("--ground-cache", default=os.path.join(HERE, ".cache", "pacenotes"),
                    help="where the ground measured along each line is kept (the calls made again without Earth's work)")
    ap.add_argument("--no-cache", action="store_true", help="measure the ground again")
    a = ap.parse_args()
    earth = os.path.abspath(a.earth)
    with open(os.path.join(a.edition, "legs.json"), encoding="utf-8") as f:
        edition = json.load(f)
    spike_m = float((edition.get("shaping") or {}).get("spikeM") or SPIKE_M)
    want = a.legs or [l["id"] for l in edition["legs"] if l["type"] == "special"]
    rt = lua_runtime(earth)
    mods = tuple(rt.eval("(require('%s'))" % m) for m in
                 ("beamngEarthRally/curate", "beamngEarthRally/pacenotes", "beamngEarthRally/legs", "beamngEarth/geodesy"))
    ground = None
    for lid in want:
        path = os.path.join(a.edition, "geometry", lid + ".json")
        with open(path, encoding="utf-8") as f:
            geo = json.load(f)
        print("%s (%.1f km):" % (lid, geo.get("lengthM", 0) / 1000.0), flush=True)
        if ground is None:
            ground = EarthGround(earth, a.cache, a.online)
        notes = special_notes(rt, mods, ground, geo, spike_m, cache_dir=None if a.no_cache else a.ground_cache)
        geo["pacenotes"] = {"ground": earth_version(earth), "made": datetime.date.today().isoformat(),
                            "sampleM": SAMPLE_M, "smoothM": SMOOTH_M, "notes": notes}
        print("  %d notes" % len(notes) + (" (not written)" if a.dry_run else " -> " + path), flush=True)
        if not a.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(geo, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
