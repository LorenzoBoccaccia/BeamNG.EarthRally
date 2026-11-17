#!/usr/bin/env python3
"""Before and after an Earth change: every junction of a set (tools/junctions.json) drawn by tools/earth_view.py from two Earth
trees, one picture per junction and a report of what moved, so a change shows where it lands and where it should not.

    python tools/earth_eval.py --before ../BeamNG.Earth-base --after ../BeamNG.Earth-underpass --out eval-out
    python tools/earth_eval.py --before ... --after ... --out eval-out --only popoli_sr17_aldo_moro,yokohama
    python tools/earth_eval.py --before ... --after ... --out eval-out --at 42.1779,13.8246 --radius 40     (a point of its own)

Each junction is built twice in its own folder (<out>/<name>/before, <out>/<name>/after; earth_view --dump, its own process
per tree, Earth's plan cache off), then:

  <out>/<name>.png      3 x 2: the before views in the top row, the after views under them; by default the overview (1 km
                        round the point in 3D from the south-west, the after one with the ground that rose red and the ground
                        that fell blue, ev.CHANGED_M or more), the close-up from above and from the south-west; the verdict
                        in the title
  <out>/<name>_diff.png the close-up's change seen from straight above: red where the surface rose, blue where it fell, grey
                        where something else is there (a road where the ground was, a deck gone), white where it is the same;
                        and its key with the numbers
  <out>/report.md       one row per junction: same / changed, how much of the surface changed what it is, how far the
                        ground, the roads, the decks and the rails moved (max and 99th percentile, m), and the geometry
                        checks' counts before -> after (earth_checks: vehicles hitting a road or rail not their own, crossings
                        at different levels whose meshes cut each other, in the close-up and in the 1 km overview); a
                        junction where one grows is "worse" whatever else
  <out>/report.json     the same numbers

A junction is "same" when nothing moved more than SAME_M, "minor" when no more than MINOR_KIND of it changed what it is and nothing
moved more than MINOR_M (a rebuild's noise), else "changed". --fail-on-change exits 1 when one did, or got worse, that
is not in --expect (the junctions the change is about). --before-from <a former run's out> takes the before sides from
that run as they are (an upstream capture made once: python tools/earth_eval.py --before <upstream tree> --after <tree>
--out <capture> --labels upstream,latest; then each run --before-from <capture>, only the after tree built), and
tools/eval_page.py makes one page of a run. --reuse draws from the data a former run left (no build), --online
fetches the map data the cache lacks (a junction never built before needs it).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import earth_view as ev  # noqa: E402

SAME_M = 0.02                   # m a surface may move and still be the same (the height lattice's rounding)
MINOR_M = 0.25                  # m ... and still be a minor change (a rebuild's noise, a smoothing pass)
MINOR_KIND = 0.001              # the share of the surface that may change what it is in a minor change
DEFAULT_SET = os.path.join(HERE, "junctions.json")
PANEL = "520x360"               # one view's pixels in the eval picture
SIDES = ("before", "after")
GROUPS = {"ground": (ev.GROUND, ev.MOVED), "road": (ev.ROAD,), "deck": (ev.DECK,), "rail": (ev.RAIL,)}
UP_RGB, DOWN_RGB, KIND_RGB, SAME_RGB = (0.85, 0.1, 0.1), (0.1, 0.25, 0.9), (0.45, 0.45, 0.45), (1.0, 1.0, 1.0)
LIDAR_WAIT_S = "900"            # s each side may wait for a place's lidar (build_side)
DIFF_FULL_M = 2.0               # m of rise or fall drawn at the full colour


def load_set(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["junctions"]


def junction_args(j):
    """earth_view's arguments for junction j (a junctions.json entry)."""
    at = j.get("at")
    if "anchor" in j:
        anchor = j["anchor"]
    elif at:
        anchor = [round(float(at[0]), 2), round(float(at[1]), 2)]
    else:
        raise ValueError(f"{j['name']}: xy needs an anchor")
    out = [f"--anchor={anchor[0]},{anchor[1]}", "--aid", str(int(j.get("aid", 9))), "--radius", str(float(j.get("radius", 40)))]
    out += [f"--at={at[0]},{at[1]}"] if at else [f"--xy={j['xy'][0]},{j['xy'][1]}"]
    if j.get("q"):
        out += ["--q", str(float(j["q"]))]
    for g in j.get("geometry", []):
        out += ["--geometry", g if os.path.isabs(g) else os.path.join(REPO, g)]
    return out


def side_dir(a, name, side):
    """Where junction name's side is drawn: <out>/<name>/<side>, the before side <before-from>/<name>/before when given
    (a former run's, such as the upstream capture: not built again)."""
    if side == "before" and getattr(a, "before_from", None):
        return os.path.join(a.before_from, name, "before")
    return os.path.join(a.out, name, side)


def build_side(j, side, tree, a, log):
    """Draw junction j from Earth tree into its side's folder (side_dir; earth_view, its own process). Returns the error or
    None. The before side taken from --before-from is not built: an error if that run has no such junction."""
    out = side_dir(a, j["name"], side)
    if side == "before" and getattr(a, "before_from", None):
        ok = os.path.isfile(os.path.join(out, "view_surface.npz"))
        log(f"  {side}: from {out}" + ("" if ok else ", missing"))
        return None if ok else f"no before side in {out} (capture it: earth_eval.py --out {a.before_from} ...)"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "earth_view.py"), "--earth", tree, "--out", out, "--name", "view", "--dump",
           "--label-prefix", f"{getattr(a, 'labels', SIDES)[SIDES.index(side)]}: ", "--image", a.image, "--views", a.views] + junction_args(j)
    if side == "after":         # (the overview marks the ground that moved since before)
        cmd += ["--compare-data", os.path.join(side_dir(a, j["name"], "before"), "data")]
    for flag in ("reuse", "online", "no_buildings"):
        if getattr(a, flag):
            cmd.append("--" + flag.replace("_", "-"))
    if a.cache:
        cmd += ["--cache", a.cache]
    # (both sides built under the same conditions: Earth plans a place without its lidar when the lidar's read takes
    # longer than BNE_LIDAR_GAPS_WAIT (8 s), and the first side to build a cold place fetched it while the second found it
    # cached: Boston's Bowker "fell" 4-6 m between two identical trees. The wait pre-cache uses, so neither side gives up)
    env = dict(os.environ, BNGE_PLANCACHE="off", BNGE_BLOCKCACHE="off", BNE_LIDAR_GAPS_WAIT=LIDAR_WAIT_S)
    t = time.time()
    with open(os.path.join(out, "view.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    log(f"  {side}: {time.time() - t:.0f} s{'' if r.returncode == 0 else f', failed ({r.returncode}): see {out}/view.log'}")
    return None if r.returncode == 0 else f"earth_view exited {r.returncode} (see {side}/view.log)"


def compare(before, after, same_m=SAME_M):
    """The numbers of two surface grids (earth_view.surface_grid's (class, height) pairs of the same box): {"kind_changed": the
    fraction of samples whose class changed, "<group>": {"max", "p99", "n"} of |height change| (m) where both are of that group,
    "changed": bool}."""
    (cb, zb), (ca, za) = before, after
    out = {"kind_changed": float(np.mean(cb != ca)) if cb.size else 0.0}
    moved = 0.0
    for name, classes in GROUPS.items():
        m = np.isin(cb, classes) & np.isin(ca, classes) & np.isfinite(zb) & np.isfinite(za)
        dz = np.abs(za[m] - zb[m])
        out[name] = {"max": float(dz.max()) if dz.size else 0.0, "p99": float(np.percentile(dz, 99)) if dz.size else 0.0,
                     "n": int(dz.size)}
        moved = max(moved, out[name]["max"])
    same_kind = (cb == ca) & np.isfinite(zb) & np.isfinite(za)
    if same_kind.any():
        moved = max(moved, float(np.abs(za[same_kind] - zb[same_kind]).max()))
    out["moved"] = moved
    if out["kind_changed"] == 0.0 and moved <= same_m:
        out["verdict"] = "same"
    elif out["kind_changed"] <= MINOR_KIND and moved <= MINOR_M:
        out["verdict"] = "minor"
    else:
        out["verdict"] = "changed"
    out["changed"] = out["verdict"] == "changed"
    return out


def diff_image(before, after, W, H, same_m=SAME_M):
    """The change seen from straight above, an image (H, W, 3 floats) the size of a panel: white the same, red risen and blue
    fallen (full colour at DIFF_FULL_M), grey a sample now something else; north up, the box fitted with its aspect kept."""
    (cb, zb), (ca, za) = before, after
    h, w = cb.shape
    img = np.ones((h, w, 3))
    dz = np.where(np.isfinite(za) & np.isfinite(zb), za - zb, 0.0)
    t = np.clip(np.abs(dz) / DIFF_FULL_M, 0.0, 1.0)[..., None]
    up, down = (dz > same_m)[..., None], (dz < -same_m)[..., None]
    img = np.where(up, (1 - t) * 1.0 + t * np.array(UP_RGB), img)
    img = np.where(down, (1 - t) * 1.0 + t * np.array(DOWN_RGB), img)
    img[cb != ca] = KIND_RGB
    img = img[::-1]                                             # (north up)
    s = min(W / w, H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    yi = np.minimum((np.arange(nh) / s).astype(int), h - 1)
    xi = np.minimum((np.arange(nw) / s).astype(int), w - 1)
    out = np.ones((H, W, 3))
    oy, ox = (H - nh) // 2, (W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = img[yi][:, xi]
    return out


def key_panel(W, H, res):
    """A panel with the diff's key and the junction's numbers."""
    from PIL import Image, ImageDraw
    pic = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(pic)
    f, b = ev._font(15), ev._font(18)
    y = 10
    d.text((10, y), "change, seen from straight above", fill=(0, 0, 0), font=b)
    y += 34
    for rgb, name in ((UP_RGB, f"risen (full colour at {DIFF_FULL_M:g} m)"), (DOWN_RGB, "fallen"),
                      (KIND_RGB, "now something else (class changed)"), (SAME_RGB, f"the same (within {SAME_M:g} m)")):
        d.rectangle([10, y + 2, 38, y + 18], fill=tuple(int(c * 255) for c in rgb), outline=(0, 0, 0))
        d.text((46, y + 1), name, fill=(0, 0, 0), font=f)
        y += 24
    y += 10
    if res.get("error"):
        d.text((10, y), res["error"], fill=(200, 0, 0), font=f)
        return pic
    d.text((10, y), {"changed": "CHANGED", "minor": "minor change"}.get(res["verdict"], "same") + f"; class changed on {100 * res['kind_changed']:.2f} % of it",
           fill=(200, 0, 0) if res["changed"] else (0, 120, 0), font=b)
    y += 30
    for g in GROUPS:
        r = res[g]
        if r["n"]:
            d.text((10, y), f"{g}: moved up to {r['max']:.2f} m (99 %: {r['p99']:.2f} m)", fill=(0, 0, 0), font=f)
            y += 22
    y += 8
    for k, name in CHECKS:
        b, a_ = res.get("checks", {}).get(k, [None, None])
        if b is None and a_ is None:
            continue
        bad = k in res.get("worse", [])
        d.text((10, y), f"{name}: {b} before, {a_} after", fill=(200, 0, 0) if bad else (0, 0, 0), font=f)
        y += 22
    return pic


def verdict_line(res):
    """The junction's result in a line: the verdict and how far each surface moved."""
    if res.get("error"):
        return "error: " + res["error"]
    parts = [f"{g} up to {res[g]['max']:.2f} m" for g in GROUPS if res[g]["n"] and res[g]["max"] > SAME_M]
    ck = res.get("checks", {})
    short = {"vehicles_hit": "vehicles hitting", "crossings_cut": "crossings cutting", "overview_crossings_cut": "in 1 km"}
    hits = ", ".join(f"{short[k]} {b} -> {a}" for k, (b, a) in ck.items() if b is not None or a is not None)
    return (("WORSE, " if res.get("worse") else "") + {"changed": "CHANGED", "minor": "minor change"}.get(res["verdict"], "same") +
            f": class changed on {100 * res['kind_changed']:.2f} % of the close-up" + ("; " + ", ".join(parts) if parts else "") +
            (f"; checks {hits}" if hits else ""))


def eval_picture(j, a, res, log):
    """<out>/<name>.png: the before views over the after views (3 x 2 by default: overview, top, SW), the verdict in the title;
    <out>/<name>_diff.png: the close-up's change from straight above and its key."""
    from PIL import Image
    W, H = (int(v) for v in a.image.lower().split("x"))
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    grey = lambda: Image.new("RGB", (W, H), (230, 230, 230))    # noqa: E731
    panels = []
    for side in SIDES:
        for v in views:
            p = os.path.join(side_dir(a, j["name"], side), f"view_panel_{ev.view_name(v)}.png")
            panels.append(Image.open(p).convert("RGB") if os.path.isfile(p) else grey())
    sub = verdict_line(res) + f"      before: {os.path.basename(a.before)}   after: {os.path.basename(a.after)}"
    title = f"{j['name']}: {j.get('what', '')}"
    used = res.get("_used", [ev.GROUND, ev.ROAD])
    has_line = bool(j.get("geometry")) or any(ev.is_overview(v) for v in views)
    pic = ev.compose(panels, title, sub, ev.legend_entries(used, has_line, True), cols=len(views))
    path = os.path.join(a.out, f"{j['name']}.png")
    pic.save(path)
    log(f"  wrote {path}")
    diff = [Image.fromarray((res["_diff"] * 255 + 0.5).astype(np.uint8)) if "_diff" in res else grey(), key_panel(W, H, res)]
    dpath = os.path.join(a.out, f"{j['name']}_diff.png")
    ev.compose(diff, title + ": the close-up's change", f"before: {a.before}   after: {a.after}", [], cols=2).save(dpath)
    log(f"  wrote {dpath}")
    return path


CHECKS = (("vehicles_hit", "vehicles hitting a road or rail not their own"),
          ("crossings_cut", "crossings at different levels with meshes cutting each other"),
          ("overview_crossings_cut", "the same in the 1 km overview"))


def load_checks(path):
    """earth_view's <name>_checks.json (earth_checks' counts), {} without one."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_checks(before, after):
    """{"checks": {key: [before, after]}, "worse": [keys that grew]}: the geometry checks' counts, before and after; a count
    one side lacks is None and never worse."""
    out = {k: [before.get(k), after.get(k)] for k, _n in CHECKS}
    worse = [k for k, (b, a) in out.items() if b is not None and a is not None and a > b]
    return {"checks": out, "worse": worse}


def load_surface(path):
    with np.load(path, allow_pickle=False) as f:
        return (f["cls"], f["z"]), [int(u) for u in f["used"]]


def run_junction(j, a, log):
    log(f"{j['name']}: {j.get('what', '')}")
    res = {"name": j["name"], "what": j.get("what", "")}
    errs = [e for e in (build_side(j, side, tree, a, log) for side, tree in zip(SIDES, (a.before, a.after))
                        if side in getattr(a, "sides", SIDES)) if e]
    if tuple(getattr(a, "sides", SIDES)) == ("before",):      # (a capture of the before side only: no comparison)
        if errs:
            res["error"] = "; ".join(errs)
        return res
    paths = [os.path.join(side_dir(a, j["name"], side), "view_surface.npz") for side in SIDES]
    if errs or not all(os.path.isfile(p) for p in paths):
        res["error"] = "; ".join(errs) or "no surface grid written"
    else:
        (sb, ub), (sa, ua) = load_surface(paths[0]), load_surface(paths[1])
        if sb[0].shape != sa[0].shape:
            res["error"] = f"the grids differ in size ({sb[0].shape} / {sa[0].shape})"
        else:
            res.update(compare(sb, sa))
            W, H = (int(v) for v in a.image.lower().split("x"))
            res["_diff"] = diff_image(sb, sa, W, H)
            res["_used"] = sorted(set(ub) | set(ua))
    res.update(compare_checks(*(load_checks(os.path.join(side_dir(a, j["name"], side), "view_checks.json")) for side in SIDES)))
    res["picture"] = eval_picture(j, a, res, log)
    return {k: v for k, v in res.items() if not k.startswith("_")}


def flagged_junctions(results):
    """[(name, [(what, count)])] of the junctions whose after side has a vehicle hitting (yellow) or meshes cutting (red), most
    first: the pictures to look at; the others need no look unless they changed."""
    out = []
    for r in results:
        what = [(name, (r.get("checks", {}).get(k) or [None, None])[1]) for k, name in
                (("vehicles_hit", "yellow"), ("crossings_cut", "red"), ("overview_crossings_cut", "red in 1 km"))]
        what = [(n, v) for n, v in what if v]
        if what:
            out.append((r["name"], what))
    return sorted(out, key=lambda t: -sum(v for _n, v in t[1]))


def write_report(results, a):
    rows = ["| junction | result | class changed | ground max / p99 (m) | road max | deck max | rail max | vehicles hitting "
            "| crossings cutting | cutting in 1 km |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("error"):
            rows.append(f"| {r['name']} | error: {r['error']} | | | | | | | | |")
            continue
        g = r["ground"]
        ck = r.get("checks", {})
        cells = []
        for k, _n in CHECKS:
            b, a_ = ck.get(k, [None, None])
            c = "-" if b is None and a_ is None else f"{b} → {a_}"
            cells.append(f"**{c}**" if k in r.get("worse", []) else c)
        result = ("**worse**, " if r.get("worse") else "") + ("**changed**" if r["changed"] else r["verdict"])
        rows.append(f"| {r['name']} | {result} | {100 * r['kind_changed']:.2f} % | "
                    f"{g['max']:.2f} / {g['p99']:.2f} | " + " | ".join(
                        f"{r[k]['max']:.2f}" if r[k]["n"] else "-" for k in ("road", "deck", "rail")) + " | " + " | ".join(cells) + " |")
    flagged = flagged_junctions(results)
    look = (["## Look at these: yellow or red after the change", ""] +
            [f"- **{n}** (`{n}.png`): " + ", ".join(f"{name} {v}" for name, v in what) for n, what in flagged] +
            [""] if flagged else ["No junction has yellow or red after the change.", ""])
    md = [f"# Earth before / after: {len(results)} junctions", "", f"- before: `{a.before}`", f"- after: `{a.after}`", "",
          *look, *rows, "", "Each junction's picture is `<name>.png` (before over after), its close-up's change `<name>_diff.png`.", ""]
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"before": a.before, "after": a.after, "junctions": results}, f, indent=1)
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="the Earth tree before the change (its server/)")
    ap.add_argument("--after", required=True, help="the Earth tree after it")
    ap.add_argument("--out", default="earth-eval")
    ap.add_argument("--set", default=DEFAULT_SET, help="the junctions (default tools/junctions.json)")
    ap.add_argument("--only", help="comma list of the set's junction names to run")
    ap.add_argument("--at", type=ev.parse_pair, help="lat,lon of a point of its own instead of the set")
    ap.add_argument("--radius", type=float, default=40.0, help="m round --at")
    ap.add_argument("--name", default="point", help="--at's name")
    ap.add_argument("--views", default="overview,top,SW", help="the picture's columns (earth_view's views); default "
                    "overview,top,SW: 3 x 2 with before and after")
    ap.add_argument("--image", default=PANEL, help="one view's pixels, WxH")
    ap.add_argument("--reuse", action="store_true", help="draw from the data a former run left (no build)")
    ap.add_argument("--before-from", help="a former run's --out whose before sides are taken as they are (the upstream "
                    "capture): only the after tree is built")
    ap.add_argument("--sides", default="before,after", help="the sides built: before (a capture for --before-from: no "
                    "comparison), or before,after")
    ap.add_argument("--jobs", type=int, default=1, help="junctions built at once (each side its own process)")
    ap.add_argument("--labels", default="before,after", help="the two sides' names on the views, e.g. upstream,latest")
    ap.add_argument("--online", action="store_true", help="fetch the map data the cache lacks")
    ap.add_argument("--no-buildings", action="store_true")
    ap.add_argument("--cache", help="Earth's map data cache (earth_view's default otherwise)")
    ap.add_argument("--fail-on-change", action="store_true", help="exit 1 if a junction not in --expect changed")
    ap.add_argument("--expect", default="", help="comma list of the junctions the change is about")
    a = ap.parse_args(ev.joined_pairs(sys.argv[1:] if argv is None else argv, ("--at",)))
    a.before, a.after, a.out = os.path.abspath(a.before), os.path.abspath(a.after), os.path.abspath(a.out)
    a.before_from = os.path.abspath(a.before_from) if a.before_from else None
    a.labels = [s.strip() for s in a.labels.split(",")]
    a.sides = [s.strip() for s in a.sides.split(",") if s.strip()]
    if a.sides not in (["before"], ["before", "after"]):
        sys.exit("--sides: before, or before,after")
    if len(a.labels) != 2 or not all(a.labels):
        sys.exit("--labels: two names, before,after")
    os.makedirs(a.out, exist_ok=True)

    if hasattr(sys.stdout, "reconfigure"):      # (the report's arrows on a redirected Windows console: cp1252 could not)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def log(msg):
        print(msg, flush=True)

    if a.at:
        junctions = [{"name": a.name, "what": f"the point {a.at[0]}, {a.at[1]}", "at": list(a.at), "radius": a.radius}]
    else:
        junctions = load_set(a.set)
        if a.only:
            want = {s.strip() for s in a.only.split(",") if s.strip()}
            unknown = want - {j["name"] for j in junctions}
            if unknown:
                sys.exit(f"not in {a.set}: {', '.join(sorted(unknown))}")
            junctions = [j for j in junctions if j["name"] in want]
    if a.jobs > 1:                  # (each side is its own earth_view process: junctions side by side)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.jobs) as ex:
            results = list(ex.map(lambda j: run_junction(j, a, lambda m, n=j["name"]: log(f"[{n}] {m.strip()}")), junctions))
    else:
        results = [run_junction(j, a, log) for j in junctions]
    if a.sides == ["before"]:
        bad = [r["name"] for r in results if r.get("error")]
        log(f"captured {len(results) - len(bad)} of {len(results)} before sides" + (f"; failed: {', '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    log(write_report(results, a))
    expect = {s.strip() for s in a.expect.split(",") if s.strip()}
    bad = [r["name"] for r in results if (r.get("changed") or r.get("worse")) and r["name"] not in expect]
    if a.fail_on_change and bad:
        log(f"changed and not expected: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
