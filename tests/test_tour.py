"""Offline tests for mod/beamng_earthrally/.../tour.lua (the junction tour) in LuaJIT (lupa): the drive picked through a
junction on a navgraph, its levels, and the tour's steps against a fake game.

    python -m pytest tests/test_tour.py
"""
import json
import os

import pytest

lupa = pytest.importorskip("lupa")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EARTH = os.environ.get("BEAMNG_EARTH_DIR") or os.path.join(os.path.dirname(ROOT), "BeamNG.Earth")
EARTH_EXT = os.path.join(EARTH, "mod", "beamng_earth", "lua", "ge", "extensions")
EARTHRALLY_EXT = os.path.join(ROOT, "mod", "beamng_earthrally", "lua", "ge", "extensions")


@pytest.fixture()
def lua():
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("math.atan2 = math.atan2 or function(y, x) return math.atan(y, x) end")
    rt.execute("package.path = package.path .. ';%s/?.lua;%s/?.lua'" % (
        EARTH_EXT.replace("\\", "/"), EARTHRALLY_EXT.replace("\\", "/")))
    rt.execute("tour = (require('beamngEarthRally/tour'))")
    # a navgraph builder: road(prefix, points, oneWay, z): nodes prefix1..n along points, linked in order
    rt.execute("""
      nodes = {}
      function road(prefix, pts, oneWay, z)
        local prev
        for i, p in ipairs(pts) do
          local n = prefix .. i
          if p.name then n = p.name end
          nodes[n] = nodes[n] or { pos = { x = p[1], y = p[2], z = z or 0 }, links = {} }
          if prev then
            nodes[prev].links[n] = { drivability = 1, oneWay = oneWay or false, inNode = prev }
          end
          prev = n
        end
      end
      function line(x0, y0, x1, y1, step, join)   -- points every step from (x0, y0) to (x1, y1); join names the one at J
        local out, L = {}, math.sqrt((x1 - x0) ^ 2 + (y1 - y0) ^ 2)
        local n = math.floor(L / step + 0.5)
        for i = 0, n do
          local x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
          local p = { x, y }
          if join and math.abs(x) < 1e-6 and math.abs(y) < 1e-6 then p.name = join end
          out[#out + 1] = p
        end
        return out
      end
    """)
    return rt


def names(t):
    return [t[i] for i in range(1, len(t) + 1)]


def pos(lua, n):
    p = lua.eval("nodes")[n].pos
    return (p.x, p.y, p.z)


def test_a_t_junction_is_driven_straight_through(lua):
    lua.execute("""
      road('w', line(-300, 0, 0, 0, 20, 'J'))
      road('e', line(0, 0, 300, 0, 20, 'J'))
      road('s', line(0, 0, 0, 300, 20, 'J'))      -- (the side road north)
    """)
    dv = lua.eval("tour.pick(nodes, 'J')")
    a, b = pos(lua, dv.names[1]), pos(lua, dv.names[len(dv.names)])
    assert "J" in names(dv.names)
    assert abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6, "on the through road, not the side road"
    assert abs(a[0]) > 100 and abs(b[0]) > 100 and a[0] * b[0] < 0, "from well before it to well past it"
    xs = [pos(lua, n)[0] for n in names(dv.names)]
    assert xs in (sorted(xs), sorted(xs, reverse=True)), "one way along, no U turn"


def test_a_one_way_road_is_driven_its_way(lua):
    lua.execute("""
      road('a', line(300, 0, 0, 0, 20, 'J'), true)     -- one way, east to west
      road('b', line(0, 0, -300, 0, 20, 'J'), true)
    """)
    dv = lua.eval("tour.pick(nodes, 'J')")
    a, b = pos(lua, dv.names[1]), pos(lua, dv.names[len(dv.names)])
    assert a[0] > 100 and b[0] < -100


def test_an_overpass_is_driven_on_both_levels(lua):
    lua.execute("""
      road('w', line(-300, 0, 0, 0, 20, 'J'))
      road('e', line(0, 0, 300, 0, 20, 'J'))
      road('n', line(0, -300, 0, 300, 20), false, 7)   -- (over it, 7 m up, not linked)
    """)
    drives = lua.eval("tour.drives(nodes, { x = 0, y = 0, z = 0 }, 40)")
    assert len(drives) == 2
    zs = {pos(lua, drives[1].root)[2], pos(lua, drives[2].root)[2]}
    assert zs == {0, 7}


def test_the_tour_drives_each_junction_and_ends(lua):
    lua.execute("""
      road('w', line(-300, 0, 0, 0, 20, 'J'))
      road('e', line(0, 0, 300, 0, 20, 'J'))
      calls = {}
      local car = { pos = { x = 0, y = 0, z = 0 }, speed = 10 }
      local anchor = {
        geodeticToWorld = function(self, lat, lon, h) return 0, 0, 0 end,
        worldToGeodetic = function(self, x, y, z) return 42 + y / 111320, 13 + x / 82000, 0 end,
      }
      ctx = {
        anchor = function() return anchor end,
        ready = function() return true end,
        teleport = function(lat, lon, name) calls[#calls + 1] = 'teleport ' .. name; return true end,
        roadsReady = function() return true end,
        nodes = function() return nodes end,
        car = function() return car end,
        place = function(p, d) calls[#calls + 1] = string.format('place %.0f', p.x); car.pos = { x = p.x, y = p.y, z = p.z } end,
        drive = function(names, dest, label)
          calls[#calls + 1] = 'gps ' .. label
          calls[#calls + 1] = 'drive ' .. #names
          local e = nodes[names[#names]].pos
          car.pos = { x = e.x, y = e.y, z = e.z }         -- (it gets there)
        end,
        stopAi = function() calls[#calls + 1] = 'stop' end,
        releaseAi = function() calls[#calls + 1] = 'release' end,
        gpsClear = function() calls[#calls + 1] = 'gps clear' end,
        say = function() end,
        log = function() end,
      }
      tour.start(ctx, { { name = 'one', lat = 42, lon = 13 }, { name = 'two', lat = 42, lon = 13 } })
      for i = 1, 400 do
        if not tour.running() then break end
        if i == 2 or i == 40 then tour.onAnchor() end
        tour.update(0.25)
      end
    """)
    calls = names(lua.eval("calls"))
    assert lua.eval("tour.running()") is False
    assert calls[0] == "teleport Junction: one"
    assert [c for c in calls if c.startswith("teleport")] == ["teleport Junction: one", "teleport Junction: two"]
    assert sum(c.startswith("drive") for c in calls) == 2
    first_drive = next(i for i, c in enumerate(calls) if c.startswith("drive"))
    assert calls.index("gps Junction: one") < first_drive, "the GPS set before the AI drives"
    assert calls[-2:] == ["release", "gps clear"]


def test_the_junctions_file_reads_both_kinds_of_entry(lua):
    if not os.path.isdir(EARTH_EXT):
        pytest.skip("BeamNG.Earth not found (set BEAMNG_EARTH_DIR)")
    lua.execute("geodesy = (require('beamngEarth/geodesy'))")
    set_ = json.load(open(os.path.join(ROOT, "tools", "junctions.json"), encoding="utf-8"))
    lua.globals().setjson = json.dumps(set_)
    # (a minimal JSON reader is not in the runtime: the set handed over as Lua tables)
    lst = lua.table()
    for i, j in enumerate(set_["junctions"], 1):
        t = lua.table()
        t.name = j["name"]
        for k in ("at", "xy", "anchor"):
            if k in j:
                t[k] = lua.table(*j[k])
        lst[i] = t
    st = lua.table()
    st.junctions = lst
    out = lua.globals().tour.fromSet(st, lua.eval("geodesy"))
    got = {out[i].name: (out[i].lat, out[i].lon) for i in range(1, len(out) + 1)}
    assert len(got) == len(set_["junctions"]), "every junction, those given by xy too"
    for j in set_["junctions"]:
        lat, lon = got[j["name"]]
        if "at" in j:
            assert (lat, lon) == tuple(j["at"])
        else:
            assert abs(lat - j["anchor"][0]) < 0.02 and abs(lon - j["anchor"][1]) < 0.02
