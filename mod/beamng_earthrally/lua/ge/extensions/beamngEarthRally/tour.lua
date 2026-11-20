-- The junction tour (beamngEarthRally_core.tour()): every junction of the eval set (tools/junctions.json, copied to
-- earthrally/junctions.json by tools/install.py) driven by the AI, one after the other, for one recording.
--
-- For each junction: Earth's teleport to it; the roads round it built (roads.readyAround) and the navgraph in; a
-- drive through it picked on the navgraph (pick: from a node DRIVE_M before the junction's nearest node to one DRIVE_M
-- past it, the straightest way through, one-way roads only their way); the car put on the start facing the way, on the road's surface as
-- the game has it (waited for: a viaduct's deck comes in later), Earth's
-- GPS set to the end and its autodrive on (the route and arrows on screen, the car's AI in its lane at the legal
-- speed); when the GPS says it arrived (or after DRIVE_MAX_S) the next. A junction with a road over another (an overpass, an underpass) is driven on each level: a
-- second pass from the nearest node standing LEVEL_M above or below the first's.
--
-- Pure but for what ctx gives it (core.lua's), so the picking runs in the tests (tests/test_tour.py):
--   ctx.anchor()                  the current geodesy anchor (geodeticToWorld / worldToGeodetic) or nil
--   ctx.ready()                   Earth can take a teleport now
--   ctx.teleport(lat, lon, name)  Earth's teleport; true when taken
--   ctx.roadsReady(x, y, r)       the roads within r of x, y are in (true when there is nothing to wait for)
--   ctx.nodes()                   the navgraph's nodes (map.getMap().nodes) or nil
--   ctx.car()                     { pos = {x, y, z}, speed = m/s } of the player's car or nil
--   ctx.surface(x, y, top, reach) the height of the first surface the game has (the graded road, a deck, the ground) a
--                                 ray down from top finds within reach, or nil (nothing there yet)
--   ctx.place(pos, dir)           put the car at pos ({x, y, z}) facing dir ({x, y})
--   ctx.drive(names, dest, label) Earth's GPS to dest ({lat, lon}) and Earth's autodrive on (no driving of ours);
--                                 ctx.arrived(): the GPS has let go (it clears itself on arrival)
--   ctx.stopAi()                  Earth's autodrive off
--   ctx.gpsClear()
--   ctx.say(msg, seconds), ctx.log(level, msg)
local M = {}

M.DRIVE_M = 160            -- m before and past the junction's node the drive starts and ends (a path this long either way)
M.DRIVE_MIN_M = 90         -- ... at least this far (a shorter road network: the farthest node it reaches)
M.LEVEL_M = 4.0            -- m above or below the first drive's node another road counts as another level
M.NEAR_M = 40              -- m round the junction the nodes of a drive may start from (its radius when it has one)
M.CUTOFF = 0.5             -- a link less drivable than this costs PENALTY times its length (guide.PLANNER's)
M.PENALTY = 10
M.ROADS_MAX_S = 300        -- s waited at most for the roads round it while Earth is still loading them (a cold cache:
                           -- a first visit's road plans, a busy server)
M.LOAD_MAX_S = 60          -- s waited at most, once they are in, for the navgraph and a drive through it (and for the
                           -- road's surface at a drive's start)
M.SURFACE_UP_M = 3.0       -- m over the drive's first node the ray for the road's surface starts (a road over it is
                           -- higher: a deck's clearance), and
M.SURFACE_DOWN_M = 8.0     -- ... m under the node it reaches (the road as graded is not at the navgraph's height)
M.PLACE_UP_M = 0.5         -- m over that surface the car is put
M.SETTLE_S = 3             -- s after the roads are in before the drive (the car settles, the GPS route is made)
M.DRIVE_MAX_S = 90         -- s a drive may take
M.ARRIVE_M = 30            -- m from the end: arrived (Earth's GPS's ROUTE_ARRIVE_M)
M.STUCK_S = 12             -- s under 1 m/s (after the first 10 s) ends a drive
M.PAUSE_S = 2              -- s at the end of a drive before the next

local function d2(a, b) return math.sqrt((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2) end

-- Can the link from node a to node b be driven that way? (a one-way link only from its inNode)
local function canGo(nodes, a, b)
  local l = nodes[a] and nodes[a].links and nodes[a].links[b]
  if not l then
    l = nodes[b] and nodes[b].links and nodes[b].links[a]
    if not l then return false end
  end
  if l.oneWay and l.inNode and l.inNode ~= a then return false end
  return true, l
end

local function neighbours(nodes, a)
  local out, seen = {}, {}
  for b in pairs(nodes[a].links or {}) do
    if nodes[b] then out[#out + 1] = b; seen[b] = true end
  end
  return out, seen
end

-- Links are kept on one of their two nodes (BeamNG's map keeps them on both); every node's other side, once.
local function adjacency(nodes)
  local adj = {}
  for a, n in pairs(nodes) do
    adj[a] = adj[a] or {}
    for b in pairs(n.links or {}) do
      if nodes[b] then
        adj[a][b] = true
        adj[b] = adj[b] or {}
        adj[b][a] = true
      end
    end
  end
  return adj
end

-- Shortest ways from root out to reach (forward: driving away from it; backward: driving toward it), costs as the
-- planner's: {node = {dist (m driven), cost, prev}}.
local function spread(nodes, adj, root, reach, forward)
  local best = { [root] = { dist = 0, cost = 0 } }
  local open = { root }
  while #open > 0 do
    local bi, bc = 1, best[open[1]].cost
    for i = 2, #open do if best[open[i]].cost < bc then bi, bc = i, best[open[i]].cost end end
    local a = table.remove(open, bi)
    local ba = best[a]
    if not ba.done then
      ba.done = true
      for b in pairs(adj[a] or {}) do
        local ok, l
        if forward then ok, l = canGo(nodes, a, b) else ok, l = canGo(nodes, b, a) end
        if ok then
          local len = d2(nodes[a].pos, nodes[b].pos)
          local dist = ba.dist + len
          local cost = ba.cost + len * (((l.drivability or 1) < M.CUTOFF) and M.PENALTY or 1)
          if dist <= reach and (not best[b] or cost < best[b].cost) then
            best[b] = { dist = dist, cost = cost, prev = a }
            open[#open + 1] = b
          end
        end
      end
    end
  end
  return best
end

local function chain(best, n)
  local out = {}
  while n do out[#out + 1] = n; n = best[n].prev end
  return out            -- (from n back to the root)
end

-- The drive through node root: { names (start .. root .. end), from, to } or nil. Start a node about DRIVE_M before it
-- (driving toward it), end one about DRIVE_M past it, leaving it another way than it came in (no U turn), the pair
-- farthest apart (the straightest way through).
function M.pick(nodes, root, adj)
  adj = adj or adjacency(nodes)
  if not nodes[root] then return nil end
  local reach = M.DRIVE_M * 1.5
  local back = spread(nodes, adj, root, reach, false)
  local fwd = spread(nodes, adj, root, reach, true)
  local function ends(best)
    local far, list = 0, {}
    for n, b in pairs(best) do if b.dist > far then far = b.dist end end
    local want = math.max(math.min(M.DRIVE_M, far), math.min(M.DRIVE_MIN_M, far))
    for n, b in pairs(best) do
      if n ~= root and b.dist >= want * 0.75 and b.dist <= want * 1.25 + 1e-6 then list[#list + 1] = n end
    end
    if #list == 0 then
      for n, b in pairs(best) do if n ~= root and b.dist >= far - 1e-6 then list[#list + 1] = n end end
    end
    return list
  end
  local starts, stops = ends(back), ends(fwd)
  local bestPair, bestD
  for _, s in ipairs(starts) do
    local inBy = chain(back, s)            -- s .. root
    local lastIn = inBy[#inBy - 1]         -- (the node before root)
    for _, e in ipairs(stops) do
      local out = chain(fwd, e)            -- e .. root
      local firstOut = out[#out - 1]
      if firstOut ~= lastIn then
        local d = d2(nodes[s].pos, nodes[e].pos)
        local key = string.format('%s>%s', s, e)
        if not bestD or d > bestD + 1e-9 or (math.abs(d - bestD) <= 1e-9 and key < bestPair.key) then
          bestD, bestPair = d, { s = s, e = e, inBy = inBy, out = out, key = key }
        end
      end
    end
  end
  if not bestPair then return nil end
  local names = {}
  for _, n in ipairs(bestPair.inBy) do names[#names + 1] = n end          -- s .. root
  for i = #bestPair.out - 1, 1, -1 do names[#names + 1] = bestPair.out[i] end  -- past root .. e
  -- (the points kept with the drive: the navgraph is made again as roads load, and its nodes may be gone by its end)
  local at = {}
  for i, n in ipairs(names) do local p = nodes[n].pos; at[i] = { x = p.x, y = p.y, z = p.z } end
  return { names = names, at = at, from = bestPair.s, to = bestPair.e }
end

-- The junction's drives: through its nearest node, and through the nearest one on another level (LEVEL_M) if any.
-- p: the junction's world point {x, y, z}; near: m round it.
function M.drives(nodes, p, near)
  local adj = adjacency(nodes)
  local cand = {}
  for n, nd in pairs(nodes) do
    local d = d2(nd.pos, p)
    if d <= near and adj[n] and next(adj[n]) then cand[#cand + 1] = { n = n, d = d } end
  end
  table.sort(cand, function(a, b) if a.d ~= b.d then return a.d < b.d end return a.n < b.n end)
  local out = {}
  for _, c in ipairs(cand) do
    local first = out[1]
    if not first or math.abs(nodes[c.n].pos.z - nodes[first.root].pos.z) >= M.LEVEL_M then
      local dv = M.pick(nodes, c.n, adj)
      if dv then
        dv.root = c.n
        out[#out + 1] = dv
        if #out == 2 then break end
      end
    end
  end
  return out
end

-- The tour --------------------------------------------------------------------------------------------------------
local T = nil       -- { list, i, phase, t, drives, k, ... } while one runs

local function pos3(v) return { x = v.x or v[1], y = v.y or v[2], z = v.z or v[3] or 0 } end

function M.running() return T ~= nil end
function M.status()
  if not T then return nil end
  local j = T.list[T.i]
  return string.format('%d of %d: %s (%s)', T.i, #T.list, j and j.name or '-', T.phase)
end

-- list: [{ name, lat, lon, radius, what }]
function M.start(ctx, list)
  T = { list = list, i = 1, phase = 'go', t = 0, ctx = ctx }
  ctx.log('I', string.format('junction tour: %d junctions', #list))
end

function M.stop(why)
  if not T then return end
  local ctx = T.ctx
  pcall(ctx.stopAi)
  pcall(ctx.gpsClear)
  ctx.log('I', 'junction tour stopped: ' .. tostring(why))
  T = nil
end

-- Earth's anchor commit (a teleport landed)
function M.onAnchor()
  if T and T.phase == 'teleporting' then T.phase, T.t = 'loading', 0 end
end

local function nextJunction(why)
  T.ctx.log('I', string.format('junction tour: %s done (%s)', T.list[T.i].name, why))
  T.i = T.i + 1
  T.phase, T.t = 'go', 0
  if T.i > #T.list then
    T.ctx.say('Junction tour: done', 8)
    M.stop('all junctions driven')
  end
end

-- The car on the road's surface at the drive's start once the game has it there (the road graded, a deck built): not at
-- the navgraph's height, which the car fell from (52 m over the ground where a viaduct was not in yet) or was put into a
-- road under.
local function beginDrive()
  T.phase, T.t = 'placing', 0
end

local function placeCar()
  local ctx, dv = T.ctx, T.drives[T.k]
  local a, b = dv.at[1], dv.at[2]
  local top = a.z + M.SURFACE_UP_M
  local z = ctx.surface(a.x, a.y, top, M.SURFACE_UP_M + M.SURFACE_DOWN_M)
  if not z then return false end
  ctx.place({ x = a.x, y = a.y, z = z + M.PLACE_UP_M }, { x = b.x - a.x, y = b.y - a.y })
  local e = dv.at[#dv.at]
  local anchor = ctx.anchor()
  local j = T.list[T.i]
  local label = string.format('%s%s', j.name, #T.drives > 1 and string.format(' (level %d of %d)', T.k, #T.drives) or '')
  T.dest, T.label = nil, label
  if anchor then
    local lat, lon = anchor:worldToGeodetic(e.x, e.y, e.z)
    T.dest = { lat = lat, lon = lon }
  end
  ctx.say(string.format('Junction %d of %d: %s', T.i, #T.list, label), 6)
  T.phase, T.t, T.slow = 'settle', 0, 0
  return true
end

-- the next drive of this junction, or the next junction
local function nextDrive(why)
  if T.k < #T.drives then
    T.k = T.k + 1
    beginDrive()
  else
    nextJunction(why)
  end
end

-- every frame (dt: s)
function M.update(dt)
  if not T then return end
  local ctx = T.ctx
  T.t = T.t + dt
  local j = T.list[T.i]
  if T.phase == 'go' then
    if ctx.ready() then
      local took, why = ctx.teleport(j.lat, j.lon, 'Junction: ' .. j.name)
      if took then
        T.phase, T.t, T.roadsIn = 'teleporting', 0, false
      elseif T.t > 30 then
        nextJunction('teleport refused: ' .. tostring(why))
      end
    end
  elseif T.phase == 'teleporting' then
    if T.t > 30 then T.phase, T.t = 'loading', 0 end        -- (no anchor commit seen: the place may be the same anchor's)
  elseif T.phase == 'loading' then
    local anchor = ctx.anchor()
    if not anchor then return end
    local x, y, z = anchor:geodeticToWorld(j.lat, j.lon, 0)
    T.p = { x = x, y = y, z = z }
    if not T.roadsIn then
      if not ctx.roadsReady(x, y, M.DRIVE_M + 40) then
        if T.t < M.ROADS_MAX_S then return end
        return nextJunction(string.format('the roads round it not in after %d s', M.ROADS_MAX_S))
      end
      T.roadsIn, T.t = true, 0          -- (the navgraph's wait from here)
    end
    local nodes = ctx.nodes()
    if not nodes or not next(nodes) then
      if T.t < M.LOAD_MAX_S then return end
      return nextJunction('no navgraph')
    end
    T.drives = M.drives(nodes, T.p, math.max(M.NEAR_M, j.radius or 0))
    if #T.drives == 0 then
      if T.t < M.LOAD_MAX_S then return end             -- (the navgraph may still be filling in)
      return nextJunction('no drive through it on the navgraph')
    end
    T.k = 1
    beginDrive()
  elseif T.phase == 'placing' then
    if not placeCar() and T.t > M.LOAD_MAX_S then
      T.ctx.log('W', string.format('junction tour: %s, drive %d: no road surface at its start', j.name, T.k))
      nextDrive('no road surface at its start')
    end
  elseif T.phase == 'settle' then
    if T.t >= M.SETTLE_S then
      ctx.drive(T.drives[T.k].names, T.dest, 'Junction: ' .. T.label)
      T.phase, T.t = 'driving', 0
    end
  elseif T.phase == 'driving' then
    local car = ctx.car()
    local at = T.drives[T.k].at
    local e = at[#at]
    local done
    if car and d2(pos3(car.pos), e) <= M.ARRIVE_M then done = 'arrived' end
    if T.t > 5 and ctx.arrived and ctx.arrived() then done = 'arrived (the GPS)' end
    if car and T.t > 10 then
      T.slow = (car.speed or 0) < 1 and (T.slow + dt) or 0
      if T.slow > M.STUCK_S then done = 'stopped short' end
    end
    if T.t > M.DRIVE_MAX_S then done = 'took too long' end
    if done then
      ctx.stopAi()
      T.why, T.phase, T.t = done, 'pause', 0
    end
  elseif T.phase == 'pause' then
    if T.t >= M.PAUSE_S then nextDrive(T.why) end
  end
end

-- The junctions file (earthrally/junctions.json: tools/junctions.json) as the tour's list: each one's lat, lon (an
-- entry given by xy in its anchor's frame: through that anchor). geodesy: Earth's (newAnchor).
function M.fromSet(set, geodesy)
  local out = {}
  for _, j in ipairs(set and set.junctions or {}) do
    local lat, lon
    if j.at then
      lat, lon = j.at[1], j.at[2]
    elseif j.xy and j.anchor and geodesy then
      lat, lon = geodesy.newAnchor(j.anchor[1], j.anchor[2]):worldToGeodetic(j.xy[1], j.xy[2], 0)
    end
    if lat and lon then out[#out + 1] = { name = j.name, lat = lat, lon = lon, radius = j.radius, what = j.what } end
  end
  return out
end

return M
