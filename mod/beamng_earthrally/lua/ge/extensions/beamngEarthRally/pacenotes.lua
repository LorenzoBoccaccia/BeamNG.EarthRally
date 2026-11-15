-- Pace notes: the corners of a special stage, read from its geometry, written as the stock structured
-- notebook the rally loop reads (<mission>/rally/notebooks/structured.notebook.json).
--
-- Pure (no game calls). The points are world points {x, y, z} (legs.toWorld), x east and y north, so a turn to the left
-- is a positive change of heading, which is also how the game's own corner measurement names a direction.
--
-- What the game does with a notebook (BeamNG.drive lua/ge/extensions/gameplay/rally, read, not run): a pacenote is two
-- waypoints on the road, "corner start" and "corner end", and a structured item { type = 'corner' } in slot 4. The game
-- measures the road between the two waypoints itself (a circle fitted through the driveline there) and takes the
-- direction, the sharpness (the circle's diameter, 1 = hardest to 6 = flattest, the bounds of the stock english
-- numbering) and the length word ("long", "extra long") from that, and fills the distance calls between notes from the
-- gaps. So the job here is where the corners are: their start and end. Everything else written (the descriptor of a
-- hairpin or a square corner, "opens" and "tightens") is the same judgement a stock notebook author makes by hand, from
-- the same geometry.
local M = {}

M.STEP_M = 5               -- the route is looked at every 5 m (the source geometry is <= 20 m)
M.SMOOTH_STEPS = 3         -- curvature is the change of heading over this many steps either side (30 m or so)
M.ENTER_RADIUS_M = 170     -- a corner begins where the road turns tighter than this ...
M.EXIT_RADIUS_M = 260      -- ... and goes on until it relaxes past this (so one bend is one corner, not several)
M.MIN_ANGLE_DEG = 14       -- a bend that turns less than this is not a corner
M.MERGE_GAP_M = 20         -- two bends the same way this close are one corner
M.TRIM = 0.05              -- the corner runs from the first 5% of its turn to the last 5% (the smoothing spreads it)
M.MIN_LENGTH_M = 15        -- the game wants a few road points between the two waypoints to measure the corner
M.MIN_GAP_M = 3            -- between one note's end and the next one's start
M.START_CLEAR_M = 5        -- nothing before this (the start line) ...
M.END_CLEAR_M = 10         -- ... and nothing this close to the end: the route builder drops what lies past the stop
M.WAYPOINT_RADIUS = 8      -- the stock notebooks' waypoint radius

-- Hairpins and square corners are called by name, not by number, so the name has to be earned by the shape: what the stock
-- notebooks call a hairpin, a tight hairpin or an open hairpin (measured on all the stock structured notebooks: the circle
-- through the corner's two ends and its middle) is a U-turn on a circle of 7 to 17 m radius, 13 to 34 m between its ends.
-- A bend that turns as far on a circle of 60 m is a wide sweeping corner, called by its number and "long", not a hairpin.
M.HAIRPIN_MIN_ANGLE_DEG = 130
M.TIGHT_HAIRPIN_RADIUS_M = 9      -- below this radius: a tight hairpin
M.HAIRPIN_RADIUS_M = 18           -- below this: a hairpin ...
M.OPEN_HAIRPIN_RADIUS_M = 25      -- ... and below this an open hairpin; a wider bend has no name, only its number
M.SQUARE_MIN_ANGLE_DEG = 70
M.SQUARE_RADIUS_M = 16

-- Crests (curate.crests: the cleaned profile's convex bends, each with the speed that lifts a car off, liftMs). A crest is called by
-- how near the car comes to that at the speed it can carry there: its unloading u = (v / liftMs)^2, the share of its weight the crest
-- takes off it (1: the wheels leave the ground). v is the speed the corners allow (each corner's sqrt(MU G radius), reached from the
-- ones before and after by braking and accelerating, at most TOP_SPEED_MS; from a standstill at the start). Only a brow is a crest:
-- the road levels off or drops after it (a climb that only eases is not called).
M.G = 9.81
M.MU = 0.8                        -- grip (a corner's speed): a rally car on tarmac or good gravel
M.TOP_SPEED_MS = 44               -- 160 km/h
M.BRAKE_MS2 = 7                   -- m/s^2 the car slows down at into a corner ...
M.ACCEL_MS2 = 3.5                 -- ... and speeds up at out of one
M.CREST_BROW_GRADE = 0.04         -- the grade after a crest at most (rise over run): it levels off or drops
M.CREST_MIN_TURN = 0.05           -- and the grade it turns the road over by at least (curate.crests' turn, over CREST_SPAN_M either side):
                                  -- a kink of the elevation data (a 30 m grid read between its posts bends the road a percent or two at
                                  -- every post) is sharp enough to lift a fast car for an instant, and no crest a co-driver calls
M.CREST_CALLS = {                 -- the call by the unloading, the stock modifiers (compositors/styles/english/modifiers.lua)
  { 1.0, 'jump' }, { 0.75, 'bigCrest' }, { 0.5, 'crest' }, { 0.3, 'smallCrest' },
}
M.CREST_BEFORE_M = 40             -- a crest this little before a corner is called with it (before its number)
M.CREST_NOTE_M = 15               -- a crest on its own is a note this long

-- "Don't cut" after a corner (the stock notebooks' slot 5) where cutting it hits something: a bridge's parapet, a kerb and a
-- sidewalk. The spans are the caller's ({ from, to, why } in metres along the line the crests are measured on: the geometry's
-- bridges and sidewalks, tools/structures.py).

-- The stock english 1-6 numbering (compositors/styles/english/numbering/one_to_six.lua): the diameter of the circle the
-- corner is fitted with, in metres, up to which each number is called.
M.DIAMETER_BOUNDS = { 35, 65, 116, 190, 260 }

local sqrt, abs, floor, min, max = math.sqrt, math.abs, math.floor, math.min, math.max
local pi = math.pi
local atan2 = math.atan2 or function(y, x) return math.atan(y, x) end

-- 1 (a hairpin's circle) .. 6 (nearly straight) for a corner fitted with this diameter
function M.severity(diameterM)
  for i, bound in ipairs(M.DIAMETER_BOUNDS) do
    if diameterM < bound then return i end
  end
  return #M.DIAMETER_BOUNDS + 1
end

-- The route as points every `step` metres along the ground (the last one is the route's end), with the length along it
-- (s), x, y, z, and the heading of the road at each point (theta, radians, continuous: it does not wrap at +-pi).
local function resample(pts, step)
  local X, Y, Z, S = { pts[1][1] }, { pts[1][2] }, { pts[1][3] }, { 0 }
  local acc, nextS = 0, step
  for i = 2, #pts do
    local a, b = pts[i - 1], pts[i]
    local d = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    while d > 0 and nextS <= acc + d + 1e-9 do
      local f = (nextS - acc) / d
      X[#X + 1], Y[#Y + 1], Z[#Z + 1], S[#S + 1] = a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f, nextS
      nextS = nextS + step
    end
    acc = acc + d
  end
  if acc - S[#S] > step * 0.25 then
    local l = pts[#pts]
    X[#X + 1], Y[#Y + 1], Z[#Z + 1], S[#S + 1] = l[1], l[2], l[3], acc
  end
  local n = #X
  local h = {}                                   -- the heading of each step
  for j = 1, n - 1 do
    local v = atan2(Y[j + 1] - Y[j], X[j + 1] - X[j])
    if j > 1 then v = v - 2 * pi * floor((v - h[j - 1]) / (2 * pi) + 0.5) end
    h[j] = v
  end
  local theta = {}
  for i = 1, n do
    theta[i] = i == 1 and h[1] or (i == n and h[n - 1] or (h[i - 1] + h[i]) / 2)
  end
  return { n = n, x = X, y = Y, z = Z, s = S, h = h, theta = theta, length = S[n] }
end

-- the index below the length s (and the fraction of the way to the next point)
local function locate(r, s)
  s = max(0, min(r.length, s))
  local lo, hi = 1, r.n
  while hi - lo > 1 do
    local mid = floor((lo + hi) / 2)
    if r.s[mid] <= s then lo = mid else hi = mid end
  end
  local d = r.s[hi] - r.s[lo]
  return lo, hi, d > 1e-9 and (s - r.s[lo]) / d or 0
end

-- The route's point at length s along it {x, y, z}
local function pointAt(r, s)
  local lo, hi, f = locate(r, s)
  return { r.x[lo] + (r.x[hi] - r.x[lo]) * f, r.y[lo] + (r.y[hi] - r.y[lo]) * f, r.z[lo] + (r.z[hi] - r.z[lo]) * f }
end

-- ... and its direction of travel (a unit vector)
local function directionAt(r, s)
  local a, b = pointAt(r, s - M.STEP_M / 2), pointAt(r, s + M.STEP_M / 2)
  local dx, dy, dz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local l = sqrt(dx * dx + dy * dy + dz * dz)
  if l < 1e-9 then return { 0, 1, 0 } end
  return { dx / l, dy / l, dz / l }
end

local function thetaAt(r, s)
  local lo, hi, f = locate(r, s)
  return r.theta[lo] + (r.theta[hi] - r.theta[lo]) * f
end

-- Signed curvature (1/m, positive to the left) at each point: the change of heading over a few steps either side.
local function curvature(r)
  local k, K = {}, M.SMOOTH_STEPS
  for i = 1, r.n do
    local a, b = max(1, i - K), min(r.n - 1, i + K - 1)
    if b > a then
      local span = (r.s[b] + r.s[b + 1]) / 2 - (r.s[a] + r.s[a + 1]) / 2
      k[i] = span > 1e-9 and (r.h[b] - r.h[a]) / span or 0
    else
      k[i] = 0
    end
  end
  return k
end

-- The radius of the circle through three points (x, y), or nil when they are in line
local function circumradius(p, q, w)
  local a, b, c = sqrt((q[1] - w[1]) ^ 2 + (q[2] - w[2]) ^ 2), sqrt((p[1] - w[1]) ^ 2 + (p[2] - w[2]) ^ 2),
    sqrt((p[1] - q[1]) ^ 2 + (p[2] - q[2]) ^ 2)
  local twiceArea = abs((q[1] - p[1]) * (w[2] - p[2]) - (w[1] - p[1]) * (q[2] - p[2]))
  if twiceArea < 1e-9 then return nil end
  return a * b * c / (2 * twiceArea)
end

-- The runs of points on which the road turns one way tighter than ENTER_RADIUS_M (with the hysteresis), merged when
-- two the same way are close together. { sign, i0, i1 } each (sign 1 = left).
local function runs(r, kappa)
  local kEnter, kExit = 1 / M.ENTER_RADIUS_M, 1 / M.EXIT_RADIUS_M
  local out, cur = {}, nil
  for i = 1, r.n do
    local k = kappa[i]
    local sign = k > 0 and 1 or -1
    if cur then
      if sign == cur.sign and abs(k) >= kExit then
        cur.i1 = i
      else
        out[#out + 1], cur = cur, nil
      end
    end
    if not cur and abs(k) >= kEnter then cur = { sign = sign, i0 = i, i1 = i } end
  end
  if cur then out[#out + 1] = cur end
  local merged = {}
  for _, run in ipairs(out) do
    local last = merged[#merged]
    if last and last.sign == run.sign and r.s[run.i0] - r.s[last.i1] <= M.MERGE_GAP_M then
      last.i1 = run.i1
    else
      merged[#merged + 1] = run
    end
  end
  return merged
end

-- Where along the route (length) the heading has turned `frac` of the way from the point `from` to the point `to`
-- (indexes), searched from `from` towards `to`. Interpolated between points.
local function crossing(r, from, to, frac)
  local t0 = r.theta[from]
  local want = frac * abs(r.theta[to] - t0)
  local step = to > from and 1 or -1
  local prev = from
  for i = from + step, to, step do
    local turned = abs(r.theta[i] - t0)
    if turned >= want then
      local before = abs(r.theta[prev] - t0)
      local f = turned > before and (want - before) / (turned - before) or 1
      return r.s[prev] + (r.s[i] - r.s[prev]) * f
    end
    prev = i
  end
  return r.s[to]
end

-- Hairpins and square corners are called by name, not by number: the turn has to be a U-turn (or a right angle) and on a
-- small circle, as in a stock notebook. The angle alone is not enough: a long bend of 150 degrees on a 60 m circle is not one.
local function descriptor(angle, radius)
  if angle >= M.HAIRPIN_MIN_ANGLE_DEG and radius < M.OPEN_HAIRPIN_RADIUS_M then
    if radius < M.TIGHT_HAIRPIN_RADIUS_M then return 'tightHairpin' end
    if radius < M.HAIRPIN_RADIUS_M then return 'hairpin' end
    return 'openHairpin'
  end
  if angle >= M.SQUARE_MIN_ANGLE_DEG and angle < M.HAIRPIN_MIN_ANGLE_DEG and radius <= M.SQUARE_RADIUS_M then return 'square' end
  return nil
end

-- "Opens" or "tightens": the second half of a long corner much wider or tighter than the first.
local function shapeOf(r, s0, s1, sign)
  if s1 - s0 < 40 then return nil end
  local sm = (s0 + s1) / 2
  local ta, tb = sign * (thetaAt(r, sm) - thetaAt(r, s0)), sign * (thetaAt(r, s1) - thetaAt(r, sm))
  local minTurn = 6 * pi / 180
  if ta < minTurn or tb < minTurn then return nil end
  local ratio = ((s1 - sm) / tb) / ((sm - s0) / ta)      -- the second half's radius over the first's
  if ratio <= 0.6 then return 'tightens' end
  if ratio >= 1.7 then return 'opens' end
  return nil
end

-- The corners of a route, in order of travel: { { s0, s1 (length along the route, m), direction = 'left' | 'right',
-- angle (degrees turned), radius (m, over the whole corner), severity (1-6), descriptor, shape, length }, ... }, and the
-- resampled route they are measured on (for placing the waypoints).
function M.scan(pts, opts)
  if #pts < 3 then return {}, nil end
  local r = resample(pts, M.STEP_M)
  if r.n < 3 then return {}, r end
  local K = M.SMOOTH_STEPS
  local rs = runs(r, curvature(r))
  local corners = {}
  for k, run in ipairs(rs) do
    -- the corner starts and ends in the run's smoothing: look at the turn a little further out, but not into a neighbour
    local prevEnd, nextStart = rs[k - 1] and rs[k - 1].i1 or 0, rs[k + 1] and rs[k + 1].i0 or r.n + 1
    local lo = max(1, run.i0 - min(K, floor((run.i0 - prevEnd) / 2)))
    local hi = min(r.n, run.i1 + min(K, floor((nextStart - run.i1) / 2)))
    local turn = r.theta[hi] - r.theta[lo]
    local angle = abs(turn) * 180 / pi
    if hi > lo and angle >= M.MIN_ANGLE_DEG then
      local s0, s1 = crossing(r, lo, hi, M.TRIM), crossing(r, hi, lo, M.TRIM)
      if s1 <= s0 then s0, s1 = r.s[lo], r.s[hi] end
      local sweep = abs(turn) * (1 - 2 * M.TRIM)
      -- The radius is what the game will measure: the circle through the corner's two ends and its middle (the game fits
      -- one circle through the road points at its two waypoints and halfway between). Not the length over the sweep: the
      -- source points are 10-20 m apart and the heading changes at each, so the length between the first and last change
      -- overstates the arc and the radius with it. Where the three points are all but in line the circle is huge and
      -- the circle through the ends that turns the road by the sweep stands in.
      local a, b = pointAt(r, s0), pointAt(r, s1)
      local radius = circumradius(a, pointAt(r, (s0 + s1) / 2), b)
      if not radius or radius > 2000 then
        local chord = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
        radius = sweep < pi * 0.8 and chord / (2 * math.sin(sweep / 2)) or (s1 - s0) / sweep
      end
      local c = {
        s0 = s0, s1 = s1, sign = run.sign, direction = run.sign > 0 and 'left' or 'right',
        angle = angle, radius = radius, severity = M.severity(2 * radius),
      }
      c.descriptor = descriptor(c.angle, radius)
      if not c.descriptor then c.shape = shapeOf(r, s0, s1, run.sign) end
      corners[#corners + 1] = c
    end
  end
  -- room for the game to measure each corner (a few road points between its two waypoints) ...
  for _, c in ipairs(corners) do
    local short = M.MIN_LENGTH_M - (c.s1 - c.s0)
    if short > 0 then c.s0, c.s1 = c.s0 - short / 2, c.s1 + short / 2 end
  end
  -- ... and between one corner and the next, where they run into each other
  for i = 2, #corners do
    local a, b = corners[i - 1], corners[i]
    if b.s0 < a.s1 + M.MIN_GAP_M then
      local mid = (a.s1 + b.s0) / 2
      a.s1, b.s0 = mid - M.MIN_GAP_M / 2, mid + M.MIN_GAP_M / 2
    end
  end
  local kept = {}
  for _, c in ipairs(corners) do
    c.s0, c.s1 = max(M.START_CLEAR_M, c.s0), min(r.length - M.END_CLEAR_M, c.s1)
    if c.s1 - c.s0 >= 6 then
      c.length = c.s1 - c.s0
      kept[#kept + 1] = c
    end
  end
  M.markDontCut(kept, opts, r.length)
  kept.crests = M.placeCrests(kept, opts, r.length)
  return kept, r
end

-- The speed (m/s) a car can carry at s along the route between the corners (see the crests above)
function M.speedAt(corners, s)
  local v = min(M.TOP_SPEED_MS, sqrt(2 * M.ACCEL_MS2 * max(s, 0)))
  for _, c in ipairs(corners) do
    local vc = min(M.TOP_SPEED_MS, sqrt(M.MU * M.G * c.radius))
    local here
    if s < c.s0 then here = sqrt(vc * vc + 2 * M.BRAKE_MS2 * (c.s0 - s))
    elseif s > c.s1 then here = sqrt(vc * vc + 2 * M.ACCEL_MS2 * (s - c.s1))
    else here = vc end
    if here < v then v = here end
  end
  return v
end

-- the call for a crest of lift speed liftMs passed at v (nil: not worth one), and the unloading
function M.crestCall(liftMs, v)
  local u = (v / liftMs) ^ 2
  for _, t in ipairs(M.CREST_CALLS) do
    if u >= t[1] then return t[2], u end
  end
  return nil, u
end

-- the scale from the caller's metres (opts.lineLength long) to the route's
local function scaleOf(opts, length)
  local l = opts and opts.lineLength
  return (l and l > 0) and length / l or 1
end

-- corner.dontCut = why (the first span it overlaps)
function M.markDontCut(corners, opts, length)
  local spans = opts and opts.dontCut
  if not spans then return end
  local k = scaleOf(opts, length)
  for _, c in ipairs(corners) do
    for _, sp in ipairs(spans) do
      if sp.from * k < c.s1 and sp.to * k > c.s0 then
        c.dontCut = sp.why or 'bridge'
        break
      end
    end
  end
end

-- The crests called: each attached to the corner it is in (c.overCrest), or before (c.preCrest), else a note of its own.
-- Returns the ones on their own { { s0, s1, at, call, liftMs, speedMs, unload } } in order.
function M.placeCrests(corners, opts, length)
  local alone = {}
  local crests = opts and opts.crests
  if not crests then return alone end
  local k = scaleOf(opts, length)
  for _, cr in ipairs(crests) do
    local at = cr.s * k
    if (cr.gradeOut or 0) <= M.CREST_BROW_GRADE and (cr.turn or 1) >= M.CREST_MIN_TURN
        and at > M.START_CLEAR_M and at < length - M.END_CLEAR_M then
      local v = M.speedAt(corners, at)
      local call, u = M.crestCall(cr.liftMs, v)
      if call then
        local info = { at = at, call = call, liftMs = cr.liftMs, speedMs = v, unload = u }
        local host, before
        for _, c in ipairs(corners) do
          if at >= c.s0 and at <= c.s1 then host = c break end
          if at < c.s0 and c.s0 - at <= M.CREST_BEFORE_M then host, before = c, true break end
        end
        if host then
          local slot = before and 'preCrest' or 'overCrest'
          if not host[slot] or host[slot].unload < u then host[slot] = info end
        else
          info.s0, info.s1 = at - M.CREST_NOTE_M / 2, at + M.CREST_NOTE_M / 2
          alone[#alone + 1] = info
        end
      end
    end
  end
  -- a note of its own stays clear of the corners' (and the other crests') waypoints, or is not called
  local out = {}
  for _, a in ipairs(alone) do
    local clear = a.s0 >= M.START_CLEAR_M and a.s1 <= length - M.END_CLEAR_M
    for _, c in ipairs(corners) do
      if a.s0 < c.s1 + M.MIN_GAP_M and a.s1 > c.s0 - M.MIN_GAP_M then clear = false end
    end
    if clear and out[#out] and a.s0 < out[#out].s1 + M.MIN_GAP_M then clear = false end
    if clear then out[#out + 1] = a end
  end
  return out
end

local function waypoint(r, name, wpType, s, id)
  return { name = name, normal = directionAt(r, s), oldId = id, pos = pointAt(r, s), radius = M.WAYPOINT_RADIUS, waypointType = wpType }
end

-- A stock pacenote: its structured items, our metadata and its two waypoints (corner start, corner end)
local function noteShell(i, items, meta, wpStart, wpEnd)
  return {
    audioMode = 1, distanceBeforeModifier = false, ignoreDistanceCalls = false, includeLinkWord = true, isolate = false,
    -- (not `calibration`: that is where the rally editor keeps what its author hears, for calibrating the numbering)
    metadata = { earthrally = meta },
    name = 'Pacenote ' .. i, oldId = 3 * i - 1,
    pacenoteWaypoints = { wpStart, wpEnd },
    pk = string.format('%08x', (i * 2654435761) % 4294967296),
    slowCorner = false, slowCornerReleaseType = 30, todo = false, triggerType = 1,
    structured = { items = items, schemaVersion = 3 },
  }
end

local function notebookOf(notes)
  return {
    audioMode = 4, authors = 'BeamNG.EarthRally', created_at = 0, updated_at = 0, description = '', metadata = {},
    name = 'structured', pacenotes = notes, version = '4',
  }
end

-- a crest's call over a corner: "over crest" (or "over jump" when the car would leave the ground)
local function overCall(cr) return cr.call == 'jump' and 'overJump' or 'overCrest' end
-- ... and before one: "brake over crest" when the car comes near to leaving the ground, else the crest's own word
local function preCall(cr) return cr.unload >= 0.75 and 'brakeOverCrest' or cr.call end

local function crestMeta(cr)
  return { atM = floor(cr.at + 0.5), call = cr.call, liftKmh = floor(cr.liftMs * 3.6 + 0.5),
    speedKmh = floor(cr.speedMs * 3.6 + 0.5), unload = floor(cr.unload * 100 + 0.5) / 100 }
end

-- The items and metadata of a scanned note: a corner (with a crest over or before it, "don't cut"), or a crest on its own
local function contentOf(c)
  local items, meta = {}, nil
  if c.call then                            -- (a crest on its own: no corner, its call before the distance)
    items['2'] = { type = c.call }
    meta = { crest = crestMeta(c) }
  else
    local corner = { type = 'corner' }
    if c.descriptor then corner.descriptor = c.descriptor end
    if c.shape then corner.shape = c.shape end
    items['4'] = corner
    if c.preCrest then items['2'] = { type = preCall(c.preCrest) } end
    local post = {}
    if c.overCrest then post[#post + 1] = { type = overCall(c.overCrest) } end
    if c.dontCut then post[#post + 1] = { type = 'dontCut' } end
    for q, it in ipairs(post) do items[tostring(4 + q)] = it end
    meta = {
      direction = c.direction, severity = c.severity, radiusM = floor(c.radius * 10 + 0.5) / 10,
      angleDeg = floor(c.angle + 0.5), lengthM = floor(c.length + 0.5), atM = floor(c.s0 + 0.5),
    }
    if c.dontCut then meta.dontCut = c.dontCut end
    if c.overCrest then meta.crest = crestMeta(c.overCrest) end
    if c.preCrest then meta.crestBefore = crestMeta(c.preCrest) end
  end
  return items, meta
end

local function pacenote(r, c, i)
  local id = 3 * i - 1                      -- (the stock ids: the note, then its two waypoints)
  local items, meta = contentOf(c)
  return noteShell(i, items, meta, waypoint(r, 'corner start', 'cornerStart', c.s0, id + 1),
    waypoint(r, 'corner end', 'cornerEnd', c.s1, id + 2))
end

-- One line for the log: how many corners the notebook calls and where, against the route's length, and the names among them
function M.summary(corners, routeLength)
  if #corners == 0 then return 'no corners' end
  local names, bySeverity = 0, {}
  for _, c in ipairs(corners) do
    if c.descriptor then names = names + 1 end
    bySeverity[c.severity] = (bySeverity[c.severity] or 0) + 1
  end
  local counts = {}
  for sev = 1, 6 do counts[#counts + 1] = string.format('%d:%d', sev, bySeverity[sev] or 0) end
  local crests, dont = #(corners.crests or {}), 0
  for _, c in ipairs(corners) do
    if c.overCrest then crests = crests + 1 end
    if c.preCrest then crests = crests + 1 end
    if c.dontCut then dont = dont + 1 end
  end
  return string.format("%d corners called, the first at %.0f m and the last at %.0f m of %.0f m; %d by name (hairpin, square); by number %s; " ..
    "%d crest%s, %d don't cut",
    #corners, corners[1].s0, corners[#corners].s0, routeLength or 0, names, table.concat(counts, ' '),
    crests, crests == 1 and '' or 's', dont)
end

-- The notebook for a route (world points): named 'structured', the one every stock voicepack lists last, so any
-- voicepack picks it up. Returns the notebook table and the corners it was made from (their .crests: the crests called on
-- their own). opts: { crests = curate.crests of the line, dontCut = { { from, to, why } }, lineLength = the length both are
-- measured on } (see the crests and "don't cut" above); without them, corners only.
function M.notebook(pts, opts)
  local corners, r = M.scan(pts, opts)
  local seq = {}
  for _, c in ipairs(corners) do seq[#seq + 1] = c end
  for _, a in ipairs(corners.crests or {}) do seq[#seq + 1] = a end
  table.sort(seq, function(a, b) return a.s0 < b.s0 end)
  local notes = {}
  for i, c in ipairs(seq) do notes[i] = pacenote(r, c, i) end
  return notebookOf(notes), corners
end

-- The notes tools/make_pacenotes.py computed offline (a special's geometry "pacenotes": every corner, crest and "don't cut" of the
-- stage, measured on the ground Earth builds there), as the stock notebook at the current anchor. stored.notes: { { items, earthrally
-- (the metadata), start, startAhead, ['end'], endAhead } }, the points { lat, lon, h }, each waypoint's "ahead" a little further
-- along the road (its direction). place(points) -> the points in the world (legs.toWorld at the anchor).
function M.fromStored(stored, place)
  local notes = {}
  for i, sn in ipairs(stored and stored.notes or {}) do
    local w = place({ sn.start, sn.startAhead, sn['end'], sn.endAhead })
    local function wp(name, wpType, a, b, id)
      local dx, dy, dz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
      local l = sqrt(dx * dx + dy * dy + dz * dz)
      local normal = l > 1e-9 and { dx / l, dy / l, dz / l } or { 0, 1, 0 }
      return { name = name, normal = normal, oldId = id, pos = { a[1], a[2], a[3] }, radius = M.WAYPOINT_RADIUS, waypointType = wpType }
    end
    local items = {}
    for k, it in pairs(sn.items or {}) do items[tostring(k)] = it end
    local id = 3 * i - 1
    notes[i] = noteShell(i, items, sn.earthrally or {}, wp('corner start', 'cornerStart', w[1], w[2], id + 1),
      wp('corner end', 'cornerEnd', w[3], w[4], id + 2))
  end
  return notebookOf(notes)
end

-- One log line for stored notes, as summary() says it of scanned ones
function M.storedSummary(stored, routeLength)
  local notes = stored and stored.notes or {}
  local corners, names, bySeverity, first, last, crests, dont = 0, 0, {}, nil, nil, 0, 0
  for _, sn in ipairs(notes) do
    local d, it = sn.earthrally or {}, sn.items or {}
    local corner = it['4'] or it[4]
    if corner then
      corners = corners + 1
      if corner.descriptor then names = names + 1 end
      if d.severity then bySeverity[d.severity] = (bySeverity[d.severity] or 0) + 1 end
      first = first or d.atM
      last = d.atM or last
    end
    if d.crest then crests = crests + 1 end
    if d.crestBefore then crests = crests + 1 end
    if d.dontCut then dont = dont + 1 end
  end
  if corners == 0 and crests == 0 then return 'no corners' end
  local counts = {}
  for sev = 1, 6 do counts[#counts + 1] = string.format('%d:%d', sev, bySeverity[sev] or 0) end
  return string.format("%d corners called, the first at %.0f m and the last at %.0f m of %.0f m; %d by name (hairpin, square); by number %s; " ..
    "%d crest%s, %d don't cut (precomputed%s)", corners, first or 0, last or 0, routeLength or 0, names, table.concat(counts, ' '),
    crests, crests == 1 and '' or 's', dont, stored.ground and (', ground: ' .. tostring(stored.ground)) or '')
end

return M
