# Credits and licences

## Our code

BeamNG.EarthRally's own code (the Lua extension, the tools, the tests) is under the MIT licence: [LICENSE](LICENSE).

Two test helpers are ports of BeamNG.drive's Lua and stay under BeamNG's bCDDL 1.1 ([licenses/bCDDL-1.1.txt](licenses/bCDDL-1.1.txt)):
`tests/stock_rally.py` (from geom.lua, mathlib.lua, drivelineV3.lua and drivelineRoute.lua) and `tests/beamng_quat.py`
(from mathlib.lua). They are not shipped in the mod.

The mod runs inside BeamNG.drive and needs [BeamNG.Earth](https://github.com/TecnaGamer/BeamNG.Earth); neither is part
of this repository. BeamNG.drive's own files are BeamNG GmbH's, under its EULA and modding guidelines.

## Route data

The editions' route lines, closures and pace notes (`editions/*/geometry`, `editions/*/closures`, the pace notes in the
geometry files) are made from:

- **OpenStreetMap**: roads through [OSRM](https://project-osrm.org/) and the Overpass API.
  © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0: https://www.openstreetmap.org/copyright
- **Overture Maps**: the transportation, base and buildings themes BeamNG.Earth reads. © Overture Maps Foundation,
  © OpenStreetMap contributors, ODbL 1.0 (and the other licences Overture lists for each theme): https://docs.overturemaps.org/attribution/
- **Elevation**: AWS Terrain Tiles (Mapzen), made from SRTM, GMTED, NED/3DEP and ETOPO1, among others:
  https://github.com/tilezen/joerd/blob/master/docs/attribution.md

The route data in `editions/` is a derivative database of OpenStreetMap and is under the ODbL 1.0 too:
https://opendatacommons.org/licenses/odbl/1-0/

The stage tracks are our own reconstruction from public route information; no organiser publishes them, and nothing
here is endorsed by any rally organiser.
