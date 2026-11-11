# Jason's frame observations: handoff and verification

Run from the repository root, with the tactical XML/TXT in `assets/tactical/`:

```python
import pandas as pd
from starter.load_atd import load_frames

frames = load_frames(data_dir="assets")
row = frames.iloc[0]
team_is_unknown = pd.isna(row["nearest_team"])
home_distance_is_unknown = pd.isna(row["home_distance_m"])
```

Only the existing numpy/pandas dependencies are needed. `assets` is relative to
the working directory; an absolute data directory also works. The original
`load()` retains its lazy kloppy import and TrackingDataset behavior.

| Column (in return order) | dtype | Meaning |
| --- | --- | --- |
| `frame` | `int64` | Video frame, ATD ID minus one; no half offsets |
| `ball_seen` | `bool` | Both ball coordinates are finite |
| `nearest_team` | nullable `string` | `H` or `A` only when both teams have valid players and one is strictly closer; otherwise `pd.NA` |
| `nearest_distance_m` | `float64` | Minimum known team distance, or NaN if neither is known |
| `home_distance_m` | `float64` | Minimum distance to a valid home player, or NaN |
| `away_distance_m` | `float64` | Minimum distance to a valid away player, or NaN |

The result has a standard RangeIndex and one ascending row per expected video
frame. Distances use `hypot(dx * 105.3, dy * 68)` with dimensions read from
`GAMES`. XML Score references identify home/away; raw slot references resolve
through PlayerChannel and Player entries. Declaration order and slot counts
are not hardcoded.

Missing ball means all distances are NaN. With a ball but no valid players,
`ball_seen` stays True and all distances are NaN. With only one team observed,
its distance and the nearest distance remain available, but `nearest_team`
is missing. Exact ties preserve distances and leave `nearest_team` missing.
Use `pd.isna()` for both NaN and `pd.NA`; do not compare them with equality or
coerce `pd.NA` to bool.

Positions require finite x and y. Zero and finite out-of-pitch coordinates are
retained. NaN, infinity, or an empty coordinate field makes that position
unavailable. Missing speed (NaN or an empty field) does not invalidate x/y.
Every observation must still have three fields, and nonnumeric text is an
error, including in speed. Corrupt XML mappings, malformed records, slot-count
mismatches, invalid frame sequences, and truncated files raise descriptive
errors instead of producing partial results. The TXT is streamed once; only
current observations and the output arrays are retained.

Pitch-plane proximity does not prove a touch, especially for an airborne ball.
Tracks are not persistent player identities. Anthony owns possession, Bruno
owns dead-ball periods, and Chahid owns integration. This loader does not
interpolate, smooth, infer either status, or read ground truth. Official
submission evaluation does not apply to this observation-only output.

## Checks run on 2026-11-10

```powershell
python -m unittest discover -s tests -v
python tests/smoke_load_frames.py
```

The synthetic suite passed all 15 tests in 0.152 seconds, including metre
scaling, team minima, exact ties versus tiny unequal distances, zero/outside
coordinates, nonfinite/missing observations, missing speed, one/no observed
teams, reordered XML declarations/slots/coordinates, mapping and record
corruption, frame-sequence errors, and the optional-kloppy boundary. Expected
frame counts are mocked; the tests never access real assets. The original
kloppy loader's delegation is checked with a mock because kloppy is absent.

Initial runs encountered Python 3.13 temporary-directory ACL errors in the
Windows sandbox. Fixtures now use unique workspace directories with inherited
permissions and cleanup; the resulting suite passed.

The separate opt-in smoke check passed on the full local tactical data:

| Diagnostic | Observed |
| --- | ---: |
| Rows / columns | 199,319 / 6 |
| Frame range | 0–199,318, contiguous |
| Ball seen / missing | 113,488 / 85,831 |
| Ball seen, both teams observed | 112,419 |
| Ball seen, home only | 170 |
| Ball seen, away only | 899 |
| Ball seen, neither team | 0 |
| Nearest team H / A / missing | 63,794 / 48,625 / 86,900 |
| Exact ties with both teams | 0 |
| Loader wall time | 14.73 seconds |

All six dtypes and the RangeIndex matched the contract. Every known distance
was finite and nonnegative. Every row satisfied the missing-data, minimum,
and strict nearest-team rules. kloppy was not installed; the smoke check also
blocked its import before importing and calling `load_frames`. Environment:
Python 3.13.3, numpy 2.2.6, pandas 2.3.0. Runtime varies by machine.

No unresolved contract issues remain. This parser deliberately targets the
provided tactical EPTS layout (one format, x/y/speed fields); other layouts
raise validation errors. Assets were only read locally. No dependencies were
installed, no Git repository was initialized, and nothing was committed,
pushed, or uploaded.
