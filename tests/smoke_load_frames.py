"""Opt-in local-assets check: python tests/smoke_load_frames.py [data_dir]."""

import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Report the actual environment, then block kloppy to test the dependency boundary
# even when this check is run on a machine that has it installed.
print("kloppy installed:", importlib.util.find_spec("kloppy") is not None)
sys.modules["kloppy"] = None
from starter.load_atd import load_frames

start = perf_counter()
frames = load_frames(sys.argv[1] if len(sys.argv) > 1 else "assets")
elapsed = perf_counter() - start
assert frames.shape == (199319, 6), frames.shape
assert list(frames.columns) == ["frame", "ball_seen", "nearest_team",
                                "nearest_distance_m", "home_distance_m", "away_distance_m"]
assert [str(dtype) for dtype in frames.dtypes] == [
    "int64", "bool", "string", "float64", "float64", "float64"]
assert isinstance(frames.index, pd.RangeIndex) and frames.index.equals(pd.RangeIndex(199319))
np.testing.assert_array_equal(frames.frame, np.arange(199319))
seen = frames.ball_seen.to_numpy()
home = frames.home_distance_m.to_numpy()
away = frames.away_distance_m.to_numpy()
nearest = frames.nearest_distance_m.to_numpy()
h, a = ~np.isnan(home), ~np.isnan(away)
counts = {
    "ball_seen_true": int(seen.sum()),
    "ball_seen_false": int((~seen).sum()),
    "both_teams": int((seen & h & a).sum()),
    "home_only": int((seen & h & ~a).sum()),
    "away_only": int((seen & ~h & a).sum()),
    "no_teams_with_ball": int((seen & ~h & ~a).sum()),
}
assert counts == dict(ball_seen_true=113488, ball_seen_false=85831,
                      both_teams=112419, home_only=170, away_only=899,
                      no_teams_with_ball=0), counts
for values in (home, away, nearest):
    known = values[~np.isnan(values)]
    assert np.isfinite(known).all() and (known >= 0).all()
    assert np.isnan(values[~seen]).all()
assert np.isnan(nearest[~h & ~a]).all()
np.testing.assert_array_equal(nearest[h & ~a], home[h & ~a])
np.testing.assert_array_equal(nearest[~h & a], away[~h & a])
np.testing.assert_array_equal(nearest[h & a], np.minimum(home[h & a], away[h & a]))
expected_team = pd.Series(pd.NA, index=frames.index, dtype="string", name="nearest_team")
expected_team[h & a & (home < away)] = "H"
expected_team[h & a & (away < home)] = "A"
pd.testing.assert_series_equal(frames.nearest_team, expected_team)
print("shape:", frames.shape, "frames:", (int(frames.frame.iloc[0]), int(frames.frame.iloc[-1])))
print("dtypes:", {name: str(dtype) for name, dtype in frames.dtypes.items()})
print("counts:", counts)
print("nearest_team:", frames.nearest_team.value_counts(dropna=False).to_dict())
print("exact ties with both teams:", int((h & a & (home == away)).sum()))
print("all schema, frame, distance, missing-value and nearest-team assertions passed")
print("load_frames runtime: {:.2f} seconds".format(elapsed))
