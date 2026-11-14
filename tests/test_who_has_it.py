"""Made-up frame observations only; no match assets are read by these tests."""

import unittest

import numpy as np
import pandas as pd

from starter.who_has_it import who_has_it

# One row of evidence per code, as (ball_seen, home_distance_m, away_distance_m).
EVIDENCE = {
    "H": (True, 0.5, 10.0),         # a home player has the ball, no opponent near
    "A": (True, 10.0, 0.5),         # an away player has the ball, no opponent near
    "gap": (False, np.nan, np.nan),  # the ball is not tracked
    "loose": (True, 8.0, 9.0),      # the ball is seen but nobody is near it
    "duel": (True, 0.5, 0.8),       # both teams are on the ball
    "far": (True, 2.5, 10.0),       # home is nearest but out of reach
    "alone": (True, 0.5, np.nan),   # no away player is tracked
}


def observations(*codes, first_frame=0):
    """A table shaped like load_frames() output, built from EVIDENCE codes."""
    seen, home, away = (np.array(column) for column in zip(*(EVIDENCE[code] for code in codes)))
    nearest = pd.array([pd.NA] * len(codes), dtype="string")
    nearest[home < away] = "H"
    nearest[away < home] = "A"
    index = pd.RangeIndex(first_frame, first_frame + len(codes))
    return pd.DataFrame({
        "frame": index.to_numpy(dtype=np.int64),
        "ball_seen": seen,
        "nearest_team": nearest,
        "nearest_distance_m": np.fmin(home, away),
        "home_distance_m": home,
        "away_distance_m": away,
    }, index=index)


def owners(*codes, **options):
    """who_has_it() answers as a plain list, with None for missing."""
    options.setdefault("frames_required", 3)
    result = who_has_it(observations(*codes), **options)
    return [None if pd.isna(value) else value for value in result]


class WhoHasItTests(unittest.TestCase):
    def test_initial_possession_needs_enough_frames_in_a_row(self):
        self.assertEqual(owners("H", "H"), [None, None])

    def test_possession_starts_on_the_first_frame_of_the_confirming_run(self):
        self.assertEqual(owners("loose", "loose", "H", "H", "H", "H"),
                         [None, None, "H", "H", "H", "H"])

    def test_brief_opposition_touch_does_not_change_the_owner(self):
        self.assertEqual(owners("H", "H", "H", "A", "A", "H", "A", "A"), ["H"] * 8)

    def test_owner_is_kept_through_a_loose_ball(self):
        self.assertEqual(owners("H", "H", "H", *["loose"] * 5), ["H"] * 8)

    def test_owner_is_kept_through_a_contest(self):
        self.assertEqual(owners("H", "H", "H", *["duel"] * 5), ["H"] * 8)

    def test_owner_is_kept_through_a_tracking_gap(self):
        self.assertEqual(owners("H", "H", "H", *["gap"] * 5), ["H"] * 8)

    def test_turnover_starts_on_the_first_frame_of_the_confirming_run(self):
        # Owner H, then evidence A, missing, A, A, A. The lone A before the gap is only
        # a touch, so it stays with H. The change goes where the unbroken run of A begins.
        self.assertEqual(owners("H", "H", "H", "A", "gap", "A", "A", "A"),
                         ["H", "H", "H", "H", "H", "A", "A", "A"])

    def test_loose_ball_and_contest_also_clear_the_count(self):
        for interruption in ("loose", "duel", "H"):
            with self.subTest(interruption=interruption):
                self.assertEqual(owners("H", "H", "H", "A", "A", interruption, "A", "A"),
                                 ["H"] * 8)

    def test_out_of_reach_or_unseen_opponents_give_no_evidence(self):
        self.assertEqual(owners(*["far"] * 4), [None] * 4)
        self.assertEqual(owners(*["alone"] * 4), [None] * 4)

    def test_each_call_starts_fresh_and_keeps_the_input_index(self):
        # After a stoppage the caller passes a new slice. Nothing carries over from before it.
        table = observations("H", "H", "H", "gap", "A", "A", "A", first_frame=500)
        before, after = table.iloc[:3], table.iloc[3:]
        self.assertEqual(list(who_has_it(before, frames_required=3))[-1], "H")
        result = who_has_it(after, frames_required=3)
        self.assertTrue(result.index.equals(after.index))
        self.assertEqual(str(result.dtype), "string")
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(list(result.iloc[1:]), ["A", "A", "A"])

    def test_distance_settings_decide_what_counts_as_control(self):
        self.assertEqual(owners(*["far"] * 3, control_distance_m=3.0), ["H", "H", "H"])
        self.assertEqual(owners(*["duel"] * 3, clear_margin_m=0.2), ["H", "H", "H"])

    def test_empty_slice_gives_an_empty_answer(self):
        result = who_has_it(observations("H").iloc[:0])
        self.assertEqual(len(result), 0)
        self.assertEqual(str(result.dtype), "string")


if __name__ == "__main__":
    unittest.main()
