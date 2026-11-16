"""Integration tests use invented frame observations, never match assets."""

import unittest

import numpy as np
import pandas as pd

from starter.make_chahid_submission import make_labels


def tables(evidence, stoppages=()):
    distance = {"H": (0.5, 10.0), "A": (10.0, 0.5),
                "?": (np.nan, np.nan)}
    frame = np.arange(len(evidence), dtype=np.int64)
    pairs = np.array([distance[code] for code in evidence])
    frames = pd.DataFrame({"frame": frame, "ball_seen": np.isfinite(pairs[:, 0]),
                           "home_distance_m": pairs[:, 0], "away_distance_m": pairs[:, 1]})
    stopped = pd.array([pd.NA] * len(evidence), dtype="boolean")
    stopped[list(stoppages)] = True
    dead = pd.DataFrame({"frame": frame, "is_dead": stopped})
    return frames, dead


class MakeChahidSubmissionTests(unittest.TestCase):
    def test_stoppages_reset_owner_and_restart_prefix_gets_first_owner(self):
        # A short A run after H cannot take possession; after a stop, it can.
        evidence = "?" * 2 + "H" * 25 + "A" * 10 + "?" + "A" * 25
        frames, dead = tables(evidence, stoppages=(37,))
        labels = make_labels(frames, dead, [(1, 2, len(evidence) - 1)])
        self.assertEqual(labels["label"].tolist(),
                         ["X"] * 2 + ["H"] * 35 + ["D"] + ["A"] * 25)
        self.assertEqual(labels["frame"].tolist(), list(range(len(evidence))))

    def test_unconfirmed_stretch_remains_x_and_outside_halves_is_x(self):
        evidence = "?" * 2 + "H" * 25 + "?" * 5 + "A" * 5 + "?" + "H" * 25 + "?" * 2
        frames, dead = tables(evidence, stoppages=(27,))
        labels = make_labels(frames, dead, [(1, 2, 36), (2, 38, 62)])
        expected = (["X"] * 2 + ["H"] * 25 + ["D"] + ["X"] * 10
                    + ["H"] * 25 + ["X"] * 2)
        self.assertEqual(labels["label"].tolist(), expected)

    def test_rejects_misaligned_tables(self):
        frames, dead = tables("H" * 4)
        dead.loc[2, "frame"] = 3
        with self.assertRaisesRegex(ValueError, "ordered video frames"):
            make_labels(frames, dead, [(1, 0, 3)])


if __name__ == "__main__":
    unittest.main()
