"""Synthetic Smart Tagging and ball trajectories for dead-ball inference."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths
from starter.is_dead import _apply_intervals, _classify, is_dead


FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]


def event(kind, start, end, code="SET PIECES"):
    tags = {} if kind is None else {"type": kind}
    return {"code": code, "start_frame": start, "end_frame": end, "tags": tags}


def touchline_exit():
    ball = np.full((100, 2), np.nan)
    ball[5:12, 0] = 50
    ball[5:12, 1] = [1.1, 0.8, 0.5, 0.2, -0.1, -0.4, -0.7]
    return ball


class DeadBallTests(unittest.TestCase):
    def classify(self, ball, events=(), halves=None):
        if halves is None:
            halves = [(1, 0, len(ball) - 1)]
        return _classify(ball, events, halves, FPS, PITCH)

    def assert_stoppage(self, state, start, restart):
        np.testing.assert_array_equal(
            np.flatnonzero(state.to_numpy(dtype=bool, na_value=False)),
            np.arange(start, restart),
        )
        self.assertFalse(state[restart])

    def test_untagged_exit_and_restart(self):
        ball = touchline_exit()
        ball[42:53, 0] = 50
        ball[42:53, 1] = [-0.9, -0.6, -0.3, 0.05, 0.25, 0.45,
                          0.65, 0.85, 1.05, 1.25, 1.45]
        self.assert_stoppage(self.classify(ball), 9, 45)

    def test_short_crossing_gap_uses_first_observation(self):
        ball = touchline_exit()
        ball[9] = np.nan
        ball[12] = [50, -1.0]
        ball[42:53, 0] = 50
        ball[42:53, 1] = [-0.9, -0.6, -0.3, 0.05, 0.25, 0.45,
                          0.65, 0.85, 1.05, 1.25, 1.45]
        state = self.classify(ball)
        self.assertTrue(pd.isna(state[9]))
        self.assert_stoppage(state, 10, 45)

    def test_tagged_throw_can_use_ball_reacquisition_and_missing_team(self):
        ball = touchline_exit()
        ball[45:55, 0] = 50
        ball[45:55, 1] = 0.1 + np.arange(10) * 0.2
        tagged = event(" THROW-IN ", 40, 50, code="set pieces")
        self.assert_stoppage(self.classify(ball, [tagged, tagged]), 9, 45)
        for events in ([], [event("Throw-in", 40, 45)], [event(None, 40, 50)]):
            state = self.classify(ball, events)
            self.assertTrue(pd.isna(state[9:45]).all())
            self.assertFalse(state.fillna(False).any())

    def test_unmatched_exit_and_missing_ball_remain_unknown(self):
        for ball in (touchline_exit(), np.full((100, 2), np.nan)):
            state = self.classify(ball, [event("Throw-in", 0, 100)])
            self.assertTrue(pd.isna(state).all())

    def test_tracking_jump_does_not_establish_an_exit(self):
        ball = touchline_exit()
        ball[5:9, 1] += 10
        ball[45:55, 0] = 50
        ball[45:55, 1] = 0.1 + np.arange(10) * 0.2
        state = self.classify(ball, [event("Throw-in", 40, 50)])
        self.assertFalse(state.fillna(False).any())

    def test_goal_line_exit_and_placed_restart(self):
        for y in (1.5, 30.0):
            with self.subTest(y=y):
                ball = touchline_exit()[:, ::-1].copy()
                ball[5:12, 1] = y
                ball[30:60] = [1.5, y]
                ball[60:70, 0] = 1.7 + np.arange(10) * 0.2
                ball[60:70, 1] = y
                self.assert_stoppage(self.classify(ball), 9, 60)

    def test_tagged_free_kick_uses_observed_pause(self):
        ball = np.full((120, 2), np.nan)
        ball[5:30, 0] = 45 + np.arange(25) * 0.2
        ball[5:30, 1] = 30
        ball[30:90] = [50, 30]
        ball[90:100, 0] = 50.2 + np.arange(10) * 0.2
        ball[90:100, 1] = 30
        self.assert_stoppage(self.classify(ball, [event("Free Kick", 75, 95)]), 28, 90)
        self.assertFalse(self.classify(ball).fillna(False).any())

    def test_after_goal_kickoff_requires_a_goal_and_restart(self):
        ball = touchline_exit()[:, ::-1].copy()
        ball[5:12, 1] = PITCH[1] / 2
        ball[30:60] = np.array(PITCH) / 2
        ball[60:70, 0] = PITCH[0] / 2 + 0.2 + np.arange(10) * 0.2
        ball[60:70, 1] = PITCH[1] / 2
        kickoff = event("Kick Off (After Goal)", 54, 66)
        goal = event(None, 9, 61, code="GOALS")
        self.assert_stoppage(self.classify(ball, [kickoff, goal]), 9, 60)
        self.assertFalse(self.classify(ball, [kickoff]).fillna(False).any())

    def test_start_kickoff_marks_only_the_current_half_before_restart(self):
        ball = np.full((100, 2), np.nan)
        ball[30:60] = np.array(PITCH) / 2
        ball[60:70, 0] = PITCH[0] / 2 + 0.2 + np.arange(10) * 0.2
        ball[60:70, 1] = PITCH[1] / 2
        kickoff = event("Kick Off (Start)", 54, 66)
        state = self.classify(ball, [kickoff], [(1, 5, 20), (2, 40, 90)])
        self.assertTrue(pd.isna(state[:40]).all())
        self.assert_stoppage(state, 40, 60)

    def test_does_not_join_exit_to_restart_in_next_half(self):
        ball = touchline_exit()
        ball[45:55, 0] = 50
        ball[45:55, 1] = 0.1 + np.arange(10) * 0.2
        state = self.classify(ball, [event("Throw-in", 40, 50)], [(1, 5, 20), (2, 40, 90)])
        self.assertFalse(state.fillna(False).any())
        self.assertTrue(pd.isna(state[21:40]).all())

    def test_overlapping_intervals_with_conflicting_boundaries_abstain(self):
        state = pd.array([False] * 20, dtype="boolean")
        _apply_intervals(state, [(1, 3), (1, 3), (5, 11), (8, 15)])
        self.assertTrue(state[1:3].all())
        self.assertFalse(state[3])
        self.assertTrue(pd.isna(state[5:16]).all())
        self.assertFalse(state[16])

    def test_public_schema_and_xml_video_time(self):
        ball = touchline_exit()
        ball[45:55, 0] = 50
        ball[45:55, 1] = 0.1 + np.arange(10) * 0.2
        with TemporaryDirectory() as data_dir:
            path = Path(asset_paths(data_dir=data_dir)["smart_tagging"])
            path.parent.mkdir()
            path.write_text(
                "<file><instance><start>1.797</start><end>1.827</end>"
                "<code>SET PIECES</code><label><group>Type</group>"
                "<text>Throw-in</text></label></instance></file>",
                encoding="utf-8",
            )
            with patch("starter.is_dead.load_ball_positions", return_value=ball / PITCH) as load:
                with patch("starter.is_dead.half_video_blocks", return_value=[(1, 0, 99)]):
                    result = is_dead(data_dir=data_dir)
            load.assert_called_once_with(data_dir=data_dir)
        self.assertEqual(list(result.columns), ["frame", "is_dead"])
        self.assertEqual([str(dtype) for dtype in result.dtypes], ["int64", "boolean"])
        np.testing.assert_array_equal(result.frame, np.arange(100))
        self.assert_stoppage(result.is_dead.array, 9, 45)
