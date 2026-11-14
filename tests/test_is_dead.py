"""Synthetic Smart Tagging and ball trajectories for dead-ball inference."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths
from starter.is_dead import _apply_intervals, _classify, _observed_stoppages, is_dead


FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
RUNNING, WALKING = 3.0, 1.0  # average player speed in m/s


def frames(seconds):
    return round(seconds * FPS)


def moving(start, velocity, seconds):
    """Ball positions in metres, one row per frame, at a steady velocity in m/s."""
    steps = np.arange(frames(seconds))[:, None]
    return np.asarray(start, dtype=float) + steps * np.asarray(velocity, dtype=float) / FPS


def resting(at, seconds):
    return np.tile(np.asarray(at, dtype=float), (frames(seconds), 1))


def missing(seconds):
    return np.full((frames(seconds), 2), np.nan)


def players(*parts):
    """Average player speed per frame from (speed, seconds) pairs."""
    return np.concatenate([np.full(frames(seconds), speed) for speed, seconds in parts])


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
            running = np.full(len(ball), RUNNING)
            with patch("starter.is_dead.load_ball_positions", return_value=ball / PITCH) as load, \
                    patch("starter.is_dead.load_player_speeds", return_value=running) as speeds, \
                    patch("starter.is_dead.half_video_blocks", return_value=[(1, 0, 99)]):
                result = is_dead(data_dir=data_dir)
            load.assert_called_once_with(data_dir=data_dir)
            speeds.assert_called_once_with(data_dir=data_dir)
        self.assertEqual(list(result.columns), ["frame", "is_dead"])
        self.assertEqual([str(dtype) for dtype in result.dtypes], ["int64", "boolean"])
        np.testing.assert_array_equal(result.frame, np.arange(100))
        self.assert_stoppage(result.is_dead.array, 9, 45)

    def test_walking_players_add_stoppages_inside_the_halves_only(self):
        ball = missing(20)
        speed = players((RUNNING, 4), (WALKING, 12), (RUNNING, 4))
        half = [(1, frames(2), frames(18))]
        with TemporaryDirectory() as data_dir:
            path = Path(asset_paths(data_dir=data_dir)["smart_tagging"])
            path.parent.mkdir()
            path.write_text("<file></file>", encoding="utf-8")
            with patch("starter.is_dead.load_ball_positions", return_value=ball / PITCH), \
                    patch("starter.is_dead.load_player_speeds", return_value=speed), \
                    patch("starter.is_dead.half_video_blocks", return_value=half):
                state = is_dead(data_dir=data_dir).is_dead
        self.assertTrue(state[frames(6):frames(15)].all())
        self.assertTrue(state[:frames(2)].isna().all())
        self.assertTrue(state[frames(18) + 1:].isna().all())


class ObservedStoppageTests(unittest.TestCase):
    """Stoppages seen from walking players and from a ball outside the lines."""

    def find(self, ball, speed, halves=None):
        self.assertEqual(len(ball), len(speed))
        if halves is None:
            halves = [(1, 0, len(ball) - 1)]
        return _observed_stoppages(ball, speed, halves, FPS, PITCH)

    def test_ball_missing_while_players_walk_is_a_stoppage(self):
        ball = np.vstack([moving((20, 30), (10, 0), 5), missing(10), moving((70, 30), (-10, 0), 5)])
        speed = players((RUNNING, 6), (WALKING, 9), (RUNNING, 5))
        dead = self.find(ball, speed)
        self.assertTrue(dead[frames(8):frames(14)].all())
        self.assertFalse(dead[:frames(5)].any())
        self.assertFalse(dead[frames(15.5):].any())

    def test_ball_missing_while_players_run_is_live(self):
        ball = np.vstack([moving((20, 30), (10, 0), 5), missing(10), moving((70, 30), (-10, 0), 5)])
        self.assertFalse(self.find(ball, players((RUNNING, 20))).any())

    def test_ball_outside_the_lines_is_dead_from_the_first_outside_frame(self):
        ball = np.vstack([
            moving((20, 10), (10, 0), 5),      # in play
            moving((70, 10), (0, -10), 1.3),   # rolls over the touchline to y = -3
            resting((70, -3), 4),              # held outside the pitch
            moving((70, -3), (0, 10), 3),      # thrown back in
        ])
        outside = ball[:, 1] < -0.5
        np.testing.assert_array_equal(self.find(ball, players((RUNNING, 13.3))), outside)

    def test_slow_passing_between_walking_players_is_live(self):
        ball = np.vstack([moving((20, 30), (10, 0), 1.6), resting((36, 30), 1.6),
                          moving((36, 30), (-10, 0), 1.6), resting((20, 30), 1.6)] * 3)
        self.assertFalse(self.find(ball, players((WALKING, 19.2))).any())

    def test_placed_ball_is_dead_until_it_is_kicked(self):
        ball = np.vstack([moving((20, 30), (10, 0), 2), resting((40, 30), 10),
                          moving((40, 30), (12, 0), 3)])
        kick = frames(12)
        dead = self.find(ball, players((WALKING, 15)))
        self.assertTrue(dead[frames(5):kick - 2].all())
        self.assertLessEqual(abs(np.flatnonzero(dead)[-1] - kick), 3)

    def test_ball_thrown_back_during_a_stoppage_does_not_split_it(self):
        ball = np.vstack([moving((20, 30), (10, 0), 2), resting((40, 30), 6),
                          moving((40, 30), (10, 0), 0.48), resting((44.8, 30), 6),
                          moving((44.8, 30), (12, 0), 3)])
        dead = self.find(ball, players((WALKING, 17.48)))
        self.assertTrue(dead[frames(5):frames(14.3)].all())

    def test_stoppage_continues_while_the_ball_is_brought_back_after_going_out(self):
        ball = np.vstack([
            moving((25, 30), (-10, 0), 2.8),   # crosses the goal line to x = -3
            resting((-3, 30), 1),              # fetched from behind the goal
            moving((-3, 30), (3, 0), 2.32),    # carried back onto the pitch
            resting((3.96, 30), 9.88),         # placed while players jog into position
            moving((3.96, 30), (12, 0), 3),    # goal kick
        ])
        speed = players((RUNNING, 10), (WALKING, 6), (RUNNING, 3))
        first_outside = int(np.flatnonzero(ball[:, 0] < -0.5)[0])
        dead = self.find(ball, speed)
        self.assertFalse(dead[:first_outside].any())
        # Players start running at the kick (16 s). The smoothed speed sees that slightly early.
        self.assertTrue(dead[first_outside:frames(15.5)].all())
        self.assertFalse(dead[frames(16.2):].any())

    def test_long_break_is_kept_when_the_ball_never_left_the_pitch(self):
        ball = np.vstack([moving((20, 30), (10, 0), 1), resting((30, 30), 17)])
        speed = players((WALKING, 7), (RUNNING, 6), (WALKING, 5))
        dead = self.find(ball, speed)
        self.assertTrue(dead[frames(4):frames(6)].all())
        self.assertFalse(dead[frames(8):frames(11)].any())
        self.assertTrue(dead[frames(14):frames(17)].all())

    def test_break_with_no_ball_seen_stays_part_of_the_stoppage(self):
        # A goal celebration: players jog for a while, but the ball never shows up.
        speed = players((WALKING, 6), (RUNNING, 8), (WALKING, 6))
        self.assertTrue(self.find(missing(20), speed)[frames(1):frames(19)].all())

    def test_brief_quiet_moment_is_not_a_stoppage(self):
        speed = players((RUNNING, 5), (WALKING, 2), (RUNNING, 5))
        self.assertFalse(self.find(missing(12), speed).any())

    def test_unknown_player_speed_gives_no_evidence(self):
        self.assertFalse(self.find(missing(12), np.full(frames(12), np.nan)).any())

    def test_nothing_is_marked_outside_the_halves(self):
        speed = players((WALKING, 20))
        dead = self.find(missing(20), speed, halves=[(1, frames(5), frames(10))])
        self.assertTrue(dead[frames(6):frames(10)].all())
        self.assertFalse(dead[:frames(5)].any())
        self.assertFalse(dead[frames(10) + 1:].any())
