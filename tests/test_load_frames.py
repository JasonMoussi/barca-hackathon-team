"""Synthetic ATD fixtures only; no match assets are read by these tests."""

import importlib
import math
from pathlib import Path
import shutil
import sys
import types
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import asset_paths
from starter import load_atd


def metadata():
    root = ET.Element("main")
    # Declaration order intentionally differs from home/away and raw slot order.
    for team in ("away", "home"):
        ET.SubElement(root, "Team", id=team)
    ET.SubElement(root, "Score", idLocalTeam="home", idVisitingTeam="away")
    for player, team in (("h2", "home"), ("a", "away"), ("h1", "home")):
        ET.SubElement(root, "Player", id=player, teamId=team)
        for channel in ("s", "y", "x"):
            ET.SubElement(root, "PlayerChannel", id=player + channel,
                          playerId=player, channelId=channel)
    layout = ET.SubElement(root, "DataFormatSpecification", separator=":")
    ET.SubElement(layout, "StringRegister", name="frameCount")
    slots = ET.SubElement(layout, "SplitRegister", separator=";")
    for player in ("a", "h1", "h2"):
        slot = ET.SubElement(slots, "SplitRegister", separator=",")
        for channel in ("x", "y", "s"):
            ET.SubElement(slot, "PlayerChannelRef", playerChannelId=player + channel)
    ball = ET.SubElement(layout, "SplitRegister", separator=",")
    for channel in ("x", "y", "s"):
        ET.SubElement(ball, "BallChannelRef", channelId=channel)
    return root


MISSING = "NaN,NaN,NaN"


def record(frame=1, away="0.1,0,0", home="0,0.1,0", extra=MISSING, ball="0,0,0"):
    return "{}:{};{};{}:{}\n".format(frame, away, home, extra, ball)


class LoadFramesTests(unittest.TestCase):
    def setUp(self):
        # Python 3.13 TemporaryDirectory's private Windows ACL excludes the
        # sandbox account. A unique workspace directory inherits usable ACLs.
        self.data_dir = Path(__file__).resolve().parent / ("fixture-" + uuid4().hex)
        self.data_dir.mkdir()
        self.addCleanup(self.remove_fixture)
        paths = asset_paths(data_dir=self.data_dir, source="tactical")
        self.meta = Path(paths["atd_meta"])
        self.raw = Path(paths["atd_raw"])
        self.meta.parent.mkdir()
        self.write_xml(metadata())

    def remove_fixture(self):
        assert self.data_dir.resolve().parent == Path(__file__).resolve().parent
        shutil.rmtree(self.data_dir)

    def write_xml(self, root):
        ET.ElementTree(root).write(self.meta, encoding="utf-8")

    def load(self, text, expected=1):
        self.raw.write_text(text, encoding="utf-8")
        with patch.object(load_atd, "source_frames", return_value=expected) as count:
            result = load_atd.load_frames(self.data_dir)
        count.assert_called_once_with("tactical", load_atd.GAME)
        return result

    def test_scaling_team_minima_and_schema(self):
        rows = self.load(record() + record(2, away="0,0.01,0") +
                         record(3, home="0.2,0.2,0", extra="0.03,0.04,0"), 3)
        self.assertEqual(list(rows.columns), ["frame", "ball_seen", "nearest_team",
                         "nearest_distance_m", "home_distance_m", "away_distance_m"])
        self.assertEqual([str(dtype) for dtype in rows.dtypes],
                         ["int64", "bool", "string", "float64", "float64", "float64"])
        self.assertTrue(rows.index.equals(pd.RangeIndex(3)))
        np.testing.assert_array_equal(rows.frame, [0, 1, 2])
        self.assertEqual(list(rows.nearest_team), ["H", "A", "H"])
        self.assertAlmostEqual(rows.home_distance_m[0], 6.8)
        self.assertAlmostEqual(rows.away_distance_m[0], 10.53)
        self.assertAlmostEqual(rows.nearest_distance_m[1], 0.68)
        self.assertAlmostEqual(rows.home_distance_m[2], math.hypot(0.03 * 105.3, 0.04 * 68))

    def test_pitch_comes_from_games(self):
        with patch.dict(load_atd.GAMES, {load_atd.GAME: {"pitch": (20.0, 30.0)}}):
            row = self.load(record()).iloc[0]
        self.assertEqual(row.home_distance_m, 3.0)
        self.assertEqual(row.away_distance_m, 2.0)
        self.assertEqual(row.nearest_team, "A")

    def test_exact_tie_and_genuinely_unequal_distances(self):
        rows = self.load(record(away="0.1,0,0", home="-0.1,0,0") +
                         record(2, away="0.1000000000001,0,0", home="0.1,0,0") +
                         record(3, away="0.1,0,0", home="0.1000000000001,0,0"), 3)
        self.assertTrue(pd.isna(rows.nearest_team[0]))
        self.assertEqual(rows.nearest_distance_m[0], rows.home_distance_m[0])
        self.assertEqual(rows.nearest_team[1], "H")
        self.assertEqual(rows.nearest_team[2], "A")

    def test_zero_out_of_pitch_and_missing_speed(self):
        rows = self.load(record(home="0,0,NaN", ball="0,0,") +
                         record(2, away="2,2,inf", home="-0.2,1.3,",
                                ball="-0.1,1.1,NaN"), 2)
        self.assertTrue(rows.ball_seen.all())
        self.assertEqual(rows.home_distance_m[0], 0.0)
        self.assertEqual(rows.nearest_team[0], "H")
        self.assertAlmostEqual(rows.home_distance_m[1], math.hypot(10.53, 13.6))

    def test_missing_and_nonfinite_ball(self):
        balls = [MISSING, "NaN,0,0", "0,NaN,0", "inf,0,0", "0,-inf,0", ",0,0"]
        rows = self.load("".join(record(i, ball=ball) for i, ball in enumerate(balls, 1)), len(balls))
        self.assertFalse(rows.ball_seen.any())
        self.assertTrue(rows.nearest_team.isna().all())
        self.assertTrue(rows.iloc[:, 3:].isna().all().all())

    def test_one_or_no_teams_observed(self):
        rows = self.load(record(away="0,NaN,0") + record(2, home="inf,0,0") +
                         record(3, away="-inf,0,0", home="0,inf,0"), 3)
        self.assertTrue(rows.ball_seen.all())
        self.assertTrue(rows.nearest_team.isna().all())
        self.assertAlmostEqual(rows.nearest_distance_m[0], 6.8)
        self.assertTrue(pd.isna(rows.away_distance_m[0]))
        self.assertAlmostEqual(rows.nearest_distance_m[1], 10.53)
        self.assertTrue(pd.isna(rows.home_distance_m[1]))
        self.assertTrue(rows.iloc[2, 3:].isna().all())

    def test_reordered_slot_and_coordinate_references(self):
        root = metadata()
        layout = root.find("DataFormatSpecification")
        slots = layout[1]
        slots[:] = [slots[1], slots[2], slots[0]]  # home, home, away
        slots[0][:] = [slots[0][2], slots[0][1], slots[0][0]]  # speed,y,x
        layout[2][:] = [layout[2][2], layout[2][1], layout[2][0]]
        self.write_xml(root)
        row = self.load("1:NaN,0.1,0;NaN,NaN,NaN;0.1,0,0:NaN,0,0\n").iloc[0]
        self.assertEqual(row.nearest_team, "H")
        self.assertAlmostEqual(row.home_distance_m, 6.8)
        self.assertAlmostEqual(row.away_distance_m, 10.53)

    def test_ball_reader_uses_raw_coordinates_and_video_index(self):
        root = metadata()
        ball = root.find("DataFormatSpecification")[2]
        ball[:] = [ball[2], ball[1], ball[0]]
        self.write_xml(root)
        self.raw.write_text(record(ball="NaN,0.2,-0.1") + record(2, ball=MISSING),
                            encoding="utf-8")
        with patch.object(load_atd, "source_frames", return_value=2):
            positions = load_atd.load_ball_positions(self.data_dir)
        np.testing.assert_array_equal(positions, [[-0.1, 0.2], [np.nan, np.nan]])

    def test_invalid_xml_mappings(self):
        mutations = {
            "duplicate team": lambda r: r.append(ET.Element("Team", id="home")),
            "missing team ID": lambda r: r.find("Team").attrib.pop("id"),
            "missing score": lambda r: r.remove(r.find("Score")),
            "unknown home": lambda r: r.find("Score").set("idLocalTeam", "unknown"),
            "same teams": lambda r: r.find("Score").set("idLocalTeam", "away"),
            "unknown team": lambda r: r.find("Player").set("teamId", "unknown"),
            "duplicate player": lambda r: r.append(ET.Element("Player", id="h1", teamId="home")),
            "unknown player": lambda r: r.find("PlayerChannel").set("playerId", "unknown"),
            "duplicate channel": lambda r: r.find("PlayerChannel").set("id", "ax"),
            "unknown channel": lambda r: r.find("PlayerChannel").set("channelId", "z"),
            "unknown reference": lambda r: r.find(".//PlayerChannelRef").set("playerChannelId", "bad"),
            "mixed players": lambda r: r.find(".//PlayerChannelRef").set("playerChannelId", "h1x"),
            "duplicate axis": lambda r: r.find(".//PlayerChannelRef").set("playerChannelId", "ay"),
            "duplicate slot": lambda r: r.find(".//SplitRegister").append(r.find(".//SplitRegister")[0]),
            "missing slot": lambda r: r.find(".//SplitRegister").remove(r.find(".//SplitRegister")[0]),
            "wrong separator": lambda r: r.find("DataFormatSpecification").set("separator", "|"),
            "wrong ball": lambda r: r.find(".//BallChannelRef").set("channelId", "z"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root = metadata()
                mutate(root)
                self.write_xml(root)
                with self.assertRaisesRegex(ValueError, "invalid XML mapping"):
                    self.load(record())

    def test_malformed_xml(self):
        self.meta.write_text("<main>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid XML mapping.*line 1"):
            self.load(record())

    def test_malformed_records_even_without_ball(self):
        cases = ["\n", "1:0,0,0\n", record() + "\n", record(away="0,0"),
                 record(home="0,0,0,0"), record(ball="0,0"),
                 record(home="garbage,0,0", ball=MISSING),
                 record(away="NaN,garbage,NaN"), record(ball="0,0,bad"),
                 record(home="0,0,bad", ball=MISSING),
                 record().replace("1:", "1.0:"), record().replace("1:", "oops:")]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "line [12]"):
                    self.load(text)

    def test_slot_counts(self):
        for text in ("1:0,0,0:0,0,0\n", record(extra=MISSING + ";" + MISSING)):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "line 1, frame ID 1: unexpected slot count"):
                    self.load(text)

    def test_frame_errors(self):
        cases = [(record(0), 1, "out-of-range"), (record(2), 1, "out-of-range"),
                 (record() + record(), 2, "duplicate/out-of-order"),
                 (record() + record(2) + record(), 3, "duplicate/out-of-order"),
                 (record(2), 2, "missing frames"),
                 (record() + record(3), 3, "missing frames"),
                 (record(), 2, "EOF.*missing frames"), ("", 1, "EOF.*missing frames")]
        for text, count, error in cases:
            with self.subTest(text=text, count=count):
                with self.assertRaisesRegex(ValueError, error):
                    self.load(text, count)

    def test_distance_overflow_is_not_missing_data(self):
        with self.assertRaisesRegex(ValueError, "line 1, frame ID 1: player slot 1 distance overflow"):
            self.load(record(away="1e308,0,0", ball="-1e308,0,0"))

    def test_import_and_call_without_kloppy(self):
        with patch.dict(sys.modules, {"kloppy": None}):
            importlib.reload(load_atd)
            self.assertEqual(self.load(record()).nearest_team[0], "H")
            with self.assertRaisesRegex(ImportError, "kloppy is not installed"):
                load_atd.load(data_dir=self.data_dir)

    def test_existing_loader_still_delegates_to_kloppy(self):
        loader = Mock(return_value=object())
        module = types.ModuleType("kloppy")
        module.metrica = types.SimpleNamespace(load_tracking_epts=loader)
        self.raw.write_text(record(), encoding="utf-8")
        with patch.dict(sys.modules, {"kloppy": module}):
            result = load_atd.load(data_dir=self.data_dir)
        loader.assert_called_once_with(meta_data=str(self.meta), raw_data=str(self.raw))
        self.assertIs(result, loader.return_value)


if __name__ == "__main__":
    unittest.main()
