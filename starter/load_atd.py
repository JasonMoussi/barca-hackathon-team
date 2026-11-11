"""Load the provided Automatic Tracking Data and line it up with the video frames you submit.

The ATD is Metrica's FIFA/EPTS export -- a metadata XML plus a raw text file:

    data/tactical/BAR-ATH_tactical_atd.xml
    data/tactical/BAR-ATH_tactical_atd.txt

games.asset_paths() builds both, so nothing here spells a filename out.

The match was recorded twice and each recording has its own ATD. This module defaults to the
tactical one, because that is the recording a submission is indexed to. The broadcast ATD is a
separate 164488-frame dataset on its own timeline:

    load(source="broadcast")

load() delegates to kloppy. load_frames() streams the tactical pair into observation-only
distances using numpy and pandas, without kloppy or a full tracking dataset:

    from starter.load_atd import load_frames
    frames = load_frames(data_dir="assets")

Both use the same join between tracking frames and video frames.

    pip install -r requirements-starter.txt
    python starter/load_atd.py                  # prints a summary and a few aligned frames

What you actually need to know
------------------------------
The ATD is **already on the video's timeline**, one record per video frame, 199319 of them. The
only catch is that it counts from 1 where your submission counts from 0:

    video_frame = frame_id - 1

That is the whole alignment story for participants. You do not need the per-half offsets in
games.py -- those describe the separate counter the ground truth was derived from, which you never
see. Verified on the delivered file: 199319 records numbered 1..199319, with 21-22 tracked objects
during play and almost none across the halftime break, exactly where the ground truth says the
match is not being played.
"""

import os
import sys
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames  # noqa: E402

# Params
# Raw ATD line: "<frame_id>:<player>;<player>;...:<ball>". Only the part before the first
# separator is needed to check the alignment.
FIELD_SEP = ":"


def atd_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(meta_data, raw_data) paths, or a readable error naming what is missing."""
    return tuple(require(["atd_meta", "atd_raw"], game, data_dir, source))


def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The ATD as a kloppy TrackingDataset.

    Coordinates come back in kloppy's Metrica convention. The real pitch is
    GAMES[game]["pitch"] metres, which is what you want for anything measured in metres rather
    than in fractions of a pitch.
    """
    try:
        from kloppy import metrica
    except ImportError:
        raise ImportError(
            "kloppy is not installed. It is optional -- the evaluation needs only numpy and "
            "pandas -- but this loader needs it:\n    pip install -r requirements-starter.txt")

    meta, raw = atd_paths(game, data_dir, source)
    return metrica.load_tracking_epts(meta_data=meta, raw_data=raw)


def video_frame_of(frame_id):
    """Video frame -- and therefore submission row -- of an ATD record."""
    return int(frame_id) - 1


def frame_id_of(video_frame):
    """ATD frame_id of a video frame. The inverse of video_frame_of."""
    return int(video_frame) + 1


def _frame_layout(meta_path):
    """Resolve raw slots through channel -> player -> team XML references."""
    try:
        root = ET.parse(meta_path).getroot()

        def indexed(tag):
            entries = list(root.iter(tag))
            result = {entry.get("id"): entry for entry in entries}
            if not entries or not all(result) or len(result) != len(entries):
                raise ValueError("{} IDs must be present and unique".format(tag))
            return result

        teams = indexed("Team")
        scores = list(root.iter("Score"))
        if len(teams) != 2 or len(scores) != 1:
            raise ValueError("expected two teams and one Score home/away mapping")
        home = scores[0].get("idLocalTeam")
        away = scores[0].get("idVisitingTeam")
        if home == away or {home, away} != set(teams):
            raise ValueError("Score must reference distinct, known home and away teams")
        team_index = {home: 0, away: 1}
        players = indexed("Player")
        if any(player.get("teamId") not in teams for player in players.values()):
            raise ValueError("Player references an unknown or missing teamId")
        channels = indexed("PlayerChannel")
        if any(channel.get("playerId") not in players or
               channel.get("channelId") not in ("x", "y", "s")
               for channel in channels.values()):
            raise ValueError("PlayerChannel references an unknown player or channel")

        formats = list(root.iter("DataFormatSpecification"))
        if len(formats) != 1:
            raise ValueError("expected one DataFormatSpecification")
        layout = formats[0]
        if (layout.get("separator") != FIELD_SEP or len(layout) != 3 or
                layout[0].tag != "StringRegister" or
                layout[0].get("name") != "frameCount" or
                layout[1].tag != "SplitRegister" or layout[1].get("separator") != ";"):
            raise ValueError("expected frame:players:ball data layout")

        def column_order(register, tag, attribute):
            if (register.tag != "SplitRegister" or register.get("separator") != "," or
                    len(register) != 3 or any(ref.tag != tag for ref in register)):
                raise ValueError("expected three comma-separated {} entries".format(tag))
            refs = [ref.get(attribute) for ref in register]
            if tag == "PlayerChannelRef":
                if any(ref not in channels for ref in refs):
                    raise ValueError("unknown PlayerChannelRef")
                names = [channels[ref].get("channelId") for ref in refs]
            else:
                names = refs
            if set(names) != {"x", "y", "s"}:
                raise ValueError("each slot must reference x, y and speed exactly once")
            return refs, (names.index("x"), names.index("y"))

        slots = []
        used_players, used_channels = set(), set()
        for slot, register in enumerate(layout[1], 1):
            refs, xy = column_order(register, "PlayerChannelRef", "playerChannelId")
            ids = {channels[ref].get("playerId") for ref in refs}
            if len(ids) != 1 or ids & used_players:
                raise ValueError("slot {} has mixed or duplicate player references".format(slot))
            player_id = ids.pop()
            used_players.add(player_id)
            used_channels.update(refs)
            slots.append((team_index[players[player_id].get("teamId")], xy))
        if used_players != set(players) or used_channels != set(channels):
            raise ValueError("slot mapping must cover every player and channel exactly once")
        _, ball_xy = column_order(layout[2], "BallChannelRef", "channelId")
        return slots, ball_xy
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("{}: invalid XML mapping: {}".format(meta_path, exc)) from exc


def _position(group, xy, context):
    """Validate all three fields, but let only finite x/y determine visibility."""
    fields = group.split(",")
    if len(fields) != 3:
        raise ValueError("{}: expected x,y,speed (three fields)".format(context))
    try:
        values = [float(field) if field.strip() else math.nan for field in fields]
    except ValueError as exc:
        raise ValueError("{}: nonnumeric observation {!r}".format(context, group)) from exc
    x, y = values[xy[0]], values[xy[1]]
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def load_frames(data_dir="assets") -> pd.DataFrame:
    """Stream tactical ATD into frame observations and nearest-player distances.

    Uses GAMES pitch dimensions in metres and frame_id - 1, without half offsets.
    Missing observations produce NaN distances. nearest_team is nullable and is
    set only when both teams are seen and one is strictly closer. No possession
    or dead-ball inference is performed. See load_frames.md for the handoff.
    """
    meta_path, raw_path = atd_paths(GAME, data_dir, source="tactical")
    slots, ball_xy = _frame_layout(meta_path)
    expected = source_frames("tactical", GAME)
    length, width = GAMES[GAME]["pitch"]
    ball_seen = np.zeros(expected, dtype=bool)
    distances = np.full((expected, 2), np.nan, dtype=np.float64)
    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            context = "{}: line {}".format(raw_path, line_number)
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("{}: expected frame:players:ball record".format(context))
            if not parts[0].isascii() or not parts[0].isdigit():
                raise ValueError("{}: invalid frame ID {!r}".format(context, parts[0]))
            frame_id = int(parts[0])
            context += ", frame ID {}".format(frame_id)
            if not 1 <= frame_id <= expected:
                raise ValueError("{}: out-of-range frame ID; expected 1..{}".format(context, expected))
            if frame_id != count + 1:
                reason = "duplicate/out-of-order frame ID" if frame_id <= count else "missing frames"
                raise ValueError("{}: {}; expected frame ID {}".format(context, reason, count + 1))
            groups = parts[1].split(";")
            if len(groups) != len(slots):
                raise ValueError("{}: unexpected slot count {}; expected {}".format(
                    context, len(groups), len(slots)))
            ball = _position(parts[2], ball_xy, context + ", ball")
            frame = video_frame_of(frame_id)
            ball_seen[frame] = ball is not None
            for slot, (group, (team, xy)) in enumerate(zip(groups, slots), 1):
                player = _position(group, xy, context + ", player slot {}".format(slot))
                if ball is None or player is None:
                    continue
                distance = math.hypot((player[0] - ball[0]) * length,
                                      (player[1] - ball[1]) * width)
                if not math.isfinite(distance):
                    raise ValueError("{}: player slot {} distance overflow".format(context, slot))
                if math.isnan(distances[frame, team]) or distance < distances[frame, team]:
                    distances[frame, team] = distance
            count += 1
    if count != expected:
        raise ValueError("{}: EOF after {} lines: missing frames; expected frame IDs {}..{}".format(
            raw_path, count, count + 1, expected))

    home, away = distances[:, 0], distances[:, 1]
    nearest_team = pd.array([pd.NA] * expected, dtype="string")
    nearest_team[home < away] = "H"
    nearest_team[away < home] = "A"
    return pd.DataFrame({
        "frame": np.arange(expected, dtype=np.int64),
        "ball_seen": ball_seen,
        "nearest_team": nearest_team,
        "nearest_distance_m": np.fmin(home, away),
        "home_distance_m": home,
        "away_distance_m": away,
    })


def check_alignment_raw(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The same check as check_alignment, straight off the raw file. No kloppy, no full load.

    Loading 199319 frames x 46 slots through kloppy to count them costs minutes and gigabytes;
    the raw file answers the same question in one streaming pass, because every line begins with
    its own frame id. Use this one unless you already have a dataset in hand.
    """
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    expected = source_frames(source, game)

    first = last = None
    count = 0
    with open(raw_path) as handle:
        for line in handle:
            frame_id = int(line[:line.index(FIELD_SEP)])
            if first is None:
                first = frame_id
            last = frame_id
            count += 1

    contiguous = count > 0 and last - first + 1 == count
    return {
        "n_records": count,
        "n_video_frames": expected,
        "frame_id_range": (first, last),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(first) if first is not None else None,
        "agrees": count == expected and contiguous and first == 1,
    }


def check_alignment(dataset, game=GAME, source=DEFAULT_SOURCE):
    """Confirm the ATD really is one record per video frame, numbered from 1.

    Cheap, and worth running once against any new export rather than trusting the note above.
    Two claims, both falsifiable: the record count equals the video length, and the ids are
    exactly 1..n with nothing missing.
    """
    expected = source_frames(source, game)
    ids = sorted(int(record.frame_id) for record in dataset.records)
    contiguous = bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))
    return {
        "n_records": len(ids),
        "n_video_frames": expected,
        "frame_id_range": (ids[0], ids[-1]) if ids else (None, None),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(ids[0]) if ids else None,
        "agrees": len(ids) == expected and contiguous and bool(ids) and ids[0] == 1,
    }


if __name__ == '__main__':
    dataset = load()
    print("loaded {} tracking records".format(len(dataset.records)))
    print("teams: {}  ->  H = {}, A = {}".format(
        [str(team) for team in dataset.metadata.teams],
        dataset.metadata.teams[0], dataset.metadata.teams[1]))
    print("pitch: {} x {} m from the ATD metadata, {} x {} m in games.py".format(
        dataset.metadata.pitch_dimensions.pitch_length,
        dataset.metadata.pitch_dimensions.pitch_width,
        *GAMES[GAME]["pitch"]))
    print("video: {} frames at {} fps".format(
        GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]))

    report = check_alignment(dataset)
    print("\nalignment check")
    print("  records {}, video frames {}".format(report["n_records"], report["n_video_frames"]))
    print("  frame_id range {}..{}, contiguous: {}".format(
        report["frame_id_range"][0], report["frame_id_range"][1], report["contiguous"]))
    if report["agrees"]:
        print("\n  OK -- one record per video frame, numbered from 1.")
        print("  video_frame = frame_id - 1. Nothing else to do.")
    else:
        print("\n  MISMATCH -- this export is not one record per video frame from 1.")
        print("  Do not invent a correction. Report it, so every team gets the same alignment.")

    print("\nsome landmarks, as video frames:")
    for video_frame, what in ((11185, "1st half kickoff, first graded frame"),
                              (18684, "end of the public 5-minute window"),
                              (100000, "halftime -- no play"),
                              (113742, "2nd half kickoff"),
                              (190197, "last graded frame")):
        print("  video {:>6}  = ATD frame_id {:>6}   {}".format(
            video_frame, frame_id_of(video_frame), what))
