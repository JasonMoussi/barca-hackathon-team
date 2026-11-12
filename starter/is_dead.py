"""Infer dead-ball intervals on the tactical video timeline."""

from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks
from starter.load_atd import load_ball_positions
from starter.load_smart_tagging import load as load_events


def _runs(mask):
    """Half-open intervals for contiguous true values."""
    return np.flatnonzero(np.diff(mask, prepend=False, append=False)).reshape(-1, 2)


def _line_crossings(lines, coherent, short_step):
    """First outside frame, corroborated by three samples on one side of the line."""
    inside = np.all(lines <= 0, axis=1)
    outside = lines > 0
    before, edges = np.nonzero((inside[:-1] & coherent[1:])[:, None] & outside[1:])
    crossings = []
    for previous, edge in zip(before, edges):
        frame = previous + 1
        if previous == 0 or frame + 2 >= len(lines) or outside[frame].sum() != 1:
            continue
        if not (inside[previous - 1] and short_step[previous] and coherent[previous]):
            continue
        following = slice(frame + 1, frame + 3)
        if (np.all(short_step[following] & coherent[following] & outside[following, edge])
                and np.max(lines[frame:frame + 3, edge]) >= 0.35):
            crossings.append((int(frame), int(edge)))
    return crossings


def _event_windows(events):
    windows = defaultdict(list)
    for event in events:
        code = (event["code"] or "").strip().casefold()
        start, stop = event["start_frame"], event["end_frame"]
        if start >= stop:
            continue
        if code == "goals":
            windows[code].append((start, stop))
        elif code == "set pieces":
            for key, value in event["tags"].items():
                if (key or "").strip().casefold() == "type" and value:
                    windows[value.strip().casefold()].append((start, stop))
    return windows


def _apply_intervals(state, intervals):
    """Deduplicate identical intervals and abstain on conflicting boundaries."""
    groups = []
    for start, restart in sorted(set(intervals)):
        if groups and start <= groups[-1][1]:
            first, last, _ = groups.pop()
            groups.append((first, max(last, restart), True))
        else:
            groups.append((start, restart, False))
    for start, restart, conflict in groups:
        if conflict:
            state[start:restart + 1] = pd.NA
        else:
            state[start:restart] = True
            state[restart] = False


def _classify(positions, events, halves, fps, pitch):
    state = pd.array(np.full(len(positions), None, dtype=object), dtype="boolean")
    motion_frames = max(2, round(0.24 * fps))
    if len(positions) <= motion_frames + 2:
        return state

    length, width = pitch
    x, y = positions.T
    lines = np.column_stack((-x, x - length, -y, y - width))
    inside = np.all(lines <= 0, axis=1)
    observed = np.isfinite(positions).all(axis=1)
    steps = np.linalg.norm(np.diff(positions, axis=0, prepend=np.nan), axis=1)
    coherent = steps <= 45 / fps

    # A tracked ball moving coherently inside the pitch is evidence of live play.
    # Use a run of samples so that a single displacement cannot establish play.
    good_step = inside[1:] & inside[:-1] & coherent[1:]
    cumulative = np.r_[0, good_step.cumsum()]
    motion = np.zeros(len(positions), dtype=bool)
    motion[:-motion_frames] = (
        (cumulative[motion_frames:] - cumulative[:-motion_frames] == motion_frames)
        & (np.linalg.norm(positions[motion_frames:] - positions[:-motion_frames], axis=1) >= 0.75)
    )
    moving = np.zeros_like(motion)
    for start, stop in _runs(motion):
        moving[start:stop + motion_frames] = True

    # Allow a short gap at the crossing itself, with continuous observations
    # on both sides. Use the first observed outside frame as the boundary.
    seen_frames = np.flatnonzero(observed)
    seen_lines = lines[observed]
    gaps = np.diff(seen_frames)
    observed_steps = np.linalg.norm(np.diff(positions[seen_frames], axis=0), axis=1)
    plausible = (gaps <= max(1, round(0.2 * fps))) & (observed_steps <= 45 * gaps / fps)
    exits = [
        (int(seen_frames[frame]), edge)
        for frame, edge in _line_crossings(seen_lines, np.r_[False, plausible],
                                           np.r_[False, gaps == 1])
    ]
    entries = sorted(
        (int(seen_frames[-frame]), edge, False)
        for frame, edge in _line_crossings(seen_lines[::-1], np.r_[False, plausible[::-1]],
                                           np.r_[False, gaps[::-1] == 1])
        if edge >= 2
    )
    windows = _event_windows(events)
    for frame in np.flatnonzero(motion[1:] & ~observed[:-1]) + 1:
        edge = int(np.argmax(lines[frame]))
        if edge >= 2 and -2 <= lines[frame, edge] <= 0:
            entries.append((int(frame), edge, True))
    throws = defaultdict(list)
    for frame, edge, after_gap in entries:
        if motion[frame] and lines[frame, edge] - lines[frame + motion_frames, edge] >= 0.6:
            throws[edge].append((frame, "throw-in", after_gap))

    # Measure a placed ball over a window so single-frame tracking jitter does
    # not break the pause. A kick still needs a coherent departure.
    launches = []
    kicks = defaultdict(list)
    pause_frames = max(2, round(0.6 * fps))
    pause_frames = min(pause_frames, len(positions) - 1)
    setup_windows = np.lib.stride_tricks.sliding_window_view(positions, pause_frames + 1, axis=0)
    still = (
        (np.linalg.norm(np.ptp(setup_windows, axis=2), axis=1) <= 0.45)
        & (cumulative[pause_frames:] - cumulative[:-pause_frames] == pause_frames)
    )
    for setup, end in _runs(still):
        possible = np.arange(setup + pause_frames + 1, min(end + pause_frames + 1, len(positions)))
        departures = possible[motion[possible - 1] & (steps[possible] >= 0.08)]
        if not len(departures):
            continue
        restart = int(departures[0])
        launches.append((int(setup), restart))
        at = positions[restart - 1]
        edge = int(at[0] > length / 2)
        goal_line = -lines[restart - 1, edge]
        touchline = min(at[1], width - at[1])
        if goal_line <= 2.5 and touchline <= 2.5:
            kicks[edge].append((restart, "corner kick", False))
        elif goal_line <= 6.5 and abs(at[1] - width / 2) <= 10:
            kicks[edge].append((restart, "goal kick", False))

    candidates = {**kicks, **throws}
    for restarts in candidates.values():
        restarts.sort()
    exit_frames = [frame for frame, _ in exits]
    center_launches = [
        restart for _, restart in launches
        if np.linalg.norm(positions[restart - 1] - np.array(pitch) / 2) <= 2
    ]

    for _, first, last in halves:
        stop = last + 1
        state[first:stop] = pd.array(
            np.where(moving[first:stop], False, None), dtype="boolean"
        )
        intervals = []
        half_exits = exits[bisect_left(exit_frames, first):bisect_left(exit_frames, stop)]

        for index, (exit_frame, edge) in enumerate(half_exits):
            deadline = min(stop, exit_frame + round(35 * fps) + 1)
            if index + 1 < len(half_exits):
                deadline = min(deadline, half_exits[index + 1][0])
            # An apparent goal is paired with a center kickoff below.
            if edge < 2 and abs(y[exit_frame] - width / 2) <= 4.16 and any(
                start - 2 * fps <= exit_frame < start + 3 * fps
                for start, _ in windows["goals"]
            ):
                continue
            for restart, kind, after_gap in candidates.get(edge, ()):
                if not exit_frame < restart < deadline:
                    continue
                tagged = any(start <= restart < end for start, end in windows[kind])
                if restart - exit_frame < round((0.5 if tagged else 1) * fps):
                    continue
                if after_gap and not tagged:
                    continue
                if edge >= 2 and abs(x[restart] - x[exit_frame]) > 15:
                    continue
                # A later restart cannot account for an intervening run of play away
                # from the restart area. Ball retrieval around the setup is possible.
                observed_play = positions[exit_frame:restart][motion[exit_frame:restart]]
                if np.any(np.linalg.norm(observed_play - positions[restart], axis=1) > 12):
                    break
                intervals.append((exit_frame, restart))
                break

        for start, end in windows["free kick"]:
            matches = [
                (setup, restart) for setup, restart in launches
                if first <= setup < restart < stop
                and start <= restart < end
                and restart - setup >= 2 * fps
                and setup > 0 and inside[setup - 1] and coherent[setup]
                and moving[setup - 1]
            ]
            if len(matches) == 1:
                intervals.append(matches[0])

        for kind in ("kick off (start)", "kick off (after goal)"):
            for start, end in windows[kind]:
                restarts = [r for r in center_launches if max(first, start) <= r < min(stop, end)]
                if len(restarts) != 1:
                    continue
                restart = restarts[0]
                if kind == "kick off (start)":
                    if first < restart:
                        intervals.append((first, restart))
                    continue
                goal_starts = [
                    goal_start for goal_start, goal_end in windows["goals"]
                    if goal_start < start < goal_end
                ]
                goal_exits = [
                    frame for frame, edge in half_exits
                    if edge < 2 and abs(y[frame] - width / 2) <= 4.16 and frame < restart
                    and any(s - 2 * fps <= frame < s + 3 * fps for s in goal_starts)
                ]
                if len(goal_exits) == 1:
                    intervals.append((goal_exits[0], restart))

        _apply_intervals(state, intervals)
    return state


def is_dead(data_dir=None) -> pd.DataFrame:
    """Return frame (int64) and is_dead (nullable Boolean) for the tactical video.

    True covers inferred stoppages up to, but excluding, the restart frame.
    False indicates evidence of live play. NA indicates insufficient evidence
    or a frame outside the graded halves. The function reads the local tactical
    Smart Tagging XML and raw ATD with the usual games.py path conventions.
    """
    events = load_events(data_dir=data_dir)
    pitch = GAMES[GAME]["pitch"]
    positions = load_ball_positions(data_dir=data_dir) * pitch
    state = _classify(positions, events, half_video_blocks(GAME), GAMES[GAME]["fps"], pitch)
    return pd.DataFrame({"frame": np.arange(len(positions), dtype=np.int64), "is_dead": state})
