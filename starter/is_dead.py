"""Infer dead-ball intervals on the tactical video timeline."""

from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks
from starter.load_atd import load_ball_positions, load_player_speeds
from starter.load_smart_tagging import load as load_events

# Settings for _observed_stoppages. Each one comes from how people and a football move.
# None of them was fitted to the public ground truth.

# Adults walk at about 1.4 m/s. When the average player is slower, nobody is chasing the ball.
WALKING_PACE = 1.4
# Nobody runs faster than about 8 to 10 m/s, so a ball this fast was kicked or thrown.
PLAYED_SPEED = 8.0
# After a kick or throw the ball counts as in play for this long. It keeps slow passing
# between walking players from looking like a stoppage.
PLAY_MEMORY_S = 2.0
# The ball must be this far over a line to count as out. It covers tracking jitter.
OUT_MARGIN = 0.5
# No restart is taken in under three seconds. Shorter quiet spells are not stoppages, and
# shorter breaks inside a stoppage are not play.
MIN_STOPPAGE_S = 3.0
# Players need about this long to slow to a walk after the whistle. A stoppage found from
# walking players is moved earlier by this much.
REACTION_S = 1.5
# A ball that left the pitch has to be fetched. A break this long before the next sign of
# the stoppage (players jogging back into position) still belongs to the same stoppage.
FETCH_S = 10.0
# Player speed is averaged over this window so one noisy frame does not matter.
SMOOTH_S = 1.0


def _runs(mask):
    """Half-open intervals for contiguous true values."""
    return np.flatnonzero(np.diff(mask, prepend=False, append=False)).reshape(-1, 2)


def _observed_stoppages(positions, player_speed, halves, fps, pitch):
    """Frames where play is stopped, judged from walking players and a ball outside the lines.

    positions holds the ball in metres (NaN when unseen). player_speed is the average
    player speed in m/s (NaN when unknown). Returns one bool per frame.

    Two kinds of evidence start a stoppage:
      1. The ball is seen outside the lines.
      2. The average player walks and the ball has not been kicked or thrown lately,
         for at least MIN_STOPPAGE_S. A missing ball counts as "not kicked".
    Short breaks between such stretches are closed, and so is any break in which the
    ball is never seen. Anything shorter than MIN_STOPPAGE_S is dropped. Nothing is
    marked outside the halves.
    """
    length, width = pitch
    x, y = positions.T
    seen = np.isfinite(positions).all(axis=1)
    with np.errstate(invalid="ignore"):
        outside = seen & ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                          | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))

        # Ball speed across a quarter of a second, centred, so jitter does not look like a kick.
        reach = max(1, round(0.12 * fps))
        ball_speed = np.full(len(positions), np.nan)
        ball_speed[reach:-reach] = np.linalg.norm(
            positions[2 * reach:] - positions[:-2 * reach], axis=1) * fps / (2 * reach)
        played = seen & ~outside & (ball_speed >= PLAYED_SPEED)

        window = max(1, round(SMOOTH_S * fps))
        smooth = pd.Series(player_speed).rolling(
            window, center=True, min_periods=window // 2 + 1).mean().to_numpy()
        walking = smooth < WALKING_PACE

    # in_play: the ball was kicked or thrown within the last PLAY_MEMORY_S seconds.
    memory = max(1, round(PLAY_MEMORY_S * fps))
    kicks = np.r_[np.zeros(memory, dtype=np.int64), np.cumsum(played)]
    in_play = kicks[memory:] > kicks[:-memory]
    idle = walking & ~in_play

    shortest = round(MIN_STOPPAGE_S * fps)
    reaction = round(REACTION_S * fps)
    fetch = round(FETCH_S * fps)
    dead = np.zeros(len(positions), dtype=bool)
    for _, first, last in halves:
        half = slice(first, last + 1)
        out = outside[half]
        core = out.copy()
        for start, stop in _runs(idle[half]):
            if stop - start >= shortest:
                core[max(0, start - reaction):stop] = True

        ball_seen = seen[half]
        pieces = _runs(core)
        for (start, stop), (resume, _) in zip(pieces[:-1], pieces[1:]):
            # A break between two stoppages is closed in three cases:
            #  - it is shorter than MIN_STOPPAGE_S, because play cannot restart and stop that fast;
            #  - the ball went out just before and the break is short enough to be fetching it;
            #  - the ball is never seen in it, so nothing shows that play restarted
            #    (players jogging around after a goal look like this).
            if (resume - stop < shortest
                    or (resume - stop <= fetch and out[start:stop].any())
                    or not ball_seen[stop:resume].any()):
                core[stop:resume] = True
        for start, stop in _runs(core):
            if stop - start < shortest:
                core[start:stop] = False
        dead[half] = core
    return dead


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

    Two detectors feed True. _classify pairs a clean line crossing with its restart.
    _observed_stoppages adds stoppages seen from walking players and a ball outside
    the lines, which also works while the ball is not tracked. Its True wins over a
    False from _classify, because a ball carried back to a restart also moves.
    """
    events = load_events(data_dir=data_dir)
    pitch = GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    halves = half_video_blocks(GAME)
    positions = load_ball_positions(data_dir=data_dir) * pitch
    state = _classify(positions, events, halves, fps, pitch)
    player_speed = load_player_speeds(data_dir=data_dir)
    state[_observed_stoppages(positions, player_speed, halves, fps, pitch)] = True
    return pd.DataFrame({"frame": np.arange(len(positions), dtype=np.int64), "is_dead": state})
