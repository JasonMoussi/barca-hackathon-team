"""Combine stoppages and live-play ownership into a tactical-frame submission.

Run from the repository root:
    python -m starter.make_chahid_submission --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from games import GAME, half_video_blocks, source_frames
from starter.is_dead import is_dead
from starter.load_atd import load_frames
from starter.who_has_it import who_has_it


def make_labels(frames, dead, halves=None):
    """Return one H/A/D/X label per video frame, resetting ownership at every stop."""
    if halves is None:
        halves = half_video_blocks(GAME)
    expected = len(frames)
    if len(dead) != expected:
        raise ValueError("tracking and stoppage tables have different row counts")
    video_frames = np.arange(expected, dtype=np.int64)
    if (not np.array_equal(frames["frame"].to_numpy(), video_frames)
            or not np.array_equal(dead["frame"].to_numpy(), video_frames)):
        raise ValueError("input tables must contain the same ordered video frames from zero")

    labels = np.full(expected, "X", dtype="<U1")
    stopped = dead["is_dead"].fillna(False).to_numpy(dtype=bool)
    for _, first, last in halves:
        if first < 0 or last >= expected or first > last:
            raise ValueError("half boundaries must be within the input frame range")
        labels[first:last + 1][stopped[first:last + 1]] = "D"
        live = ~stopped[first:last + 1]
        boundaries = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for start, end in boundaries.reshape(-1, 2):
            start += first
            end += first
            owners = who_has_it(frames.iloc[start:end])
            if len(owners) != end - start:
                raise ValueError("owner detector returned the wrong number of frames")
            known = np.flatnonzero(owners.notna().to_numpy())
            if not len(known):
                continue
            values = owners.to_numpy(dtype=object, na_value=None)
            values[:known[0]] = values[known[0]]
            if any(value not in ("H", "A") for value in values):
                raise ValueError("owner detector returned an invalid or missing label")
            labels[start:end] = values
    return pd.DataFrame({"frame": video_frames, "label": labels})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None,
                        help="bundle root containing tactical/ (default: MTAH_DATA_DIR or data/)")
    parser.add_argument("--output", default="evaluation/results/chahid.csv",
                        help="submission CSV path")
    args = parser.parse_args()

    frames = load_frames(data_dir=args.data_dir)
    dead = is_dead(data_dir=args.data_dir)
    if len(frames) != source_frames("tactical", GAME):
        raise ValueError("tactical tracking table has the wrong number of video frames")
    submission = make_labels(frames, dead)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print("Wrote {} labels to {}".format(len(submission), output))
    print(submission["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
