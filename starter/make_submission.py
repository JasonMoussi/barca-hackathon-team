"""Combine stoppages and live-play ownership into a tactical-frame submission.

Run from the repository root:
    python -m starter.make_submission --data-dir data

Writes evaluation/results/AlfenLiraByte.csv by default. The file is named after the team
because the scorer names each row of the board after the uploaded file.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from starter.is_dead import is_dead  # noqa: E402
from starter.load_atd import load_frames  # noqa: E402
from starter.who_has_it import who_has_it  # noqa: E402


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
    parser.add_argument("--output", default="evaluation/results/AlfenLiraByte.csv",
                        help="submission CSV path (name it after the team)")
    args = parser.parse_args()

    frames = load_frames(data_dir=args.data_dir)
    dead = is_dead(data_dir=args.data_dir)
    if len(frames) != source_frames("tactical", GAME):
        raise ValueError("tactical tracking table has the wrong number of video frames")
    submission = make_labels(frames, dead)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Always "\n" line endings, so the file is byte-identical on every operating system.
    submission.to_csv(output, index=False, lineterminator="\n")
    print("Wrote {} labels to {}".format(len(submission), output))
    print(submission["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
