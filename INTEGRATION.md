# Integration handoff: from the two detectors to the submission file

Join Bruno's `is_dead()` and Anthony's `who_has_it()` into one label per frame, and write the submission CSV. A frame is `D` when play is stopped, `H` (Barcelona) or `A` (Athletic Club) when a team has the ball in live play, and `X` when there is no answer. See [the submission format](SUBMISSION.md).

## Load the inputs

From the repo root, install the pinned NumPy and pandas versions with `python -m pip install -r requirements-submission.txt`. Unpack the authorized match bundle so that `data/tactical/` holds the tactical files, which is where [games.py](games.py) looks by default. Pass the actual bundle root to `data_dir` if yours is elsewhere. It is the folder that holds `tactical/`, not a file.

```python
from starter.is_dead import is_dead
from starter.load_atd import load_frames

frames = load_frames(data_dir="data")
dead = is_dead(data_dir="data")
```

Load each table once. Both have 199,319 ordered rows, one per tactical video frame at 25 fps.

| Function | What it gives |
| --- | --- |
| `load_frames(data_dir)` | Jason's table: `ball_seen` and the distance from the ball to each team's nearest player. About 15 seconds. |
| `is_dead(data_dir)` | `frame` and `is_dead`. `True` means play is stopped. `False` and missing both mean "not known to be stopped". About 7 seconds. |
| `who_has_it(frames)` | Takes an ordered, unbroken slice of Jason's table. Returns `H`, `A` or missing for each row, with the same index. Check missing values with `pd.isna()`. |

`is_dead()` changed since Bruno's version. It used to mark 9 stoppages in the whole match, which last 67 seconds when added together. It now marks 105, which last 46 minutes added together, or 44% of the graded frames. The reason: it also treats play as stopped when the average player walks and nobody plays the ball, and when the ball is seen outside the lines. Bruno's detector still runs unchanged, and the return format is the same.

## Join them into one label per frame

1. **Start with `X` everywhere.** Only the two halves are graded. Get their first and last frames from `half_video_blocks()` in [games.py](games.py). Frames outside them stay `X`.
2. **Mark the stoppages.** Inside each half, a frame with `is_dead=True` becomes `D`. Every other frame goes to the next step.
3. **Ask for the owner, one stretch at a time.** Split each half into stretches of live play: the unbroken runs of frames between two stoppages. Call `who_has_it()` once per stretch and write its `H` and `A` answers. Each call starts with no owner, which gives the fresh start after every stoppage and at the start of each half.
4. **Decide what to do with the missing answers.** There are two kinds. The frames between a restart and the first confirmed owner come back missing. [CHALLENGE.md](CHALLENGE.md) says a restart always has a clear possessor, so giving those frames to the first owner of the stretch is a sound choice. A stretch with no owner at all should stay `X`. [How abstention is scored](EVALUATION.md#how-an-abstention-is-scored)
5. **Write the CSV.** A header line `frame,label`, then 199,319 rows in frame order.

One thing differs from the original handoff. Its example said that after owner `H`, the evidence `A, missing, A, A, A` with three required frames gives `H, H, H, H, A`: the owner changed on the last frame of the confirmation. `who_has_it()` gives `H, H, A, A, A` instead. The change goes on the first frame of the unbroken run, because that is when the new team got the ball. With the real setting of 25 frames in a row, the original way would make every change of owner one second late, and [EVALUATION.md](EVALUATION.md) puts "perfect but 1 s late" at spellF1 0.65.

## Check and test

Add tests on made-up tables for the join: `D` where `is_dead` is `True`, a fresh owner after each stoppage, `X` outside the halves, and the right number of rows. Run them with `python -m unittest discover -s tests -v`. Check the file with `python evaluation/validate_submission.py <your file>`, then score it with `python evaluation/run_evaluation.py`, which reads every CSV in `evaluation/results/`.

For comparison, a temporary join that follows the five steps above (not in the repo) gave `H` 54,062, `A` 32,104, `D` 69,487 and `X` 43,666 rows. 40,462 of the `X` rows are outside the halves, and the other 3,204 are 9 stretches with no owner. On the public five minutes it scored macroF1 0.889 and spellF1 0.614, against 0.621 and 0.326 for `baseline_smooth50`. Read that as "the frames line up and the labels mean what we think", not as our expected rank: it is 17 possessions, and one missed possession moves spellF1 by about six points.

Use the public five-minute ground truth for pipeline correctness checks only. The competition prohibits hand labels, manual corrections, and frame-specific exceptions on the provided match. The repository must be able to regenerate the submitted file exactly, so commit the script that writes it. [Competition rules](RULES.md)

Work on a feature branch in the private team repository and keep match assets and local inspection exports out of Git.

## Reproduce the integration

From the repository root, with the authorized bundle unpacked so that its root contains `tactical/`:

```bash
python -m pip install -r requirements-submission.txt
python -m unittest discover -s tests -v
python -m starter.make_submission --data-dir data --output evaluation/results/AlfenLiraByte.csv
python evaluation/validate_submission.py evaluation/results/AlfenLiraByte.csv
python evaluation/run_evaluation.py
```

If the bundle is elsewhere, replace `data` with its root directory, or omit `--data-dir` after setting `MTAH_DATA_DIR`. The generator streams the raw tracking file three times (once for the distances, twice inside the stoppage detector), which takes under a minute. The CSV is named after the team because the scorer names each row of the board after the uploaded file. It is ignored by Git; upload the generated file separately. The public scorer writes its reports under `evaluation/`.
