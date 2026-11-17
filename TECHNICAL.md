# AlfenLiraByte: possession from tracking data

Team AlfenLiraByte: Anthony Lahoud, Bruno Fares, Jason Moussi, Chahid Aboud. Code: https://github.com/JasonMoussi/barca-hackathon-team

## Approach and inputs

The submission generator joins two deterministic detectors on the tactical video's frame index. `load_frames()` reads the tactical ATD metadata and raw tracking file, extracting ball visibility and each team's nearest player distance. `is_dead()` reads tactical ATD positions and speeds plus Smart Tagging events to infer stoppages. The broadcast video, tactical video pixels, and homography are not used by this integration.

Only the two half blocks in `games.py` are graded. The generator starts with `X` on every frame, assigns `D` where `is_dead()` returns `True` inside a half, and calls `who_has_it()` separately on each uninterrupted live stretch. A missing or `False` stoppage result is treated as live. This resets ownership at every stoppage and at halftime.

`who_has_it()` confirms a team when its nearest player is within 2 metres of the ball, the opposing nearest player is at least 1 metre farther away, and this evidence lasts 25 consecutive frames. Once confirmed, the team owns the ball from the first frame of that confirming run. A loose ball, deflection, challenge, or tracking gap retains the prior owner. Before the first confirmed owner in a live stretch, frames receive that first owner's label; if no owner is ever confirmed, the entire stretch remains `X`.

## Result on the public five minutes

On the full match the generator writes 54,062 `H`, 32,104 `A`, 69,487 `D` and 43,666 `X` frames. 40,462 of the `X` frames lie outside the two halves; the other 3,204 are live stretches where no owner was ever confirmed. On the public five-minute window the file scores macroF1 0.889 and spellF1 0.614, with 11 predicted possessions against 17 in the ground truth, compared with 0.621 and 0.326 for `baseline_smooth50`. With only 17 possessions in that window, one missed possession moves spellF1 by about six points, so this is a pipeline check rather than an expected rank. Running the generator twice gives the same file byte for byte.

## Training data and limitations

No labelled training data or model fitting is used. The public five-minute ground truth is reserved for pipeline checks and is not used to tune rules. No hand labels, frame-specific corrections, or private ground truth are used.

The method can abstain for an entire stretch when tracking never shows 25 consecutive frames of clear control. Inferred stoppage boundaries can be wrong if player motion or ball visibility is misleading. The fixed control and confirmation thresholds can miss short possessions or delay an unconfirmed change. The public five minutes are too small to predict full-match ranking reliably.

## Reproduction

The submitted file was produced with Python 3.14.7, numpy 2.5.4 and pandas 3.1.0; `requirements-submission.txt` pins the two library versions. Place the authorized bundle under `data/` (or pass its root with `--data-dir`), then run from the repository root:

```bash
python -m pip install -r requirements-submission.txt
python -m starter.make_submission --data-dir data --output evaluation/results/AlfenLiraByte.csv
python evaluation/validate_submission.py evaluation/results/AlfenLiraByte.csv
python evaluation/run_evaluation.py
```

The generated CSV has `frame,label` and one row for each of the 199,319 tactical video frames, written with `\n` line endings so the same bytes come out on every operating system. It is excluded from Git and is uploaded separately as `AlfenLiraByte.csv`.
