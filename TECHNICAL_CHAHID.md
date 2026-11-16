# Chahid possession integration

## Approach and inputs

The submission generator joins two deterministic detectors on the tactical video's frame index. `load_frames()` reads the tactical ATD metadata and raw tracking file, extracting ball visibility and each team's nearest player distance. `is_dead()` reads tactical ATD positions and speeds plus Smart Tagging events to infer stoppages. The broadcast video, tactical video pixels, and homography are not used by this integration.

Only the two half blocks in `games.py` are graded. The generator starts with `X` on every frame, assigns `D` where `is_dead()` returns `True` inside a half, and calls `who_has_it()` separately on each uninterrupted live stretch. A missing or `False` stoppage result is treated as live. This resets ownership at every stoppage and at halftime.

`who_has_it()` confirms a team when its nearest player is within 2 metres of the ball, the opposing nearest player is at least 1 metre farther away, and this evidence lasts 25 consecutive frames. Once confirmed, the team owns the ball from the first frame of that confirming run. A loose ball, deflection, challenge, or tracking gap retains the prior owner. Before the first confirmed owner in a live stretch, frames receive that first owner's label; if no owner is ever confirmed, the entire stretch remains `X`.

## Training data and limitations

No labelled training data or model fitting is used. The public five-minute ground truth is reserved for pipeline checks and is not used to tune rules. No hand labels, frame-specific corrections, or private ground truth are used.

The method can abstain for an entire stretch when tracking never shows 25 consecutive frames of clear control. Inferred stoppage boundaries can be wrong if player motion or ball visibility is misleading. The fixed control and confirmation thresholds can miss short possessions or delay an unconfirmed change. The public five minutes are too small to predict full-match ranking reliably.

## Reproduction

Install `requirements.txt`, place the authorized bundle under `data/` (or pass its root with `--data-dir`), then run:

```bash
python -m starter.make_chahid_submission --data-dir data --output evaluation/results/chahid.csv
python evaluation/validate_submission.py evaluation/results/chahid.csv
python evaluation/run_evaluation.py
```

The generated CSV has `frame,label` and one row for each of the 199,319 tactical video frames. It is excluded from Git and must be uploaded separately.
