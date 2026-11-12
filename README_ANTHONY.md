# Anthony's possession handoff

Implement `who_has_it()` using Jason's `load_frames()` output. During live play, keep the last team in possession through a loose ball, deflection, contest, or tracking gap. Change from `H` (Barcelona) to `A` (Athletic Club), or vice versa, only when the other team shows sustained evidence of control. See [the challenge's possession definition](CHALLENGE.md#possession-rules).

## Load the frame observations

From the repo root, install the existing NumPy and pandas dependencies with `python -m pip install -r requirements.txt`. Place the authorized ATD files at `data/tactical/BAR-ATH_tactical_atd.xml` and `data/tactical/BAR-ATH_tactical_atd.txt`. Pass the actual bundle root to `data_dir` if yours is elsewhere.

```python
from starter.load_atd import load_frames

frames = load_frames(data_dir="data")
print(frames.head())
```

Load the table once. It contains 199,319 ordered rows, one per tactical video frame at 25 fps.

| Column | How to use it |
| --- | --- |
| `frame` | Zero-based tactical video frame, from 0 through 199,318 |
| `ball_seen` | Whether both raw ball coordinates are finite |
| `nearest_team` | `H` or `A` when both teams have observations and one is strictly closer |
| `home_distance_m`, `away_distance_m` | Distance in metres to the closest observed player on each team |
| `nearest_distance_m` | Smallest available team distance |

Missing distances are `NaN`. Ties and frames with only one observed team leave `nearest_team` as `pd.NA`. Check missing values with `pd.isna()`. Proximity gives you a signal to evaluate over time. Player slots can change identities, so the rule should track team-level evidence. [Loader contract and verification](starter/load_frames.md)

## Implement the possession rule

Proposed interface for Chahid to review with you is `who_has_it(frames) -> pd.Series`. It takes an ordered, contiguous slice of frame observations and returns one nullable string per input row, with the same index. Return `H`, `A`, or `pd.NA` if you have yet to establish possession in that slice.

1. **Find a candidate team.** Start with a tracked ball and two known team distances. Treat the closer team as a candidate when its nearest player is close enough to the ball and clearly closer than the opposition.
2. **Require sustained evidence.** Track how many consecutive frames support the same candidate. A gap, tie, weak distance signal, or change of candidate clears that count. Keep the confirmed owner through those interruptions.
3. **Update on confirmation.** Establish the initial owner when a candidate meets the persistence rule. Once there is an owner, switch on the frame where the opposition meets that rule. Document the distance and persistence thresholds and why you chose them.

For a synthetic test with three required frames and an established `H` owner, candidate evidence `A, missing, A, A, A` should yield `H, H, H, H, A`.

## Integrate and test

Chahid joins [Bruno's `is_dead()` output](starter/is_dead.py) to Jason's table on `frame`. `is_dead=True` produces `D`. On graded frames, both `False` and `pd.NA` defer to `who_has_it()`. Run possession from a fresh state after each dead-ball interval and at the start of each half. Chahid handles any remaining abstentions and writes the final submission CSV. [How abstention is scored](EVALUATION.md#how-an-abstention-is-scored)

Add synthetic tests for initial possession, a brief opposition touch, a loose ball, a tracking gap, a confirmed turnover, and a fresh state after a stoppage. Run them with `python -m unittest discover -s tests -v`.

Use the public five-minute ground truth for pipeline correctness checks. Use synthetic data or permitted data from other matches to choose thresholds. The competition prohibits hand labels, manual corrections, and frame-specific exceptions on the provided match. [Competition rules](RULES.md)

Work on a feature branch in the private team repository and keep match assets and local inspection exports out of Git.
