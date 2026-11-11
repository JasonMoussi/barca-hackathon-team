# Bruno: using Jason's frame loader and building is_dead()

## Status and purpose

Jason's assigned loader is complete. It reports tracking observations and player-to-ball distances for every tactical video frame. It does not determine possession or whether play is stopped.

Bruno's task is to implement is_dead(): identify periods when the ball is out of play and the restart has not yet been taken, using the supplied automatic Smart Tagging XML.

You can develop your Smart Tagging parser independently of Jason's loader. His output provides a common frame index for integration; it does not contain the Smart Tagging events or the missing ball positions.

## Files in this handoff

| File | Purpose |
| --- | --- |
| starter/load_atd.py | Updated loader, including load_frames(); original kloppy load() remains available |
| starter/load_frames.md | Detailed loader contract and original verification report |
| tests/test_load_frames.py | Synthetic tests, no match assets required |
| tests/smoke_load_frames.py | Optional full-match check, requires local tracking assets |
| games.py | Unchanged official game configuration and path helpers |
| README_BRUNO.md | This handoff |

These files live in the private team repository alongside the official source and history. Compare and merge changes on feature branches so existing team work is preserved. games.py remains the official version.

The handoff does not include match assets, Bruno's implementation, or the final possession pipeline.

## Setup

Clone the private team repository with your own authorised GitHub account:

```powershell
git clone https://github.com/JasonMoussi/barca-hackathon-team.git
cd barca-hackathon-team
git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git
```

If using Jason's prepared working copy, both remotes are already configured.
`origin` is the private team repository; `upstream` is the official source.
Keep the official history and LICENSE. Push team work only to `origin`.

From the project root, use the project's existing Python environment and dependencies:

```powershell
python -m pip install -r requirements.txt
```

Jason's load_frames() only needs NumPy and pandas. It does not need kloppy.

Place your authorised local match assets at:

- assets/tactical/BAR-ATH_tactical_atd.xml
- assets/tactical/BAR-ATH_tactical_atd.txt
- assets/tactical/BAR-ATH_tactical_smart_tagging.xml

The first two are needed for Jason's loader. The third is needed for Bruno's task and is not included in this package. Use data_dir="assets" with path helpers because the original project otherwise defaults to data/ or MTAH_DATA_DIR.

## Calling Jason's code

Run from the project root:

```python
import pandas as pd
from starter.load_atd import load_frames

frames = load_frames(data_dir="assets")
print(frames.head())
unknown_team = pd.isna(frames["nearest_team"])
```

The function returns a pandas DataFrame in memory. It does not automatically write a CSV. Call it once and pass the returned table through the pipeline rather than reloading it for each frame.

There are exactly 199,319 rows, ordered by frame, with a RangeIndex.

| Column | Type | Meaning |
| --- | --- | --- |
| frame | int64 | Tactical video index, 0 through 199,318 |
| ball_seen | bool | Both raw ball coordinates are finite; not a verified video-visibility label |
| nearest_team | nullable string | H or A only when both teams have observed players and one is strictly closer; otherwise pd.NA |
| nearest_distance_m | float64 | Minimum available team distance |
| home_distance_m | float64 | Nearest observed Barcelona player to the ball |
| away_distance_m | float64 | Nearest observed Athletic Club player to the ball |

Distances are Euclidean distances on the pitch plane in metres, with x/y scaled by the configured 105.3 × 68 m pitch. XML references determine team membership; track slots are not permanent player identities.

### Missing data and interpretation

- Missing ball: ball_seen=False, unknown team, all distances NaN.
- Ball tracked but no players: ball_seen=True, unknown team, all distances NaN.
- Only one team tracked: its distance and nearest_distance_m are available; the other distance is NaN and nearest_team is pd.NA.
- Exact tie: both distances remain available, but nearest_team is pd.NA.
- Zero distance is valid. Finite positions outside the pitch are retained.
- Missing speed does not invalidate finite x/y coordinates.
- Use pd.isna(value), not equality comparisons with NaN or pd.NA.
- Missing tracking does not mean dead ball.
- A nearby player, or a change in nearest_team, does not prove control or a turnover.
- No positions are interpolated or reconstructed.
- Malformed records or incomplete/invalid frame sequences raise errors rather than silently returning partial results.

## Bruno's implementation: where to start

Read DATA.md, CHALLENGE.md and starter/load_smart_tagging.py in the official repository before implementing. Reuse the existing standard-library Smart Tagging loader if it fits; kloppy is not required for that approach.

The official documentation establishes:

- Smart Tagging start/end values are seconds from the start of the tactical video.
- Convert to video frames using round(seconds * 25).
- These are already tactical-video times: do not apply ATD's minus-one conversion or the separate per-half offsets.
- Codes include SET PIECES; Type values include Throw-in, Corner Kick, Goal Kick, Free Kick, and kickoffs.
- Normalise text case when matching codes and tags.
- Team can be N/A; all throw-ins in the supplied tagging have that value.
- Smart Tagging is automatic and can miss events or mark them incorrectly.
- Event intervals describe phases. Their endpoints are not guaranteed to be exact dead-ball or restart boundaries.

Inspect the actual XML and helper output to establish what each interval represents. Do not automatically mark every frame in a SET PIECES interval as dead. Implement general, documented rules for finding stoppage/restart boundaries from the available evidence.

### Agree on the interface before integration

Jason's interface is implemented and fixed as described above. Bruno's interface is not implemented or agreed yet.

A simple proposed output is a table with frame and is_dead, one row for each frame 0–199,318. is_dead would be Boolean and mean the algorithm classifies this frame as out of play. Agree the function signature and return type with Chahid before coding.

If the dead-ball algorithm needs to express uncertainty, agree explicitly how that is represented and how Chahid handles it. Do not silently interpret missing event coverage as proof that play is live.

For your handoff, document:

1. Input paths and dependencies.
2. Exact callable and output types.
3. Whether interval endpoints are inclusive or exclusive.
4. How the restart frame is labelled.
5. How overlapping events, missing team tags, and uncertain periods are handled.
6. Whether any tracking features are used.
7. The commands that run your tests.

### Tests to include

Use synthetic event fixtures for timing conversion, endpoint behaviour, overlapping events, missing tags, and restarts. Check that output has every frame exactly once and that missing ball tracking alone never forces is_dead=True.

Watching this match for understanding/debugging is allowed. Creating hand labels, manual corrections, or frame-specific exceptions for this match is prohibited. The public five-minute ground truth is for pipeline correctness checks, not training, threshold tuning, or selecting the best variant. Use synthetic tests or permitted data from other games for labelled validation/tuning.

## How the team combines the work

- Jason: raw frame observations and distances; complete.
- Bruno: stoppage/restart evidence and dead-ball classification.
- Anthony: possession through loose balls and tracking gaps, changing teams only with evidence of control.
- Chahid: integrate the functions, resolve agreed precedence and uncertainty, export and validate the final frame,label CSV.

At integration, an inferred dead-ball state produces D. On live play, use Anthony's H/A decision. Use X only under an explicit abstention policy. Do not blindly carry possession across a stoppage or halftime; a restart needs a fresh possession decision.

X is allowed, but not score-free on graded frames: macroF1 counts it as a miss; spellF1 fills it from the last non-X label within the half, with backward filling at the head. A previous label can be D as well as H/A. Read EVALUATION.md before deciding the policy.

Reconstructing missing ball positions is optional additional work, not part of the completed loader and not currently assigned.

## Verification already completed

Both Codex and an independent rerun of the supplied tests passed:

```powershell
python -m unittest discover -s tests -v
python tests/smoke_load_frames.py
```

- 15 synthetic tests passed.
- Full-data output: 199,319 rows × 6 columns.
- Frame range: 0–199,318, complete and ordered.
- Ball tracked / missing: 113,488 / 85,831.
- With ball tracked: both teams 112,419; home only 170; away only 899.
- nearest_team H / A / unknown: 63,794 / 48,625 / 86,900.
- All schema, distance and missing-value assertions passed without kloppy.
- Runtime was about 13–15 seconds on the tested environments; it varies by machine.

These checks verify the implementation contract, not perfect real-world tracking or final possession accuracy. The combined pipeline is not yet verified.

## Sharing and GitHub

Use the existing private team repository:
https://github.com/JasonMoussi/barca-hackathon-team

This is a private team repository, not a public fork. Keep its visibility unchanged.
Each teammate needs access through their own GitHub account. Do not redistribute
competition match assets through GitHub, including this private repository.

For subsequent team work, start from the team's current main branch, create a
feature branch, push only that branch to origin, and open a pull request:

```powershell
git switch main
git pull --ff-only origin main
git switch -c bruno/dead-ball
# Make and test your changes; stage only the intended files.
git diff
git diff --cached
git diff --cached --name-only
git status --short
# Commit your reviewed changes, then:
git push -u origin bruno/dead-ball
```

Never push to upstream or force-push over team work. Keep the official README.md
and LICENSE, this README_BRUNO.md handoff, and starter/load_frames.md.

Preserve the existing .gitignore rules. The team additions exclude assets/,
frame_observations*.csv inspection exports, local environments, caches,
credentials, and local secret files. Do not ignore all CSV files: the official
public ground truth and baseline evaluation CSVs must remain tracked.

Check git status and the staged file list before every commit. Ignore rules do
not remove files already tracked. Never commit match videos, tracking/tagging/
homography assets, credentials, or local inspection exports. Final submission
CSVs are separate deliverables handled by Chahid.

## Official references

- Rules: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
- Data and timing: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md
- Possession definition: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md
- Evaluation and abstention: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md

Prepared for Bruno on 11 November 2026 from the reviewed implementation and official documentation. No Bruno or Anthony functions were implemented in this handoff.
