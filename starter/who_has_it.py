"""Decide which team has the ball on each frame of live play.

    from starter.load_atd import load_frames
    from starter.who_has_it import who_has_it

    frames = load_frames(data_dir="assets")
    owner = who_has_it(frames.iloc[11185:12851])    # one stretch of live play

The rule follows the challenge definition: while play is live, one team has the ball,
and it keeps the ball through a loose ball, a deflection, a contest or a tracking gap.
The other team takes over only when it shows clear control for long enough.

It works in three steps, one frame at a time:

  1. Candidate. A team is a candidate on a frame when its nearest player is within reach
     of the ball and the other team's nearest player is clearly farther away. A missing
     ball, an unseen team, a loose ball or a contest gives no candidate.
  2. Count. Count the frames in a row with the same candidate. A frame with no candidate,
     or with the other team as candidate, sets the count back to zero.
  3. Confirm. When the count reaches frames_required, that team becomes the owner,
     starting from the first frame of that run. Until then the last owner stays.

Call it once per stretch of live play. Each call starts with no owner, so pass a new
slice after every stoppage and at the start of each half.
"""

import numpy as np
import pandas as pd

# The three settings below come from how football is played, not from fitting the public
# ground truth. Change them through the function arguments, not here.

# A player controls a ball that is within about one stride. Two metres also leaves room
# for the half metre or so of tracking error on the ball and on the player.
CONTROL_DISTANCE_M = 2.0
# An opponent within one more metre can still reach the ball, so that is a contest.
# The nearest opponent must be at least this much farther away than the candidate.
CLEAR_MARGIN_M = 1.0
# One second at 25 frames per second. A deflection, a blocked pass or a poke in a tackle
# keeps the ball next to a player for well under half a second. A controlled touch
# followed by a second action takes about a second.
FRAMES_REQUIRED = 25


def _candidates(frames, control_distance_m, clear_margin_m):
    """Per frame: "H" or "A" when that team clearly controls the ball, else None."""
    seen = frames["ball_seen"].to_numpy(dtype=bool)
    home = frames["home_distance_m"].to_numpy(dtype=float)
    away = frames["away_distance_m"].to_numpy(dtype=float)
    # A comparison with a missing distance (NaN) is False, so both teams must be seen.
    with np.errstate(invalid="ignore"):
        home_clear = seen & (home <= control_distance_m) & (away - home >= clear_margin_m)
        away_clear = seen & (away <= control_distance_m) & (home - away >= clear_margin_m)
    candidates = np.full(len(frames), None, dtype=object)
    candidates[home_clear] = "H"
    candidates[away_clear] = "A"
    return candidates


def who_has_it(frames, frames_required=FRAMES_REQUIRED,
               control_distance_m=CONTROL_DISTANCE_M, clear_margin_m=CLEAR_MARGIN_M) -> pd.Series:
    """Owner of the ball per row of frames: "H", "A", or missing before the first owner.

    frames is an ordered, unbroken slice of load_frames() output. The result has the
    same index and the pandas "string" type. Check missing values with pd.isna().

    A change of owner is placed on the first frame of the run that confirms it, because
    that is when the new team got the ball. After owner H, evidence A, missing, A, A, A
    with frames_required=3 gives H, H, A, A, A. The lone A before the gap is only a
    touch, so it stays with H.

    Placing a change needs the frames that follow it, so this suits a whole match read
    from a file, not a live feed.
    """
    candidates = _candidates(frames, control_distance_m, clear_margin_m)
    owners = np.full(len(frames), None, dtype=object)
    owner = None
    run_team, run_length = None, 0
    for row, team in enumerate(candidates):
        if team is None:
            run_team, run_length = None, 0
        elif team == run_team:
            run_length += 1
        else:
            run_team, run_length = team, 1

        if run_length >= frames_required and run_team != owner:
            owner = run_team
            # The team got the ball on the first frame of this run, so go back and
            # give it those frames too.
            owners[row - run_length + 1:row] = owner
        owners[row] = owner
    return pd.Series(pd.array(owners, dtype="string"), index=frames.index, name="possession")
