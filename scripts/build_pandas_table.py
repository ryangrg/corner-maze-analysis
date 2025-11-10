from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sqlite3
import ast

#Global Variables
SEED_CLUSTERING = 100002
# KMEANS_INIT_CENTS_
# seed 100002 splits into the nicest clusters

# set panda display options
pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.expand_frame_repr', False)

# DATA PROCESSING HELPER FUNCTIONS

def approach_to_goal_from_cue(cue_goal_assignment):
    """
    Classify whether the rat should approach a goal by moving toward or away from its cue.
    The cue/goal assignment pairs (e.g., 'N/NE') describe the relationship between cue location
    and goal corner during acquisition trials. This helper reduces those assignments to a
    simpler directional rule used downstream in scoring and visualization.

    Parameters
    ----------
    cue_goal_assignment : str
        Text encoding of cue orientation and goal corner separated by '/', e.g., 'N/NE'.

    Returns
    -------
    str or None
        'toward' when the cue and goal share the same hemifield, 'away' when they are opposite.
        Returns None if the mapping is not recognized (callers should handle this explicitly).

    Behavior / Notes
    ----------------
    - Only the canonical North cue assignments are currently listed; extend this mapping if new
      cue orientations appear in the database.
    - The return value feeds into categorical analyses; keep the labels consistent across scripts.

    Examples
    --------
    >>> approach_to_goal_from_cue('N/NE')
    'toward'
    >>> approach_to_goal_from_cue('N/SW')
    'away'
    """
    if cue_goal_assignment == 'N/NE' or cue_goal_assignment == 'N/NW':
        return 'toward'
    elif cue_goal_assignment == 'N/SE' or cue_goal_assignment == 'N/SW':
        return 'away'


def correct_route(start_arm, goal_location):
    """
    Determine the two-step turn sequence required to travel from a start arm to a corner goal.
    This function maps a cardinal start arm (one of the four compass arms of a maze)
    to a corner goal (one of the four intercardinal corners) and returns a two-character
    string describing the sequence of turns to reach that corner directly. Each character in the
    returned string represents a turn at a decision point and is one of:
    - 'L' : turn left
    - 'R' : turn right
    The first character corresponds to the first decision/intersection encountered after
    leaving the start arm; the second character corresponds to the next decision needed
    to arrive at the specified corner.
    Parameters
    ----------
    start_arm : str
        The arm from which navigation begins. Expected exact values (case-sensitive):
        'North', 'East', 'South', or 'West'.
    goal_location : str
        The corner/goal location to reach. Expected exact values (case-sensitive):
        'Northeast', 'Southeast', 'Southwest', or 'Northwest'.
    Returns
    -------
    str or None
        A two-character string composed of 'L' and 'R' (e.g. 'LL', 'LR', 'RL', 'RR')
        representing the sequence of turns. If either input does not match one of the
        expected values, the function returns None (no route determined).
    Behavior / Mapping
    ------------------
    The mapping implemented is as follows (start_arm -> goal_location : route):
    - North -> Northeast : 'LL'
    - North -> Southeast : 'LR'
    - North -> Southwest : 'RL'
    - North -> Northwest : 'RR'
    - East  -> Northeast : 'RR'
    - East  -> Southeast : 'LL'
    - East  -> Southwest : 'LR'
    - East  -> Northwest : 'RL'
    - South -> Northeast : 'RL'
    - South -> Southeast : 'RR'
    - South -> Southwest : 'LL'
    - South -> Northwest : 'LR'
    - West  -> Northeast : 'LR'
    - West  -> Southeast : 'RL'
    - West  -> Southwest : 'RR'
    - West  -> Northwest : 'LL'
    Examples
    --------
    >>> correct_route('North', 'Northeast')
    'LL'
    >>> correct_route('East', 'Southwest')
    'LR'
    Notes / Comments
    ----------------
    - Input strings are compared exactly; this function is case-sensitive. Normalize
      inputs (e.g., using .title() or .capitalize()) before calling if needed.
    - The function returns None for invalid or unexpected inputs; callers may prefer
      to validate inputs and raise a ValueError instead if stricter behavior is desired.
    - The naming convention for the return encodes simple left/right decisions; ensure
      that callers interpret the order of characters consistently (first = first turn).
    """
    
    if start_arm == 'North':
        if goal_location == 'Northeast':
            return 'LL'
        if goal_location == 'Southeast':
            return 'LR'
        if goal_location == 'Southwest':
            return 'RL'
        if goal_location == 'Northwest':
            return 'RR'
    elif start_arm == 'East':
        if goal_location == 'Northeast':
            return 'RR'
        if goal_location == 'Southeast':
            return 'LL'
        if goal_location == 'Southwest':
            return 'LR'
        if goal_location == 'Northwest':
            return 'RL'
    elif start_arm == 'South':
        if goal_location == 'Northeast':
            return 'RL'
        if goal_location == 'Southeast':
            return 'RR'
        if goal_location == 'Southwest':
            return 'LL'
        if goal_location == 'Northwest':
            return 'LR'
    elif start_arm == 'West':
        if goal_location == 'Northeast':
            return 'LR'
        if goal_location == 'Southeast':
            return 'RL'
        if goal_location == 'Southwest':
            return 'RR'
        if goal_location == 'Northwest':
            return 'LL'

# TODO: Drop this function if not used anywhere by the ende of the project
def egocentric_cue_direction(start_arm, goal_location, cue_orientation):
    """
    Describe the cue's egocentric position relative to the rat's start arm and destination.
    This helper collapses absolute maze coordinates (start arm, goal corner, cue orientation)
    into intuitive descriptors (forward, reverse, left, right) for use in behavioral metrics.

    Parameters
    ----------
    start_arm : str
        Arm from which the subject starts the trial ('North', 'East', 'South', 'West').
    goal_location : str
        Corner goal reached on the trial ('Northeast', 'Southeast', 'Southwest', 'Northwest').
    cue_orientation : str
        Cardinal cue placement for that subject/session.

    Returns
    -------
    str or None
        'forward', 'reverse', 'left', or 'right' depending on the cue's egocentric position.
        Returns None when the inputs do not match the supported cardinal labels.

    Behavior / Notes
    ----------------
    - When cue_orientation equals start_arm the cue sits behind the rat (`reverse`).
    - Contralateral cues produce a lateral label determined by the goal corner visited.
    - Extend or refactor this logic if diagonal start positions are introduced.

    Examples
    --------
    >>> egocentric_cue_direction('North', 'Northeast', 'South')
    'forward'
    >>> egocentric_cue_direction('East', 'Southwest', 'North')
    'left'
    """
    if start_arm == cue_orientation:
        return 'reverse'
    if ((start_arm == 'North' and cue_orientation == 'South') or
            (start_arm == 'East' and cue_orientation == 'West') or
            (start_arm == 'South' and cue_orientation == 'North') or
            (start_arm == 'West' and cue_orientation == 'East')):
        return 'forward'
    else:
        if start_arm == 'North':
            if goal_location == 'Northeast':
                return 'left'
            if goal_location == 'Southeast':
                return 'left'
            if goal_location == 'Southwest':
                return 'right'
            if goal_location == 'Northwest':
                return 'right'
        elif start_arm == 'East':
            if goal_location == 'Northeast':
                return 'right'
            if goal_location == 'Southeast':
                return 'left'
            if goal_location == 'Southwest':
                return 'left'
            if goal_location == 'Northwest':
                return 'right'
        elif start_arm == 'South':
            if goal_location == 'Northeast':
                return 'right'
            if goal_location == 'Southeast':
                return 'right'
            if goal_location == 'Southwest':
                return 'left'
            if goal_location == 'Northwest':
                return 'left'
        elif start_arm == 'West':
            if goal_location == 'Northeast':
                return 'left'
            if goal_location == 'Southeast':
                return 'right'
            if goal_location == 'Southwest':
                return 'right'
            if goal_location == 'Northwest':
                return 'left'


def actual_route(turn_1, turn_2):
    """
    Combine two route segments using the + operator and return the result.
    This is represents the first two turns made by the subject during a trial.
    Parameters
    ----------
    turn_1 : any
        The first route segment. Expected to be a value that supports the binary
        + operator with turn_2 (for example: numbers, lists/tuples of steps, or strings).
    turn_2 : any
        The second route segment. Should be the same type as or compatible with turn_1
        for the + operation to succeed.
    Returns
    -------
    any
        The result of turn_1 + turn_2, representing the combined route segment.
    Raises
    ------
    TypeError
        If the operands are not compatible for addition.
    Notes
    -----
    - This function performs a direct addition/concatenation and does not perform
      semantic validation of route data (e.g., direction consistency or path validity).
    - If you work with structured route objects, convert them to a common, additive
      representation (such as lists of steps) before calling this function.
    Examples
    --------
    >>> actual_route([0, 1], [2, 3])
    [0, 1, 2, 3]
    >>> actual_route(5, 10)
    15
    >>> actual_route('L', 'R')
    'LR'
    """  
    return turn_1 + turn_2


def direct_route(route_correct, route_actual, errors):
    """
    Score whether a trial reached the correct goal directly, indirectly, or via an error.

    Parameters
    ----------
    route_correct : str
        Two-character ideal route label for the trial ('LL', 'LR', 'RL', 'RR').
    route_actual : str
        Two-character label describing the subject's first two turns on the trial.
    errors : int
        Count of incorrect corner entries; any non-zero value marks the trial as an error.

    Returns
    -------
    int
        1: Direct, error-free route that matches the ideal path
        0: for an indirect but error-free route (did not enter an unrewarded well-arm)
       -1: when any errors occurred (entered unrewarded well-arm).

    Behavior / Notes
    ----------------
    - Treats any non-zero error count as a failure regardless of routing; adjust if partial
      credit should be awarded later.
    - Keep the integer codes aligned with downstream visualization/color maps.

    Examples
    --------
    >>> direct_route('LL', 'LL', 0)
    1
    >>> direct_route('LL', 'LR', 0)
    0
    >>> direct_route('LL', 'LL', 1)
    -1
    """
    if errors > 0:
        return -1
    elif route_actual == route_correct:
        return 1
    else:
        return 0


def first_response_perserveration(route_correct, route_actual, session_type):
    """
    Detect perseverative first-turn errors during reversal-style sessions.
    The metric flags when the initial turn during a reversal repeats the old strategy that
    used to be correct, signaling cognitive inflexibility.

    Parameters
    ----------
    route_correct : str
        Ideal two-turn route for the current trial.
    route_actual : str
        Observed two-turn route taken by the rat.
    session_type : str or None
        Label describing the session protocol (e.g., 'Dark Reverse').

    Returns
    -------
    int
        1: First turn matches the perseverative pattern
        0: when first turn does not match the perseverative pattern
       -1: Sessions that are not reversals (not evaluated).

    Behavior / Notes
    ----------------
    - Only specific reversal session types are eligible; others default to -1 to signal N/A.
    - The mapping compares specific LL/LR/RL/RR transitions documented in legacy analyses.

    Examples
    --------
    >>> first_response_perserveration('LL', 'RL', 'Dark Reverse')
    1
    >>> first_response_perserveration('LL', 'LL', 'Fixed Cue 1')
    -1
    """
    reversal_session_types = ['Dark Reverse', 'Rotate Reverse', 'Fixed Cue Switch', 
                              'Fixed Cue Swithch Twist']
    if session_type is None or session_type not in reversal_session_types:
        return -1
    elif route_correct == 'LL' and route_actual == 'RL':
        return 1
    elif route_correct == 'RR' and route_actual == 'LR':
        return 1
    elif route_correct == 'RL' and route_actual == 'LL':
        return 1
    elif route_correct == 'LR' and route_actual == 'RR':
        return 1
    else:
        return 0


def set_trial_type(start_arm, cue_orientation):
    """
    Derive a categorical trial type (standard, novel_route, reversal) from start and cue arms.
    This may seem vauge, since there are subtypes within each category, but this helper serves
    to seperate acquisition trials from probe trials for reversal and novel route analyses.

    Parameters
    ----------
    start_arm : str
        Entry arm for the trial ('North', 'East', 'South', 'West').
    cue_orientation : str
        Cue placement relative to the maze ('North', 'East', 'South', 'West').

    Returns
    -------
    str
        'reversal', 'novel_route', or 'standard' depending on the cue relative to the start arm.

    Behavior / Notes
    ----------------
    - Matching cue/start implies a reversal; opposite arms denote a novel route; otherwise it's
      treated as a standard training trial.
    - Update this helper before introducing additional trial categories.

    Examples
    --------
    >>> set_trial_type('North', 'North')
    'reversal'
    >>> set_trial_type('North', 'South')
    'novel_route'
    """
    if start_arm == cue_orientation:
        return 'reversal'
    elif ((start_arm == 'North' and cue_orientation == 'South') or
            (start_arm == 'East' and cue_orientation == 'West') or
            (start_arm == 'South' and cue_orientation == 'North') or
            (start_arm == 'West' and cue_orientation == 'East')):
        return 'novel_route'
    else:
        return 'standard'


def set_session_type_and_subtype(row):
    """
    Assign protocol-level labels based on the unique session types present for a subject.
    Rows aggregate all session_type values observed during a protocol; this helper maps that set
    to friendly names used in figures and tables (e.g., 'PI+VC_f2'). There are three main types of 
    aquisition session PI+VC (Path Integration + Visual Cue), VC (Visual Cue) and
    PI (Path Integration) - see paper for details. Frame refers to the room frame of so when it is 
    fixed or stable that means the cue is always in the same spot relative to the room. When it is 
    rotating the cue moves around the maze relative to the room. 

    Parameters
    ----------
    row : pandas.Series
        A row with a 'unique_session_types' field containing the set/list of session types.

    Returns
    -------
    tuple
        (session_type, session_subtype) where subtype may be None if the protocol lacks a
        finer-grained label.
        session_type:
            PI+VC_f2: Corresponds to aquisition with visual cues protocol on a fixed frame where
                      the second turn choice is always the same for a given start arm.
            PI+VC_f1: Corresponds to aquisition with visual cues protocol on a fixed frame where
                      the first turn choice is always the same for a given start arm.
                  VC: Corresponds to acquisition with visual cues protocol on a rotating frame.
                  NC: Corresponds to acquisition with no visual cues protocol on a fixed frame.
        session_subtype:
    Behavior / Notes
    ----------------
    - Uses exact set equality; ensure upstream code populates the set deterministically.
    - Returns (None, None) when the observed combination does not match a known protocol.

    Examples
    --------
    >>> row = pd.Series({'unique_session_types': {'Fixed Cue 1','Fixed Cue 2a','Fixed No Cue',
                         'Fixed Cue Rotate','Fixed Cue Switch'}})
    >>> set_session_type_and_subtype(row)
    ('PI+VC_f2', None)
    """

    session_type_set = set(row['unique_session_types'])
    fix_protocol = set(['Fixed Cue 1', 'Fixed Cue 2a', 'Fixed No Cue', 'Fixed Cue Rotate', 
                        'Fixed Cue Switch'])
    fix_ptotocol_twist = set(['Fixed Cue 1 Twist', 
                              'Fixed Cue Novel Route Twist', 'Fixed No Cue Twist',
                              'Fixed Cue Rotate Twist', 'Fixed Cue Switch Twist'])
    rot_protocol_fix_novelroute_rot_reversal = set(['Rotate Train', 'Rotate Detour', 
                                            'Rotate Reverse']) # Rotate Detour is a fixed novel route session, bad name I know.
    rot_protocol_rot_novelroute_fix_reversal = set(['Rotate Train', 'Rotate Detour Moving',
                                            'Fixed Cue Switch'])
    nc_protocol_cued_novelroute = set(['Dark Train', 'Dark Detour', 'Dark Reverse'])
    nc_protocol_uncued_novelroute = set(['Dark Train', 'Dark Detour No Cue', 'Dark Reverse'])

    if session_type_set == fix_protocol:
        return 'PI+VC_f2', None
    elif session_type_set == fix_ptotocol_twist:
        return 'PI+VC_f1', None
    elif session_type_set == rot_protocol_fix_novelroute_rot_reversal:
        return 'VC', 'PI+VC_VC'
    elif session_type_set == rot_protocol_rot_novelroute_fix_reversal:
        return 'VC', 'VC_PI+VC'
    elif session_type_set == nc_protocol_cued_novelroute:
        return 'PI', 'PI+VC_PI'
    elif session_type_set == nc_protocol_uncued_novelroute:
        return 'PI', 'PI_PI+VC'
    else:
        return None, None


def set_sessions_to_acquisition(row):
    """
    Count how many sessions belong to the acquisition phase for a subject.
    Upstream groupby objects pass each subject's acquisition subset to this helper; returning
    the length keeps downstream columns explicit about the number of qualifying sessions.

    Parameters
    ----------
    row : pandas.Series or pandas.DataFrame
        Collection of sessions filtered to acquisition-only rows.

    Returns
    -------
    int
        The number of rows/sessions provided in `row`.

    Notes / Comments
    ----------------
    - Accepts any iterable with `len` defined; no further validation occurs here.
    """
    return len(row)


# Gives score for novel route trials in a novel route session
def novel_route_score(x):
    """
    Count perfect novel-route trials after the warmup period within a session.
    The function assumes the first 16 trials belong to reacclimation and should be excluded
    before scoring novel-route performance.

    Parameters
    ----------
    x : pandas.DataFrame
        Session-level trial data containing 'trial_type' and 'errors' columns.

    Returns
    -------
    int
        Number of novel_route trials (after index 16) with zero errors.

    Behavior / Notes
    ----------------
    - Uses zero-based indexing, so `x.iloc[16:]` begins with the 17th trial.
    - Modify the slice boundaries if protocol definitions change.
    """
    post_16_trials = x.iloc[16:]
    novel_trials = post_16_trials[post_16_trials['trial_type'] == 'novel_route']
    zero_error_count = (novel_trials['errors'] == 0).sum()
    return zero_error_count


# Gives score for novel route trials in a novel route session
def standard_score_during_novel_route(x):
    """
    Quantify standard-trial performance during the novel-route phase of a session.
    Similar to `novel_route_score`, this focuses on the portion of a session after the initial
    16 trials but filters for 'standard' trial_type records instead.

    Parameters
    ----------
    x : pandas.DataFrame
        Session-level trial data containing 'trial_type' and 'errors' columns.

    Returns
    -------
    int
        Count of standard trials (post-index-16) completed with zero errors.

    Notes / Comments
    ----------------
    - Helps compare carry-over performance across standard and novel trial types.
    """
    post_16_trials = x.iloc[16:]
    novel_trials = post_16_trials[post_16_trials['trial_type'] == 'standard']
    zero_error_count = (novel_trials['errors'] == 0).sum()
    return zero_error_count

# Get number of reversal trials correct in each blokc of 8 trials and return as a list
# includes first two blocks which are pre-probe trials on the original training contingency
def reversal_block_scores(trial_scores):
    """
    Aggregate reversal performance into consecutive blocks of eight trials.
    Converts a trial-by-trial score vector into block-level correctness counts, while capping
    later blocks based on performance stability (>=7 correct the block before).

    Parameters
    ----------
    trial_scores : iterable
        Sequence where 0 marks a correct trial and non-zero marks an error.

    Returns
    -------
    list
        Ten-element list, each entry containing the number of correct trials per block.

    Behavior / Notes
    ----------------
    - The first two blocks correspond to pre-probe training and are included unmodified.
    - Once a block achieves >=7 correct trials (after block index 2), subsequent blocks are
      forced to the maximum (8) to reflect asymptotic performance.
    """
    trial_scores = list(trial_scores)
    block_scores = []
    max_correct_trials = 8
    number_of_blocks = 10
    for i in range(number_of_blocks):
        block = trial_scores[i*8:(i+1)*8]
        correct_trials = block.count(0)
        if i > 2 and block_scores[-1] >= 7:
            block_scores.append(max_correct_trials)
        else:
            block_scores.append(correct_trials)
    return block_scores

def perseverative_errors_in_reversal(goal_locations, goal_locations_visited):
    """
    Count perseverative corner entries during reversal trials.
    The metric tallies visits to the previously rewarded corner after the protocol flips the
    correct location, focusing on probes after the 16th trial.

    Parameters
    ----------
    goal_locations : iterable
        Sequence of correct goal identifiers for each trial.
    goal_locations_visited : iterable
        Sequence of lists describing which goal zones were entered per trial.

    Returns
    -------
    int
        Total number of perseverative corner entries observed during reversal trials.

    Behavior / Notes
    ----------------
    - Internally remaps textual locations to numeric maze zone IDs for direct comparison.
    - Only trials >= index 16 contribute, matching the novel-route/reversal phase.
    """
    goal_locations = list(goal_locations)
    goal_locations_visited = list(goal_locations_visited)
    goal_locations_size = len(goal_locations)
    goal_location_mapping = {'Northeast': 21, 'Southeast': 17, 'Southwest': 1, 'Northwest': 5}
    goal_locations = [goal_location_mapping[loc] for loc in goal_locations]
    perseverative_errors = 0
    for i in range(16, goal_locations_size):
        if goal_locations[i] == goal_location_mapping['Northeast']:
            #print(goal_location_mapping['Southwest'])
            #print(goal_locations_visited[i])
            perseverative_errors += goal_locations_visited[i].count(goal_location_mapping['Southwest'])
        elif goal_locations[i] == goal_location_mapping['Southeast']:
            perseverative_errors += goal_locations_visited[i].count(goal_location_mapping['Northwest'])
        elif goal_locations[i] == goal_location_mapping['Southwest']:
            perseverative_errors += goal_locations_visited[i].count(goal_location_mapping['Northeast'])
        elif goal_locations[i] == goal_location_mapping['Northwest']:
            perseverative_errors += goal_locations_visited[i].count(goal_location_mapping['Southeast'])
    #print("Perseverative Errors:", perseverative_errors)
    return perseverative_errors

# How many trials it takes to get to 7/8 correct in first 16 trials
def std_trials_crit(trial_scores):
    """
    Determine how many standard trials elapsed before reaching 7/8 correct.
    Slides an eight-trial window across the trial_scores vector and reports the trial count
    when criterion is first met during the first 80 trials.

    Parameters
    ----------
    trial_scores : iterable
        Per-trial error codes where 0 denotes a correct trial.

    Returns
    -------
    int or None
        Trial index (1-based) when criterion was met, or None if it never occurred.

    Behavior / Notes
    ----------------
    - Returns `trials_to_crit + 1` when the 8th trial in the window was an error despite 7/8 correct.
    - Stops after evaluating ten blocks; adjust `number_of_blokcs` to extend the search.
    """
    trial_scores = list(trial_scores)
    trials_to_crit = 7
    number_of_blokcs = 10
    for i in range(number_of_blokcs):
        perfect_trial_count = trial_scores[i:i+8].count(0)
        if (perfect_trial_count == 7 and trial_scores[i+7] > 0) or perfect_trial_count == 8:
            return trials_to_crit
        elif perfect_trial_count == 7:
            return trials_to_crit + 1
        trials_to_crit += 1
    return None

def reversal_trials_crit(trial_scores):
    """
    Compute the trials-to-criterion for reversal performance.
    Similar to `std_trials_crit` but restricted to the reversal portion of the session
    (starting at trial index 16) and capped at 64 trials by default.

    Parameters
    ----------
    trial_scores : iterable
        Sequence of trial results where 0 equals correct.

    Returns
    -------
    int
        Trial count needed to reach criterion, or 64 if criterion was not met earlier.

    Behavior / Notes
    ----------------
    - Uses a moving window of eight trials to evaluate the 7/8 rule.
    - Always returns an integer (default 64) to simplify downstream plotting.
    """
    trial_scores = list(trial_scores)
    trials_to_crit = 7
    for i in range(16, 81):
        perfect_trial_count = trial_scores[i:i+8].count(0)
        if (perfect_trial_count == 7 and trial_scores[i+7] > 0) or perfect_trial_count == 8:
            return trials_to_crit
        elif perfect_trial_count == 7:
            return trials_to_crit + 1
        trials_to_crit += 1
    return 64



    max_len = max([len(ln) for ln in var_list])
    insert_len_list = [[np.nan] * (max_len - len(vl)) for vl in var_list]
    [insert_len_list[i].extend(var_list[i]) for i in range(len(var_list))]
    return insert_len_list

def modified_z_score(arr):
    """
    Calculate the modified z-score using the median absolute deviation (MAD).
    This robust alternative to the classic z-score resists outliers and is preferred
    for skewed distributions common in behavioral metrics.

    Parameters
    ----------
    arr : array-like
        Numeric vector (NumPy array, list, or pandas Series) to standardize.

    Returns
    -------
    numpy.ndarray
        Modified z-scores for each element in `arr`.

    Notes / Comments
    ----------------
    - Uses the constant 0.6745 to approximate standard deviations from MAD.
    - Callers should ensure `mad` is non-zero; otherwise infinities may occur.
    """
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return 0.6745 * (arr - median) / mad 

def string_to_list(x):
    """
    Convert various representations of list-like data into a true Python list.
    Handles already-materialized lists, None, empty strings, serialized Python literals,
    and fallbacks by wrapping the value in a single-element list.

    Parameters
    ----------
    x : any
        Value to normalize (list, string, None, etc.).

    Returns
    -------
    list
        Parsed list representation, or `[x]` when no conversion rule matches.

    Notes / Comments
    ----------------
    - Uses `ast.literal_eval` for safety when parsing strings.
    - Empty strings and None both map to an empty list for consistency.
    """
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return val
            else:
                return [val]
        except Exception:
            # string that isn't a valid literal
            return [s]
    return [x]

# connect to sqlite database
con = sqlite3.connect('data/processed/MazeControl-clean.db')

# list subject ids to be used for conrner maze analysis papoer
subject_id = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 71, 
              73, 75, 76, 78, 80, 82, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 97, 98, 
              99, 100, 106, 107, 108, 109, 110, 112, 113, 117, 119, 120, 122, 123, 124, 126,
              127, 129, 130]

# convert subject_id to tuple for use in sqlite querry
subject_id_tuple = tuple(subject_id)

# get subject data from db and create pandas dataframe
subject_df = pd.read_sql(f'''
                        SELECT
                            subject_id,
                            name,
                            sex,
                            cue_goal_orientation
                        FROM subjects
                        WHERE subject_id IN ({','.join('?' for _ in subject_id_tuple)})
                        ORDER BY subject_id ASC;
                        ''', con, params = subject_id_tuple)

# Retrieve subject data with relevant fields
trial_data_df = pd.read_sql(f'''
                            SELECT
                                a.subject_id,
                                a.name,
                                a.sex,
                                a.cue_goal_orientation,
                                a.date_of_birth,
                                b.session_id,
                                b.session_number,
                                b.session_type,
                                b.total_perfect,
                                b.total_trials,
                                b.total_errors,
                                b.score,
                                b.session_duration,
                                c.trial_number, 
                                c.start_arm,
                                c.goal_location,
                                c.cue_orientation,
                                c.time_duration,
                                c.errors,
                                c.turn_1,
                                c.turn_2,
                                c.goal_zones_visited
                            FROM subjects AS a
                            JOIN session AS b ON a.subject_id = b.subject_id
                            JOIN trial AS c ON b.session_id = c.session_id
                            WHERE
                                a.subject_id IN ({','.join('?' * len(subject_id_tuple))})
                                AND b.session_number !='1e'
                                AND b.session_number !='2e'
                            ORDER BY
                                a.subject_id ASC;
                            ''', con, params = subject_id_tuple)

# Convert string representation of lists to actual lists
trial_data_df['goal_zones_visited'] = trial_data_df['goal_zones_visited'].apply(string_to_list)
#print(trial_data_df[['name','goal_zones_visited']].head())
# Column Description
# route_correct: This gives what should be the correct for the trial as RR, LL, RL, LR
# route_actual: Gives the actual route taken in terms of the first two choices made. It is possible
#   that the second turn occurs at the unexpected intersection if the rat back tracks to the opposite
#   side of the maze after making the firts choice. This could be determined by looking at what corner
#   the rat entered after the making the second turn - although it is possible the rat could back track
#   but that is extremely unlikely.
# direct_route: this gives a 1 if the subject takes a direct route to the correct goal and a 0 if the
#   subject took an indirect route to the goal without entering a incorrect corner, and a -1 if they 
#   entered an incorrect corner.
# add columns to trial data
# trial_type: tells if it's a standard trained trial, novel route trial, or reversal trial
trial_data_df['route_correct'] = pd.Series(dtype=str)
trial_data_df['route_actual'] = pd.Series(dtype=str)
trial_data_df['direct_route'] = pd.Series(dtype=int)
trial_data_df['trial_type'] = pd.Series(dtype=str)

# supply values to added columns
trial_data_df['route_correct'] = trial_data_df.apply(
    lambda row: correct_route(row['start_arm'], row['goal_location']), axis=1)
trial_data_df['route_actual'] = trial_data_df.apply(
    lambda row: actual_route(row['turn_1'], row['turn_2']), axis=1)
trial_data_df['direct_route'] = trial_data_df.apply(
    lambda row: direct_route(row['route_correct'], row['route_actual'], row['errors']), axis=1)
trial_data_df['trial_type'] = trial_data_df.apply(
    lambda row: set_trial_type(row['start_arm'], row['cue_orientation']), axis=1)

# Create data frame for sessions and add columns
session_data_df = pd.read_sql(f'''
                            SELECT
                                a.subject_id,
                                a.name,
                                a.sex,
                                a.cue_goal_orientation,
                                a.date_of_birth,
                                b.cue_conf,
                                b.session_id,
                                b.session_number,
                                b.session_type,
                                b.total_perfect,
                                b.total_trials,
                                b.total_errors,
                                b.score,
                                b.session_duration
                            FROM
                                subjects AS a
                            JOIN
                                session AS b
                            ON
                                a.subject_id = b.subject_id
                            WHERE
                                a.subject_id IN ({','.join('?' * len(subject_id_tuple))})
                                AND b.session_number !='1e'
                                AND b.session_number !='2e'
                            ORDER BY
                                a.subject_id ASC;
                            ''', con, params = subject_id_tuple)

# Create a data frame to store results of sessions for each rat
# rscr: raw score only a count of trials with no errors during a session
# acq: acquisition, session used to train two start two choice.
# tstc: two start two choice
tstc_results_df = pd.DataFrame({'subject_id' : pd.Series(dtype=int),
                                'name' : pd.Series(dtype=str),
                                'sex' : pd.Series(dtype=str),
                                'cue_goal_orientation' : pd.Series(dtype=str),
                                'training_type' : pd.Series(dtype=str),
                                'training_subtype' : pd.Series(dtype=str),
                                'cue_approach' : pd.Series(dtype=str),
                                'acquisition_scrs' : pd.Series(dtype=object), # Numpy array of raw error scores from each training session
                                'trials_to_acq' : pd.Series(dtype=int),
                                'novel_route_probe_rscr' : pd.Series(dtype=int), # raw score of novel route trials
                                'novel_route_probe_scr' : pd.Series(dtype=int), # percent score of novel route trials
                                'novel_route_probe_fix_modz_scr' : pd.Series(dtype=int), # modfied-z normalization of percent score for FIX group
                                'novel_route_probe_fix_minmax_scr' : pd.Series(dtype=int),
                                'novel_route_probe_fix_z_scr' : pd.Series(dtype=int),
                                'novel_route_preprobe_std_rscr' : pd.Series(dtype=int), # raw score on trained
                                'novel_route_preprobe_std_scr' : pd.Series(dtype=int), # percent score on trained
                                'novel_route_probe_std_rscr' : pd.Series(dtype=int), # raw score on trained trials after first 16 trials
                                'novel_route_probe_std_scr' : pd.Series(dtype=int), # percent score on trained trials after first 16 trials
                                'no_cue_rscr' : pd.Series(dtype=int), # raw score is number of trials without errors
                                'no_cue_scr' : pd.Series(dtype=int), # percent score raw score over total trials
                                'no_cue_fix_modz_scr' : pd.Series(dtype=int), # modified z-score of percent score
                                'no_cue_fix_minmax_scr' : pd.Series(dtype=int), # min max of percent score
                                'no_cue_fix_z_scr' : pd.Series(dtype=int),
                                'rotation_rscr' : pd.Series(dtype=int), # raw score is number of trials without errors
                                'rotation_scr' : pd.Series(dtype=int), # percent score raw score over total trials
                                'rotation_fix_modz_scr' : pd.Series(dtype=int), # mondified z normalization of percent score
                                'rotation_fix_minmax_scr' : pd.Series(dtype=int), # mimax normalize of percent score
                                'rotation_fix_z_scr' : pd.Series(dtype=int),
                                'reversal_block_scores' : pd.Series(dtype=object), # numpy array of correct trials for each block of 8 trials during reversal probe
                                'reversal_block_percent_scores' : pd.Series(dtype=object), # numpy array of percent correct trials for each block of 8 trials during reversal probe
                                'reversal_preprobe_rscr' : pd.Series(dtype=int), # rscr is number of trials to 7/8 correct here before reversal
                                'reversal_rscr' : pd.Series(dtype=int), # rscr is number of trials to 7/8 reversal trials correct
                                'reversal_scr' : pd.Series(dtype=int), # Percentage based score of trials to criterion as totla number of window steps minus number trial to cirterion all divied by total number of window steps.
                                'reversal_fix_modz_scr' : pd.Series(dtype=int), # modified z-score normalization of the percent score
                                'reversal_fix_minmax_scr' : pd.Series(dtype=int),
                                'reversal_fix_z_scr' : pd.Series(dtype=int),
                                'reversal_perseverative_err_scr' : pd.Series(dtype=int), # Counted number of times rat took first route to the old goal
                                'reversal_exploratory_err_scr' : pd.Series(dtype=int), # Counted number of times rat entered incorrect goal
                                'pca_modz_data_points' : pd.Series(dtype=object), # 2D cordinate obtained from PCA on probe trial modz-normalized scores
                                'pca_minmax_data_points' : pd.Series(dtype=object), # 2D cordinate obtained from PCA on probe trial minmax-normalized scores
                                'pca_z_data_points' : pd.Series(dtype=object),
                                'pca_modz_kmeans_cluster_label' : pd.Series(dtype=int), # label obtained from running k-means cluster analysis on PCA data
                                'pca_minmax_kmeans_cluster_label' : pd.Series(dtype=int), # label obtained from running k-means cluster analysis on PCA data
                                'pca_z_kmeans_cluster_label' : pd.Series(dtype=int),
                                'kmeans_minmax_cluster_cords_nr_nc_rt_rv' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_labels_nr_nc_rt_rv' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_cords_nr_nc_rt' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_labels_nr_nc_rt' : pd.Series(dtype=int),
                                'kmeans_minmax_cluster_cords_nr_nc_rv' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_labels_nr_nc_rv' : pd.Series(dtype=int),
                                'kmeans_minmax_cluster_cords_nr_rv_rt' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_labels_nr_rv_rt' : pd.Series(dtype=int),
                                'kmeans_minmax_cluster_cords_nc_rv_rt' : pd.Series(dtype=object),
                                'kmeans_minmax_cluster_labels_nc_rv_rt' : pd.Series(dtype=int)
                                })

# ADD subject_id, name, sex and cue_goal_orientation TO tstc_results_df DATAFRAME
#   names are added by using concat to add subject_df name data into two_start_two_choice_results
tstc_results_df = pd.concat([tstc_results_df, subject_df], ignore_index=True)

# ADD training_type, training_subtype TO tstc_results_df DATAFRAME
unique_session_types = session_data_df.groupby('subject_id')['session_type'].apply(lambda x:
    list(set(x))).reset_index()
unique_session_types.rename(columns={'session_type': 'unique_session_types'}, inplace=True)
tstc_results_df = tstc_results_df.merge(
    unique_session_types, on='subject_id', how='left')
tstc_results_df[['training_type', 'training_subtype']] = tstc_results_df.apply(
    lambda row: pd.Series(set_session_type_and_subtype(row)),axis=1)

# ADD cue_approach TO tstc_results_df DATAFRAME
tstc_results_df['cue_approach'] = tstc_results_df.apply(
    lambda row: approach_to_goal_from_cue(row['cue_goal_orientation']), axis=1)

# ADD acquisition_scrs TO tstc_results_df DATAFRAME
session_filter = ['Fixed Cue 1', 'Fixed Cue 1 Twist', 'Rotate Train', 'Dark Train']
filtered_session_data_df = session_data_df[session_data_df['session_type'].isin(session_filter)]
acquisition_scrs_df = filtered_session_data_df.groupby(['subject_id', 'session_type'])['score'].apply(
    lambda x: [float(i) for i in x]).reset_index()
tstc_results_df['acquisition_scrs'] = tstc_results_df['acquisition_scrs'].combine_first(
    tstc_results_df['subject_id'].map(acquisition_scrs_df.set_index('subject_id')['score']))

# ADD trials_to_acquisition TO tstc_results_df DATAFRAME
tstc_results_df['trials_to_acq'] = tstc_results_df['acquisition_scrs'].apply(
    lambda x: len(x))

# ADD novel_route_preprobe_std_rscr, novel_route_preprobe_std_scr TO tstc_results_df DATAFRAME
num_novel_route_preprobe_trials = 16
novel_route_session_filter = ['Fixed Cue 2a', 'Dark Detour', 'Dark Detour No Cue', 'Rotate Detour',
                              'Rotate Detour Moving', 'Fixed Cue Novel Route Twist'] 
filtered_novel_trial_data_df = trial_data_df[trial_data_df['session_type'].isin(
    novel_route_session_filter)] # Use this filtered dataframe for all novel route scores
novel_route_preprobe_std_scr_df = (
    filtered_novel_trial_data_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: (x.head(16)['errors'] == 0).sum(), include_groups=False)
    .reset_index(name='novel_route_preprobe_std_rscr')
)
novel_route_preprobe_std_scr_df = pd.merge(
    novel_route_preprobe_std_scr_df,
    filtered_novel_trial_data_df[['session_id', 'subject_id']].drop_duplicates(),
    on='session_id'
)
novel_route_preprobe_std_scr_df = novel_route_preprobe_std_scr_df[
    ['subject_id', 'novel_route_preprobe_std_rscr']
]
novel_route_preprobe_std_scr_df['novel_route_preprobe_std_scr'] = novel_route_preprobe_std_scr_df['novel_route_preprobe_std_rscr'].apply(
    lambda x: x/num_novel_route_preprobe_trials
)
tstc_results_df.set_index('subject_id', inplace=True)
novel_route_preprobe_std_scr_df.set_index('subject_id', inplace=True)
tstc_results_df.update(novel_route_preprobe_std_scr_df)
tstc_results_df.reset_index(inplace=True)

# Add novel_route_probe_rscr, novel_route_probe_scr, novel_route_probe_fix_minmax_scr, novel_route_probe_fit_modz_scr, 
# novel_route_probe_fix_z_scr, TO tstc_results_df
num_novel_route_trials = 16
novel_route_probe_scrs_df = (
    filtered_novel_trial_data_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: novel_route_score(x), include_groups=False)
    .reset_index(name='novel_route_probe_rscr')
)
novel_route_probe_scrs_df = pd.merge(
    novel_route_probe_scrs_df,
    filtered_novel_trial_data_df[['session_id', 'subject_id', 'session_type']].drop_duplicates(),
    on='session_id'
)
novel_route_probe_scrs_df = novel_route_probe_scrs_df[['subject_id', 'session_type', 'novel_route_probe_rscr']]
novel_route_probe_scrs_df['novel_route_probe_scr'] = novel_route_probe_scrs_df['novel_route_probe_rscr'].apply(lambda x: x/num_novel_route_trials)
novel_route_probe_fix_scrs_np = novel_route_probe_scrs_df[
    novel_route_probe_scrs_df['session_type'].isin(['Fixed Cue 2a', 'Fixed Cue Novel Route Twist'])]['novel_route_probe_scr'].to_numpy()

novel_route_fix_max_scr = np.max(novel_route_probe_fix_scrs_np)
novel_route_fix_min_scr = np.min(novel_route_probe_fix_scrs_np)
novel_route_probe_scrs_df['novel_route_probe_fix_minmax_scr'] = novel_route_probe_scrs_df[
    novel_route_probe_scrs_df['session_type'].isin(['Fixed Cue 2a', 'Fixed Cue Novel Route Twist'])]['novel_route_probe_scr'].apply(
    lambda x: (x - novel_route_fix_min_scr)/(novel_route_fix_max_scr - novel_route_fix_min_scr))
novel_route_probe_fix_median = np.median(novel_route_probe_fix_scrs_np)
novel_route_probe_fix_mad = np.median(np.abs(novel_route_probe_fix_scrs_np - novel_route_probe_fix_median))
novel_route_probe_scrs_df['novel_route_probe_fix_modz_scr'] = novel_route_probe_scrs_df[
    novel_route_probe_scrs_df['session_type'].isin(['Fixed Cue 2a', 'Fixed Cue Novel Route Twist'])]['novel_route_probe_scr'].apply(
    lambda x: 0.6745 * (x - novel_route_probe_fix_median) / novel_route_probe_fix_mad)
novel_route_fix_mean = np.mean(novel_route_probe_fix_scrs_np)
novel_route_fix_std = np.std(novel_route_probe_fix_scrs_np)
novel_route_probe_scrs_df['novel_route_probe_fix_z_scr'] = novel_route_probe_scrs_df[
    novel_route_probe_scrs_df['session_type'].isin(['Fixed Cue 2a', 'Fixed Cue Novel Route Twist'])]['novel_route_probe_scr'].apply(
    lambda x: (x - novel_route_fix_mean) / novel_route_fix_std)
tstc_results_df.set_index('subject_id', inplace=True)
novel_route_probe_scrs_df.set_index('subject_id', inplace=True)
tstc_results_df.update(novel_route_probe_scrs_df)
tstc_results_df.reset_index(inplace=True)

# ADD novel_route_probe_std_rscr, novel_route_probe_std_scr TO tstc_results_df
num_novel_route_probe_std_trials = 8
novel_route_probe_std_rscr_df = (
    filtered_novel_trial_data_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: novel_route_score(x), include_groups=False)
    .reset_index(name='novel_route_probe_std_rscr')
)
novel_route_probe_std_rscr_df = pd.merge(
    novel_route_probe_std_rscr_df,
    filtered_novel_trial_data_df[['session_id', 'subject_id']].drop_duplicates(),
    on='session_id'
)
novel_route_probe_std_rscr_df = novel_route_probe_std_rscr_df[
    ['subject_id', 'novel_route_probe_std_rscr']
]
novel_route_probe_std_rscr_df['novel_route_probe_std_scr'] = novel_route_probe_std_rscr_df['novel_route_probe_std_rscr'].apply(
    lambda x: x/num_novel_route_probe_std_trials
)
tstc_results_df.set_index('subject_id', inplace=True)
novel_route_probe_std_rscr_df.set_index('subject_id', inplace=True)
tstc_results_df.update(novel_route_probe_std_rscr_df)
tstc_results_df.reset_index(inplace=True)

#ADD no_cue_rscr, no_cue_scr, no_cue_fix_modz_scr, no_cue_fix_minmax_scr, no_cue_fix_z_scr TO two_start_two_choice_df
num_no_cue_trials = 32
no_cue_trial_filter_df = trial_data_df[
    trial_data_df['session_type'].isin(['Fixed No Cue', 'Fixed No Cue Twist'])]
no_cue_scrs_df = (
    no_cue_trial_filter_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: (x['errors'] == 0).sum(), include_groups=False)
    .reset_index(name='no_cue_rscr')
) # add no_cue raw scores
no_cue_scrs_df = pd.merge(
    no_cue_scrs_df,
    no_cue_trial_filter_df[['session_id', 'subject_id', 'session_type']].drop_duplicates(),
    on='session_id'
)
no_cue_scrs_df = no_cue_scrs_df[['subject_id', 'session_type', 'no_cue_rscr']]
no_cue_scrs_df['no_cue_scr'] = no_cue_scrs_df['no_cue_rscr'].apply(lambda x: x/num_no_cue_trials) # add percent no_cue percent scores

no_cue_scrs_np = no_cue_scrs_df[
    no_cue_scrs_df['session_type'].isin(['Fixed No Cue', 'Fixed No Cue Twist'])]['no_cue_scr']
no_cue_median = np.median(no_cue_scrs_np)
no_cue_mad = np.median(np.abs(no_cue_scrs_np - no_cue_median))
no_cue_scrs_df['no_cue_fix_modz_scr'] = no_cue_scrs_df[
    no_cue_scrs_df['session_type'].isin(['Fixed No Cue', 'Fixed No Cue Twist'])]['no_cue_scr'].apply(
    lambda x: 0.6745 * (x - no_cue_median) / no_cue_mad)
no_cue_fix_max_scr = np.max(no_cue_scrs_np)
no_cue_fix_min_scr = np.min(no_cue_scrs_np)
no_cue_scrs_df['no_cue_fix_minmax_scr'] = no_cue_scrs_df[
    no_cue_scrs_df['session_type'].isin(['Fixed No Cue', 'Fixed No Cue Twist'])]['no_cue_scr'].apply(
    lambda x: (x - no_cue_fix_min_scr)/(no_cue_fix_max_scr - no_cue_fix_min_scr))
no_cue_fix_mean = np.mean(no_cue_scrs_np)
no_cue_fix_std = np.std(no_cue_scrs_np)
no_cue_scrs_df['no_cue_fix_z_scr'] = no_cue_scrs_df[
    no_cue_scrs_df['session_type'].isin(['Fixed No Cue', 'Fixed No Cue Twist'])]['no_cue_scr'].apply(
    lambda x: (x - no_cue_fix_mean)/no_cue_fix_std)
tstc_results_df.set_index('subject_id', inplace=True)
no_cue_scrs_df.set_index('subject_id', inplace=True)
tstc_results_df.update(no_cue_scrs_df)
tstc_results_df.reset_index(inplace=True)

# ADD rotation_rscr, rotation_scr, rotation_fix_modz_scr, rotation_fix_z_scr TO tstc_results_df
num_rotation_trials = 16
rotation_trial_filter_df = trial_data_df[trial_data_df['session_type'].isin(['Fixed Cue Rotate',
                                                                             'Fixed Cue Rotate Twist'])]
rotation_scrs_df = (
    rotation_trial_filter_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: (x['errors'] == 0).sum(), include_groups=False)
    .reset_index(name='rotation_rscr')
) # add raw rotation scores
rotation_scrs_df = pd.merge(
    rotation_scrs_df,
    rotation_trial_filter_df[['session_id', 'subject_id', 'session_type']].drop_duplicates(),
    on='session_id'
)
rotation_scrs_df = rotation_scrs_df[
    ['subject_id', 'session_type', 'rotation_rscr']
]
rotation_scrs_df['rotation_scr'] = rotation_scrs_df['rotation_rscr'].apply(lambda x: x/num_rotation_trials) # add percent scores for rotation
rotation_scrs_np = rotation_scrs_df[
    rotation_scrs_df['session_type'].isin(['Fixed Cue Rotate', 'Fixed Cue Rotate Twist'])]['rotation_scr'].to_numpy()
rotation_scrs_median = np.median(rotation_scrs_np)
rotation_scrs_mad = np.median(np.abs(rotation_scrs_np - rotation_scrs_median))
rotation_scrs_df['rotation_fix_modz_scr'] = rotation_scrs_df[
    rotation_scrs_df['session_type'].isin(['Fixed Cue Rotate', 'Fixed Cue Rotate Twist'])]['rotation_scr'].apply(
        lambda x: 0.6745 * (x - rotation_scrs_median)/rotation_scrs_mad)
rotation_fix_max_scr = np.max(rotation_scrs_np)
rotation_fix_min_scr = np.min(rotation_scrs_np)
rotation_scrs_df['rotation_fix_minmax_scr'] = rotation_scrs_df[
    rotation_scrs_df['session_type'].isin(['Fixed Cue Rotate', 'Fixed Cue Rotate Twist'])]['rotation_scr'].apply(
    lambda x: (x - rotation_fix_min_scr)/(rotation_fix_max_scr - rotation_fix_min_scr))
rotation_fix_mean = np.mean(rotation_scrs_np)
rotation_fix_std = np.std(rotation_scrs_np)
rotation_scrs_df['rotation_fix_z_scr'] = rotation_scrs_df[
    rotation_scrs_df['session_type'].isin(['Fixed Cue Rotate', 'Fixed Cue Rotate Twist'])]['rotation_scr'].apply(
    lambda x: (x - rotation_fix_mean)/rotation_fix_std)
tstc_results_df.set_index('subject_id', inplace=True)
rotation_scrs_df.set_index('subject_id', inplace=True)
tstc_results_df.update(rotation_scrs_df)
tstc_results_df.reset_index(inplace=True)

# ADD reversal_block_scores TO tstc_results_df
reversal_trial_filter = ['Fixed Cue Switch', 'Fixed Cue Switch Twist', 'Dark Reverse', 'Rotate Reverse']
reversal_trials_df = trial_data_df[trial_data_df['session_type'].isin(reversal_trial_filter)]
reversal_block_scores_df = (
    reversal_trials_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: reversal_block_scores(x['errors']), include_groups=False)
    .reset_index(name='reversal_block_scores')
)
reversal_block_scores_df = pd.merge(
    reversal_block_scores_df,
    reversal_trials_df[['session_id', 'subject_id']].drop_duplicates(),
    on='session_id'
)
reversal_block_scores_df = reversal_block_scores_df[
    ['subject_id', 'reversal_block_scores']
]

tstc_results_df.set_index('subject_id', inplace=True)
reversal_block_scores_df.set_index('subject_id', inplace=True)
tstc_results_df.update(reversal_block_scores_df)
tstc_results_df.reset_index(inplace=True)

# Add reversal_perseverative_err_scr TO tstc_results_df
reversal_trial_filter = ['Fixed Cue Switch', 'Fixed Cue Switch Twist', 'Dark Reverse', 'Rotate Reverse']
reversal_trials_df = trial_data_df[trial_data_df['session_type'].isin(reversal_trial_filter)]
reversal_perseverative_scores_df = (
    reversal_trials_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: perseverative_errors_in_reversal(x['goal_location'],
                                                      x['goal_zones_visited']), include_groups=False)
    .reset_index(name='reversal_perseverative_err_scr')
)
reversal_perseverative_scores_df = pd.merge(
    reversal_perseverative_scores_df,
    reversal_trials_df[['session_id', 'subject_id']].drop_duplicates(),
    on='session_id'
)
reversal_perseverative_scores_df = reversal_perseverative_scores_df[
    ['subject_id', 'reversal_perseverative_err_scr']
]
tstc_results_df.set_index('subject_id', inplace=True)
reversal_perseverative_scores_df.set_index('subject_id', inplace=True)
tstc_results_df.update(reversal_perseverative_scores_df)
tstc_results_df.reset_index(inplace=True)

# ADD reversal_preprobe_rscr TO tstc_results_df
reversal_trial_filter = ['Fixed Cue Switch', 'Fixed Cue Switch Twist', 'Dark Reverse', 'Rotate Reverse']
reversal_trials_df = trial_data_df[trial_data_df['session_type'].isin(reversal_trial_filter)]
reversal_preprobe_rscr_df = (
    reversal_trials_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: std_trials_crit(x['errors']), include_groups=False)
    .reset_index(name='reversal_preprobe_rscr')
)
reversal_preprobe_rscr_df = pd.merge(
    reversal_preprobe_rscr_df,
    reversal_trials_df[['session_id', 'subject_id']].drop_duplicates(),
    on='session_id'
)
reversal_preprobe_rscr_df = reversal_preprobe_rscr_df[
    ['subject_id', 'reversal_preprobe_rscr']
]
tstc_results_df.set_index('subject_id', inplace=True)
reversal_preprobe_rscr_df.set_index('subject_id', inplace=True)
tstc_results_df.update(reversal_preprobe_rscr_df)
tstc_results_df.reset_index(inplace=True)

# ADD reversal_rscr, reversal_scr, reversal_fix_modz_scr, reversal_fix_z_scr TO tstc_results_df
tot_rev_trials = 64
trials_to_rev_crit = 7 # number of trials out 8 required to reach reversal criterion
num_trials_to_scr = tot_rev_trials - trials_to_rev_crit
reversal_trials_df = trial_data_df[trial_data_df['session_type'].isin(reversal_trial_filter)]
reversal_scrs_df = (
    reversal_trials_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: reversal_trials_crit(x['errors']), include_groups=False)
    .reset_index(name='reversal_rscr')
)
reversal_scrs_df = pd.merge(
    reversal_scrs_df,
    reversal_trials_df[['session_id', 'subject_id', 'session_type']].drop_duplicates(),
    on='session_id'
)
reversal_scrs_df = reversal_scrs_df[
    ['subject_id', 'session_type', 'reversal_rscr']
]
reversal_scrs_df['reversal_scr'] = reversal_scrs_df['reversal_rscr'].apply(lambda x: (num_trials_to_scr - (x - trials_to_rev_crit))/num_trials_to_scr)
reversal_scrs_np = reversal_scrs_df[
    reversal_scrs_df['session_type'].isin(['Fixed Cue Switch', 'Fixed Cue Switch Twist'])]['reversal_scr'].to_numpy()
reversal_median = np.median(reversal_scrs_np)
reversal_mad = np.median(np.abs(reversal_scrs_np - reversal_median))
reversal_scrs_df['reversal_fix_modz_scr'] = reversal_scrs_df[
    reversal_scrs_df['session_type'].isin(['Fixed Cue Switch', 'Fixed Cue Switch Twist'])]['reversal_scr'].apply(
    lambda x: 0.6745 * (x - reversal_median) / reversal_mad)
reversal_fix_max_scr = np.max(reversal_scrs_np)
reversal_fix_min_scr = np.min(reversal_scrs_np)
reversal_scrs_df['reversal_fix_minmax_scr'] = reversal_scrs_df[
    reversal_scrs_df['session_type'].isin(['Fixed Cue Switch', 'Fixed Cue Switch Twist'])]['reversal_scr'].apply(
    lambda x: (x - reversal_fix_min_scr) / (reversal_fix_max_scr - reversal_fix_min_scr))
reversal_scrs_mean = np.mean(reversal_scrs_np)
reversal_scrs_std = np.std(reversal_scrs_np)
reversal_scrs_df['reversal_fix_z_scr'] = reversal_scrs_df[
    reversal_scrs_df['session_type'].isin(['Fixed Cue Switch', 'Fixed Cue Switch Twist'])]['reversal_scr'].apply(
    lambda x: 0.6745 * (x - reversal_scrs_mean) / reversal_scrs_std)
tstc_results_df.set_index('subject_id', inplace=True)
reversal_scrs_df.set_index('subject_id', inplace=True)
tstc_results_df.update(reversal_scrs_df)
tstc_results_df.reset_index(inplace=True)

# Add reversal_block_percent_scores TO tstc_results_df
tstc_results_df['reversal_block_percent_scores'] = tstc_results_df['reversal_block_scores'].apply(
    lambda scores: [score / 8 for score in scores] if isinstance(scores, list) else np.nan
)

# ADD pca_modz_data_point TO tstc_result_df
# Perform PCA on FIX probe sessions on modified z-score values
filtered_tstc_modz_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                      'PI+VC_f1'])]
filtered_tstc_modz_results_df  = filtered_tstc_modz_results_df[['name', 
                                                                'novel_route_probe_fix_modz_scr', 
                                                                'no_cue_fix_modz_scr',
                                                                'rotation_fix_modz_scr',
                                                                'reversal_fix_modz_scr']]
pca_data_np = filtered_tstc_modz_results_df.iloc[:, 1:5].to_numpy()
#print(pca_data_np)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_data = pca.fit_transform(pca_data_np)
# Print the shape of the transformed data
#print("Transformed data shape:", pca_data.shape)
# Explained variance
#print("Explained variance ratio:", pca.explained_variance_ratio_)
pca_data_df = pd.DataFrame({'name' : filtered_tstc_modz_results_df.iloc[:, 0].to_list()})
pca_data_points_list = [np.array(row) for row in pca_data]
pca_data_df['pca_modz_data_points'] = pca_data_points_list
tstc_results_df.set_index('name', inplace=True)
pca_data_df.set_index('name', inplace=True)
tstc_results_df.update(pca_data_df)
tstc_results_df.reset_index(inplace=True)

#ADD pca_modz_kmeans_cluster_label TO tstc_result_df
#k_matrix = np.column_stack((pca_data[:,0], pca_data[:,1]))
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(pca_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : filtered_tstc_modz_results_df.iloc[:, 0].to_list()})
kmeans_data_df['pca_modz_kmeans_cluster_label'] = labels
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD pca_minmax_data_points TO tstc_result_df
# Perform PCA on FIX probe sessions on modified z-score values
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                      'PI+VC_f1'])]
filtered_tstc_minmax_results_df  = filtered_tstc_minmax_results_df[['name', 
                                                                    'novel_route_probe_fix_minmax_scr',
                                                                    'no_cue_fix_minmax_scr',
                                                                    'rotation_fix_minmax_scr', 
                                                                    'reversal_fix_minmax_scr']]
pca_data_np = filtered_tstc_minmax_results_df.iloc[:, 1:5].to_numpy()
pca = PCA(n_components=3)  # Reduce to 2 dimensions for visualization
pca_data = pca.fit_transform(pca_data_np)
# Print the shape of the transformed data
#print("Transformed data shape:", pca_data.shape)
# Explained variance
#print("Explained variance ratio:", pca.explained_variance_ratio_)
pca_data_df = pd.DataFrame({'name' : filtered_tstc_minmax_results_df.iloc[:, 0].to_list()})
pca_data_points_list = [np.array(row) for row in pca_data]
pca_data_df['pca_minmax_data_points'] = pca_data_points_list
tstc_results_df.set_index('name', inplace=True)
pca_data_df.set_index('name', inplace=True)
tstc_results_df.update(pca_data_df)
tstc_results_df.reset_index(inplace=True)

#ADD pca_minmax_kmeans_cluster_label TO tstc_result_df
#k_matrix = np.column_stack((pca_data[:,0], pca_data[:,1]))
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(pca_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : filtered_tstc_minmax_results_df.iloc[:, 0].to_list()})
kmeans_data_df['pca_minmax_kmeans_cluster_label'] = labels
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD pca_z_data_points TO tstc_result_df
# Perform PCA on FIX probe sessions on z-score values
filtered_tstc_z_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
filtered_tstc_z_results_df  = filtered_tstc_z_results_df[['name', 'novel_route_probe_fix_z_scr', 'no_cue_fix_z_scr',
                                                      'rotation_fix_z_scr', 'reversal_fix_z_scr']]
pca_data_np = filtered_tstc_z_results_df.iloc[:, 1:5].to_numpy()
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_data = pca.fit_transform(pca_data_np)
# Print the shape of the transformed data
#print("Transformed data shape:", pca_data.shape)
# Explained variance
#print("Explained variance ratio:", pca.explained_variance_ratio_)
pca_data_df = pd.DataFrame({'name' : filtered_tstc_z_results_df.iloc[:, 0].to_list()})
pca_data_points_list = [np.array(row) for row in pca_data]
pca_data_df['pca_z_data_points'] = pca_data_points_list
tstc_results_df.set_index('name', inplace=True)
pca_data_df.set_index('name', inplace=True)
tstc_results_df.update(pca_data_df)
tstc_results_df.reset_index(inplace=True)

#ADD pca_z_kmeans_cluster_label TO tstc_results_df
#k_matrix = np.column_stack((pca_data[:,0], pca_data[:,1]))
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(pca_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : filtered_tstc_z_results_df.iloc[:, 0].to_list()})
kmeans_data_df['pca_z_kmeans_cluster_label'] = labels
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD kmeans_minmax_cluster_labels_nr_nc_rt_rv, kmeans_minmax_cluster_cords_nr_nc_rt_rv TO tstc_results_df
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
tstc_minmax_nr_nc_rt_rv_df  = filtered_tstc_minmax_results_df[['name', 'novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr', 'reversal_fix_minmax_scr']]
tstc_minmax_nr_nc_rt_rv_np = filtered_tstc_minmax_results_df[['novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr', 'reversal_fix_minmax_scr']].to_numpy()
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(tstc_minmax_nr_nc_rt_rv_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : tstc_minmax_nr_nc_rt_rv_df.iloc[:, 0].to_list()})
kmeans_data_df['kmeans_minmax_cluster_labels_nr_nc_rt_rv'] = labels
kmeans_data_df['kmeans_minmax_cluster_cords_nr_nc_rt_rv'] = tstc_minmax_nr_nc_rt_rv_df.apply(
    lambda row: [row['novel_route_probe_fix_minmax_scr'], row['no_cue_fix_minmax_scr'], row['rotation_fix_minmax_scr'],
                 row['reversal_fix_minmax_scr']], axis=1)
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD kmeans_minmax_cluster_labels_nr_nc_rt, kmeans_minmax_cluster_cords_nr_nc_rt TO tstc_results_df
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
tstc_minmax_nr_nc_rt_df  = filtered_tstc_minmax_results_df[['name', 'novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']]
tstc_minmax_nr_nc_rt_np = filtered_tstc_minmax_results_df[['novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']].to_numpy()
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(tstc_minmax_nr_nc_rt_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : tstc_minmax_nr_nc_rt_df.iloc[:, 0].to_list()})
kmeans_data_df['kmeans_minmax_cluster_labels_nr_nc_rt'] = labels
kmeans_data_df['kmeans_minmax_cluster_cords_nr_nc_rt'] = tstc_minmax_nr_nc_rt_df.apply(
    lambda row: [row['novel_route_probe_fix_minmax_scr'], row['no_cue_fix_minmax_scr'], row['rotation_fix_minmax_scr']], axis=1)
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD kmeans_minmax_cluster_labels_nr_nc_rv, kmeans_minmax_cluster_cords_nr_nc_rv TO tstc_results_df
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
tstc_minmax_nr_nc_rv_df  = filtered_tstc_minmax_results_df[['name', 'novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'reversal_fix_minmax_scr']]
tstc_minmax_nr_nc_rv_np = filtered_tstc_minmax_results_df[['novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'reversal_fix_minmax_scr']].to_numpy()
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(tstc_minmax_nr_nc_rv_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : tstc_minmax_nr_nc_rv_df.iloc[:, 0].to_list()})
kmeans_data_df['kmeans_minmax_cluster_labels_nr_nc_rv'] = labels
kmeans_data_df['kmeans_minmax_cluster_cords_nr_nc_rv'] = tstc_minmax_nr_nc_rv_df.apply(
    lambda row: [row['novel_route_probe_fix_minmax_scr'], row['no_cue_fix_minmax_scr'], row['reversal_fix_minmax_scr']], axis=1)
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD kmeans_minmax_cluster_labels_nr_rv_rt, kmeans_minmax_cluster_cords_nr_rv_rt TO tstc_results_df
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
tstc_minmax_nr_rv_rt_df  = filtered_tstc_minmax_results_df[['name', 'novel_route_probe_fix_minmax_scr', 'reversal_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']]
tstc_minmax_nr_rv_rt_np = filtered_tstc_minmax_results_df[['novel_route_probe_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']].to_numpy()
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(tstc_minmax_nr_rv_rt_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : tstc_minmax_nr_rv_rt_df.iloc[:, 0].to_list()})
kmeans_data_df['kmeans_minmax_cluster_labels_nr_rv_rt'] = labels
kmeans_data_df['kmeans_minmax_cluster_cords_nr_rv_rt'] = tstc_minmax_nr_rv_rt_df.apply(
    lambda row: [row['novel_route_probe_fix_minmax_scr'], row['reversal_fix_minmax_scr'], row['rotation_fix_minmax_scr']], axis=1)
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# ADD kmeans_minmax_cluster_labels_nc_rv_rt, kmeans_minmax_cluster_cords_nc_rv_rt TO tstc_results_df
filtered_tstc_minmax_results_df = tstc_results_df[tstc_results_df['training_type'].isin(['PI+VC_f2', 
                                                                                   'PI+VC_f1'])]
tstc_minmax_nc_rv_rt_df  = filtered_tstc_minmax_results_df[['name', 'no_cue_fix_minmax_scr', 'reversal_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']]
tstc_minmax_nc_rv_rt_np = filtered_tstc_minmax_results_df[['no_cue_fix_minmax_scr', 'no_cue_fix_minmax_scr',
                                                      'rotation_fix_minmax_scr']].to_numpy()
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED_CLUSTERING)
kmeans.fit(tstc_minmax_nc_rv_rt_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_data_df = pd.DataFrame({'name' : tstc_minmax_nc_rv_rt_df.iloc[:, 0].to_list()})
kmeans_data_df['kmeans_minmax_cluster_labels_nc_rv_rt'] = labels
kmeans_data_df['kmeans_minmax_cluster_cords_nc_rv_rt'] = tstc_minmax_nc_rv_rt_df.apply(
    lambda row: [row['no_cue_fix_minmax_scr'], row['reversal_fix_minmax_scr'], row['rotation_fix_minmax_scr']], axis=1)
tstc_results_df.set_index('name', inplace=True)
kmeans_data_df.set_index('name', inplace=True)
tstc_results_df.update(kmeans_data_df)
tstc_results_df.reset_index(inplace=True)

# Convert names of training_subtype to more readable format
# tstc_results_df['training_subtype'] = tstc_results_df['training_subtype'].apply(convert_trial_subtype_names)

# Save data to parquet format
tstc_results_df.to_parquet('data/processed/two-start-two-choice-results.parquet', engine='pyarrow')
# Save data as pickle
#tstc_results_df.to_pickle('Data/two-start-two-choice-results-test.pkl')
