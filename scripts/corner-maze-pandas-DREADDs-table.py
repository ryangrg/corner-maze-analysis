from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sqlite3

#Global Variables
SEED_CLUSTERING = 100002
# KMEANS_INIT_CENTS_
# seed 100002 splits into the nicest clusters

# set panda display options
pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.expand_frame_repr', False)

# DATA INFERENCE FUNCTIONS: extra data we can infer from existing data
def approach_to_goal_from_cue(cue_goal_assignment):
    if cue_goal_assignment == 'N/NE' or cue_goal_assignment == 'N/NW':
        return 'toward'
    elif cue_goal_assignment == 'N/SE' or cue_goal_assignment == 'N/SW':
        return 'away'


def correct_route(start_arm, goal_location):
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


def egocentric_cue_direction(start_arm, goal_location, cue_orientation):
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
    return turn_1 + turn_2


def direct_route(route_correct, route_actual, errors):
    if errors > 0:
        return -1
    elif route_actual == route_correct:
        return 1
    else:
        return 0


def first_response_perserveration(route_correct, route_actual, session_type):
    reversal_session_types = ['Dark Reverse', 'Rotate Reverse', 'Fixed Cue Switch']
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
        session_type_set = set(row['unique_session_types'])
        fix_protocol = set(['Fixed Cue 1', 'Fixed Cue 2a', 'Fixed No Cue', 'Fixed Cue Rotate', 
                        'Fixed Cue Switch'])
        rot_protocol_fix_novelroute_rot_reversal = set(['Rotate Train', 'Rotate Detour', 
                                                'Rotate Reverse'])
        rot_protocol_rot_novelroute_fix_reversal = set(['Rotate Train', 'Rotate Detour Moving',
                                                'Fixed Cue Switch'])
        nc_protocol_cued_novelroute = set(['Dark Train', 'Dark Detour', 'Dark Reverse'])
        nc_protocol_uncued_novelroute = set(['Dark Train', 'Dark Detour No Cue', 'Dark Reverse'])

        if session_type_set == fix_protocol:
            return 'Fixed Frame Protocol', None
        elif session_type_set == rot_protocol_fix_novelroute_rot_reversal:
            return 'Rotating Frame Protocol', 'Fixed Novel Route Rotating Reversal'
        elif session_type_set == rot_protocol_rot_novelroute_fix_reversal:
            return 'Rotating Frame Protocol', 'Rotating Novel Route Fixed Reversal'
        elif session_type_set == nc_protocol_cued_novelroute:
            return 'No Cue Protocol', 'Cued Novel Route'
        elif session_type_set == nc_protocol_uncued_novelroute:
            return 'No Cue Protocol', 'Uncued Novel Route'
        else:
            return None, None


def set_sessions_to_acquisition(row):
    return len(row)


# Gives score for novel route trials in a novel route session
def novel_route_score(x):
    # Select trials after the 16th
    post_16_trials = x.iloc[16:]
    # Filter for 'novel' trial type
    novel_trials = post_16_trials[post_16_trials['trial_type'] == 'novel_route']
    # Count the number of trials with 0 errors
    zero_error_count = (novel_trials['errors'] == 0).sum()
    return zero_error_count


# Gives score for novel route trials in a novel route session
def standard_score_during_novel_route(x):
    # Select trials after the 16th
    post_16_trials = x.iloc[16:]
    # Filter for 'novel' trial type
    novel_trials = post_16_trials[post_16_trials['trial_type'] == 'standard']
    # Count the number of trials with 0 errors
    zero_error_count = (novel_trials['errors'] == 0).sum()
    return zero_error_count


# How many trials it takes to get to 7/8 correct in first 16 trials
def std_trials_crit(trial_scores):
    trial_scores = list(trial_scores)
    trials_to_crit = 7
    for i in range(10):
        perfect_trial_count = trial_scores[i:i+8].count(0)
        if (perfect_trial_count == 7 and trial_scores[i+7] > 0) or perfect_trial_count == 8:
            return trials_to_crit
        elif perfect_trial_count == 7:
            return trials_to_crit + 1
        trials_to_crit += 1
    return None



def reversal_trials_crit(trial_scores):
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

def reversal_block_scores(trial_scores):
    trial_scores = list(trial_scores)
    block_scores = []
    for i in range(0, len(trial_scores), 8):
        block_score = trial_scores[i:i+8].count(0)
        block_scores.append(block_score)
    return block_scores


def modified_z_score(arr):
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return 0.6745 * (arr - median) / mad 

# connect to sqlite database
con = sqlite3.connect('Data/MazeControl-DREADDs-Shaped.db')

# list subject ids to be used for conrner maze analysis papoer
subject_id = [81, 83, 84, 85, 86, 88, 89, 100, 106, 107, 108, 109, 110, 112, 113, 117, 118, 119, 120, 122]
# subject_id = [111, 118, 121] timed out on reversal and dropped from analysis

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
                                c.turn_2
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

# Column Description
# route_correct: This gives what should be the correct route for the trial as RR, LL, RL, LR
# route_actual: Gives the actual route taken in terms of the first two choices made. It is possible
#   that the second turn occurs at the unexpected intersection if the rat back tracks to the opposite
#   side of the maze after making the firts choice. This could be determined by looking at what corner
#   the rat entered after the making the second turn - although it is possible the rat could back track
#   but that is extremely unlikely.
# direct_route: this gives a 1 if the subject takes a direct route to the correct goal and a 0 if the
#   subject took an indirect route to the goal without entering a incorrect corner, and a -1 if they 
#   entered an incorrect corner.
# persevered: This gives a 1 if the subject made the old correct response during the reversal probe.
# add columns to trial data
# trial_type: tells if it's a standard trained trial, novel route trial, or reversal trial
trial_data_df['route_correct'] = pd.Series(dtype=str)
trial_data_df['route_actual'] = pd.Series(dtype=str)
trial_data_df['direct_route'] = pd.Series(dtype=int)
trial_data_df['persevered'] = pd.Series(dtype=int)
trial_data_df['trial_type'] = pd.Series(dtype=str)

# supply values to added columns
trial_data_df['route_correct'] = trial_data_df.apply(
    lambda row: correct_route(row['start_arm'], row['goal_location']), axis=1)
trial_data_df['route_actual'] = trial_data_df.apply(
    lambda row: actual_route(row['turn_1'], row['turn_2']), axis=1)
trial_data_df['direct_route'] = trial_data_df.apply(
    lambda row: direct_route(row['route_correct'], row['route_actual'], row['errors']), axis=1)
trial_data_df['persevered'] = trial_data_df.apply(
    lambda row: first_response_perserveration(row['route_correct'], 
                                       row['route_actual'], row['session_type']), axis=1)
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
# dreadds: two start two choice
dreadds_results_df = pd.DataFrame({'subject_id' : pd.Series(dtype=int),
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
                                'novel_route_preprobe_std_rscr' : pd.Series(dtype=int), # raw score on trained
                                'novel_route_preprobe_std_scr' : pd.Series(dtype=int), # percent score on trained
                                'novel_route_probe_std_rscr' : pd.Series(dtype=int), # raw score on trained trials after first 16 trials
                                'novel_route_probe_std_scr' : pd.Series(dtype=int), # percent score on trained trials after first 16 trials
                                'reversal_preprobe_rscr' : pd.Series(dtype=int), # rscr is number of trials to 7/8 correct here before reversal
                                'reversal_bscr' : pd.Series(dtype=object), # bscr is scr for each block of 8 trials
                                'reversal_rscr' : pd.Series(dtype=int), # rscr is number of trials to 7/8 reversal trials correct
                                'reversal_scr' : pd.Series(dtype=int), # Percentage based score of trials to criterion as totla number of window steps minus number trial to cirterion all divied by total number of window steps.
                                'reversal_persveration_rscr' : pd.Series(dtype=int), # Counted number of times rat took first route to the old goal
                                })

# ADD subject_id, name, sex and cue_goal_orientation TO dreadds_results_df DATAFRAME
#   names are added by using concat to add subject_df name data into two_start_two_choice_results
dreadds_results_df = pd.concat([dreadds_results_df, subject_df], ignore_index=True)

# ADD training_type, training_subtype TO dreadds_results_df DATAFRAME
unique_session_types = session_data_df.groupby('subject_id')['session_type'].apply(lambda x:
    list(set(x))).reset_index()
unique_session_types.rename(columns={'session_type': 'unique_session_types'}, inplace=True)
dreadds_results_df = dreadds_results_df.merge(
    unique_session_types, on='subject_id', how='left')
dreadds_results_df[['training_type', 'training_subtype']] = dreadds_results_df.apply(
    lambda row: pd.Series(set_session_type_and_subtype(row)),axis=1)

# ADD cue_approach TO dreadds_results_df DATAFRAME
dreadds_results_df['cue_approach'] = dreadds_results_df.apply(
    lambda row: approach_to_goal_from_cue(row['cue_goal_orientation']), axis=1)

# ADD acquisition_scrs TO dreadds_results_df DATAFRAME
session_filter = ['Fixed Cue 1', 'Rotate Train', 'Dark Train']
filtered_session_data_df = session_data_df[session_data_df['session_type'].isin(session_filter)]
acquisition_scrs_df = filtered_session_data_df.groupby(['subject_id', 'session_type'])['score'].apply(
    lambda x: [float(i) for i in x]).reset_index()

dreadds_results_df['acquisition_scrs'] = dreadds_results_df['acquisition_scrs'].combine_first(
    dreadds_results_df['subject_id'].map(acquisition_scrs_df.set_index('subject_id')['score']))

# ADD trials_to_acquisition TO dreadds_results_df DATAFRAME
dreadds_results_df['trials_to_acq'] = dreadds_results_df['acquisition_scrs'].apply(
    lambda x: len(x))

# ADD novel_route_preprobe_std_rscr, novel_route_preprobe_std_scr TO dreadds_results_df DATAFRAME
num_novel_route_preprobe_trials = 16
novel_route_session_filter = ['Fixed Cue 2a', 'Dark Detour', 'Dark Detour No Cue', 'Rotate Detour',
                              'Rotate Detour Moving'] 
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
dreadds_results_df.set_index('subject_id', inplace=True)
novel_route_preprobe_std_scr_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(novel_route_preprobe_std_scr_df)
dreadds_results_df.reset_index(inplace=True)

# Add novel_route_probe_rscr, novel_route_probe_scr, novel_route_probe_fix_minmax_scr, novel_route_probe_fit_modz_scr, 
# novel_route_probe_fix_z_scr, TO dreadds_results_df
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

dreadds_results_df.set_index('subject_id', inplace=True)
novel_route_probe_scrs_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(novel_route_probe_scrs_df)
dreadds_results_df.reset_index(inplace=True)

# ADD novel_route_probe_std_rscr, novel_route_probe_std_scr TO dreadds_results_df
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
dreadds_results_df.set_index('subject_id', inplace=True)
novel_route_probe_std_rscr_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(novel_route_probe_std_rscr_df)
dreadds_results_df.reset_index(inplace=True)

# ADD reversal_preprobe_rscr TO tstc_results_df
reversal_trial_filter = ['Fixed Cue Switch', 'Dark Reverse', 'Rotate Reverse']
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
dreadds_results_df.set_index('subject_id', inplace=True)
reversal_preprobe_rscr_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(reversal_preprobe_rscr_df)
dreadds_results_df.reset_index(inplace=True)

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
dreadds_results_df.set_index('subject_id', inplace=True)
reversal_scrs_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(reversal_scrs_df)
dreadds_results_df.reset_index(inplace=True)

reversal_bscr_df = (
    reversal_trials_df
    .sort_values(by=['session_id', 'trial_number'])
    .groupby('session_id')
    .apply(lambda x: reversal_block_scores(x['errors']), include_groups=False)
    .reset_index(name='reversal_bscr')
)
reversal_bscr_df = pd.merge(
    reversal_bscr_df,
    reversal_trials_df[['session_id', 'subject_id', 'session_type']].drop_duplicates(),
    on='session_id'
)
dreadds_results_df.set_index('subject_id', inplace=True)
reversal_bscr_df.set_index('subject_id', inplace=True)
dreadds_results_df.update(reversal_bscr_df)
dreadds_results_df.reset_index(inplace=True)

# Save data to parquet format
dreadds_results_df.to_parquet('Data/DREADDs-results.parquet', engine='pyarrow')
# Save data as pickle
#dreadds_results_df.to_pickle('Data/two-start-two-choice-results.pkl')


