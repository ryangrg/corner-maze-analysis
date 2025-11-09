# Corner Maze Analysis

## Overview

This repository houses analysis utilities and figures for the tentatively titled "Neural Mechanisms of Flexible and Inflexible Navigational Behaviors", including SQLite database helpers, exploratory notebooks, and plotting scripts. Use this document as a starting point and update the placeholder sections with project-specific details.

## Repository Layout

- `data/`: Raw and processed MazeControl databases or exports.
- `scripts/`: Python utilities such as `shape_2c2s_database.py` for cleaning and transforming sessions.
- `notebooks/`: Jupyter notebooks for exploratory analysis or reporting.
- `plots/`: Generated figures kept under version control when needed.
- `test/`: Unit or integration tests (expand or remove if unused).

# Raw Data Overview
`data/raw/MazeControl.db`
MazeControl.db holds subject, session and trial data from all experiments I conducted on the Corner Maze. Session numbers labeled as X are removed and where session numbers are the same the session are combined. This removes sessions that had issues with the maze and had to be stopped or sessions in which rats did not meat criterion on standard trials before starting the session. This also joins sessions in which an issue happened on the maze that required the session to be stopped, but the issue was quickly resolved and the rat was able to coninute.

The subjects used in this anaylsis are:
subject_id = [47<sup>*</sup>, 48<sup>*</sup>, 49, 50<sup>*</sup>, 51, 52, 53<sup>*</sup>, 54, 55, 56,57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 71, 73, 75, 76, 78, 80, 82, 84, 85, 86, 87, 88<sup>*</sup>, 89, 90, 93<sup>*</sup>, 94, 95, 96<sup>*</sup>, 97<sup>*</sup>, 98, 99<sup>*</sup>, 100, 106, 112, 118, 119, 120, 122, 100]

# <sup>*</sup>Subjects that had sessions that needed to be removed for one reason or another or sessions that needed to be merged.
47: Novel Route Session Drop on PI+VC Training did not reach criterion in the start of the session to start probe trials.
48: Had a full acqucition session run between the Novel Route probe and No Cue, this is removed.
50: Had to have a No Cue session merged, no issues with couter balancing trial types
53: Had to have No Cue sessions merged, not perfectly counterbalnced 17 trials from west arm and 15 trials form east arm. We take this as a trivial issue.
80: Had to end a No Cue Detour session 4 trials into the probe trials do to maze issues. The subject experienced three Detour trials and session terminated. Subject didn't not run until the next day. Hard to say what effect there could have been on nexts days performance but first four Detour trials had errors so it did not seem to boost performance. Subject had a total of 5 Detour trials with errors.
88: Dropped reversal sessions that failed to get criterion in first 16 trials.
93: Two Dropped reversal sessions that failed to get to criterion in first 16 trials.
96: No Cue Novel Route session dropped for failure to get to criterion in first 16 trials.
97: No Cue Novel Route session dropped for failure to get to criterion in first 16 trials.
99: Dropped reversal sessions that failed to get criterion in first 16 trials.
106: Accidentally had an acquisition session between novel route and switch/reversal, this was removed from the data. This rat also continued training after reversal to test DREADDs on a detour session from opposite arm of initial detour and switched to opposite corner for goal. These sessions weren't used for the paper and are dropped from analysis here.
112: Issue occurred with maze on an acquisition session which is removed along with secondary detour session from an alternate arm for a protocol that was dropped. No issues with first detour session.
122: Just has bad session starts where something went wrong on setup, didn't actually run on deleted session
123: Lots of false starts setting up maze, no issue with subject.


