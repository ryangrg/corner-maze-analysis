import pandas as pd
import sqlite3

# This code will remove rows were session number is labeled as X and will merge rows and the values
# within those rows where the session numbers are the same. This removes sessions that had issues with 
# the maze and had to be stopped or session in which rats did not meat criterion on standard trials
# before starting the session. This also joins sessions in which an issue happened on the maze that
# required the session to be stopped, but the issue was quickly resolved and the rat was able to coninute.

def copy_db(original_db_path, backup_db_path):
    # Connect to the original database
    with sqlite3.connect(original_db_path) as original_conn:
        # Connect to the new (backup) database
        with sqlite3.connect(backup_db_path) as backup_conn:
            # Use the backup method to copy the original database to the new database
            original_conn.backup(backup_conn)

# Paths to the original and backup databases
original_db_path = 'data/raw/MazeControl.db'
shaped_db_path = 'data/processed/MazeControl-clean.db'

# Call the function to copy the database
copy_db(original_db_path, shaped_db_path)


# set panda display options
pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.expand_frame_repr', False)

# connect to sqlite database
conn = sqlite3.connect('data/processed/MazeControl-clean.db')
cursor = conn.cursor()

# Enter Rat IDs you want data for here
# create id list for subjects you want from database
subject_id = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 71, 
              73, 75, 76, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 97, 98, 
              99, 100, 106, 107, 108, 109, 110, 112, 113, 117, 119, 120, 122]

# Remove all data associated with subjects we are not keeping
placeholders = ",".join("?" for _ in subject_id)
session_filter = f'''SELECT session_id
                     FROM session
                     WHERE subject_id NOT IN ({placeholders})'''

cursor.execute(f'''DELETE FROM trial
                   WHERE session_id IN ({session_filter})''', tuple(subject_id))
cursor.execute(f'''DELETE FROM session_event
                   WHERE session_id IN ({session_filter})''', tuple(subject_id))
cursor.execute(f'''DELETE FROM session
                   WHERE subject_id NOT IN ({placeholders})''', tuple(subject_id))

# Delete rows from sessions that had issues with the maze and it had to stop and was rerun or the
# subjects did not meet criterium for the probe session and it was stopped. These are labled with
# session number 'X'
# Clean up rows that were bad session starts and not actual runs
# Create query to find session tables from subject_id and blank duration fields
for sbj in subject_id:
    select_query = f'''SELECT *
                        FROM session
                        WHERE subject_id = ?
                        AND session_duration IS NULL

                        UNION

                        SELECT *
                        FROM session
                        WHERE subject_id = ?
                        AND session_number IS 'X'
                        '''
    
    cursor.execute(select_query, (sbj, sbj,))
    rows = cursor.fetchall()
    if rows:
        for data in rows:
            # Delete trial data with session_id
            delete_query = f'''DELETE FROM trial
                               WHERE session_id = ? 
                            '''
            cursor.execute(delete_query, (data[0],))

            # Delete session_event data with session_id
            delete_query = f'''DELETE FROM session_event
                               WHERE session_id = ?
                            '''
            cursor.execute(delete_query, (data[0],))

            # Delete session with session_id
            delete_query = f'''DELETE FROM session
                               WHERE session_id = ?
                            '''
            cursor.execute(delete_query, (data[0],))

double_session = []
# find session that were split in two and combine them
for sbj in subject_id:
    select_query = f'''SELECT t1.*
                       FROM session AS t1
                       JOIN session AS t2 ON t1.session_number = t2.session_number
                       WHERE t1.session_id < t2.session_id
                       AND t1.subject_id = ?
                       AND t2.subject_id = ?
                       
                       UNION
                       
                       SELECT t2.*
                       FROM session AS t1
                       JOIN session AS t2 ON t1.session_number = t2.session_number
                       WHERE t1.session_id < t2.session_id
                       AND t1.subject_id = ?
                       AND t2.subject_id = ?
                    '''
    split_session = cursor.execute(select_query, (sbj, sbj, sbj, sbj))
    if split_session:
        for splt in split_session:
            double_session.append(list(splt))

for i, data in enumerate(double_session):
    if i % 2 == 0:
        print(data)
        data[7] = data[7] + double_session[i+1][7]
        data[8] = data[8] + double_session[i + 1][8]
        data[9] = data[9] + double_session[i + 1][9]
        data[10] = (float(data[10]) + float(double_session[i + 1][10]))/2
        data[11] = data[11] + double_session[i + 1][11]
        data[12] = (data[12] + double_session[i + 1][12])/2
        update_query = '''UPDATE trial
                          SET session_id = ?
                          WHERE session_id = ?
                       '''
        cursor.execute(update_query, (data[0], double_session[i + 1][0]))
        update_query = '''UPDATE session_event
                          SET session_id = ?
                          WHERE session_id = ?
                       '''
        cursor.execute(update_query, (data[0], double_session[i + 1][0]))
        update_query ='''UPDATE session
                         SET session_duration = ?,
                             total_trials = ?,
                             total_perfect = ?,
                             score = ?,
                             total_errors = ?,
                             average_errors = ?
                          WHERE session_id = ?'''
        cursor.execute(update_query, (data[7], data[8], data[9], str(data[10]), data[11],
                                      data[12], data[0]))
        delete_query = '''DELETE FROM session
                          WHERE session_id = ?'''
        cursor.execute(delete_query, (double_session[i+1][0],))

conn.commit()
conn.close()
