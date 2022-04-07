import os
from os.path import join as opj
import pandas as pd
import numpy as np
import h5py
from collections import defaultdict


def load_session_data(session_dir, session_name, subject):
    """
    Loads data from each session
    Args:
        session_dir:        Directory where session data is stored
        session_name:       Name of session (the date usually)
        subject:            Code for subject

    Returns:

    """
    data_summary_file = [name for name in os.listdir(session_dir) if name.startswith('DataSummary')]
    if 'tt/Session_2015_08_18' in session_dir:
        data_summary_file = ['/data/shared/OCD_DBS_bidirectional/Ruwe_Data/OCD_DBS_bidirectional/ocdbd1/visit1/DataSummary_tt_2015_08_18 (OCDBD1) exact timing.xlsm']
    if not data_summary_file:
        print(session_dir)
        all_data = []
        return all_data, all_data, all_data
    if len(data_summary_file) > 1:
        my_filter = ['exact timing' in files for files in data_summary_file]
        data_summary_file = [fname for indx, fname in enumerate(data_summary_file) if my_filter[indx] == True][0]
    else:
        data_summary_file = data_summary_file[0]
    if 'tt/Session_2015_08_18' not in session_dir:
        data_summary = pd.read_excel(opj(session_dir, data_summary_file), sheet_name='Recording List', header=5)
    else:
        data_summary = pd.read_excel(data_summary_file, sheet_name='Recording List', header=5)
    data_summary.dropna(axis=1, how='all', inplace=True)
    data_summary.dropna(subset=['File name'], inplace=True)
    data_summary.Notes = data_summary.Notes.astype('str')
    data_summary.sort_values(by=['File name'], inplace=True)
    data_summary = data_summary.reset_index(drop=True)

    if session_name == 'visit_1':
        # create new column with "task"
        start_index = []
        end_index = 0
        task = []
        round = []
        round_indx = 1
        for ix, notes in enumerate(data_summary.Notes):
            start_bool = any(substring in notes.lower() for substring in ['vas', 'nan', 'end', 'stop'])
            start_bool = not start_bool
            if 'VAS gedaan na start meting' in notes:
                start_bool = True
            end_bool = any(substring in notes.lower() for substring in ['end', 'stop'])
            if start_bool:
                start_index = ix
                if data_summary.iloc[ix].Notes.lower().split(' ')[0] == 'sart':
                    task_name = data_summary.iloc[ix].Notes.lower().replace('sart ', '').strip()
                else:
                    task_name = data_summary.iloc[ix].Notes.lower().replace('start ', '').strip()
                if ' ' in task_name[:]:
                    round_name = task_name.split(' ')[-1]
                    if len(round_name) > 1:
                        if round_name[-1] != ')':
                            round_name = round_name[-1]
                        else:
                            round_name = str(round_indx)
                    task_name = task_name[:task_name.index(' ')]
                else:
                    round_name = str(round_indx)
                if task_name == 'relief':
                    round_indx += 1
            if end_bool:
                end_index = ix
                start_index = []
            if start_index and end_index:
                task.append(task_name)
                round.append(round_name)
            elif end_index == ix:
                task.append(task_name)
                round.append(round_name)
            elif start_index == end_index:
                task.append(task_name)
                round.append(round_name)
            else:
                task.append('no_task')
                round.append('no_round')

        data_summary['Task'] = task
        data_summary['Round'] = round

        if 'ocdbd2' in data_summary_file:
            # the Notes field is weird for this subject I assign manually according to
            # /data/shared/OCD_DBS_bidirectional/Gesorteerde_data/Visit1/
            data_summary.loc[12:17, 'Round'] = 2  # obs r2
            data_summary.loc[18:23, 'Round'] = 3  # obs r3
            data_summary.loc[24:35, 'Round'] = 4  # obs r4

            data_summary.loc[42:47, 'Round'] = 2  # comp r2
            data_summary.loc[48:53, 'Round'] = 3  # comp r3
            data_summary.loc[54:65, 'Round'] = 4  # comp r4

            data_summary.loc[78:83, 'Round'] = 3  # rel r3, r4 is missing??

            data_summary.loc[90:95, 'Round'] = 3  # base r3
            data_summary.loc[96:101, 'Round'] = 4  # base r4   ]

        # locate files I'm interested in
        index = [(data_summary.Task != 'no_task') &
                 (data_summary.Notes != 'Test') &
                 (~data_summary.Notes.str.lower().str.startswith('Gain'.lower())) &
                 (data_summary.Freq == 422)]
        index = index[0][:]

        [fnames, tasks, round, channels] = [data_summary[index]['File name'].values,
                                            data_summary[index]['Task'].values,
                                            data_summary[index]['Round'].values,
                                            data_summary[index]['Ch1'].values]

        tasks = ['compulsions' if x == 'compulsies' else x for x in tasks]
        round = [str(r) for r in round]


    else:
        strings = 'baseline|Baseline|Wash-out|Resting state|Washout'

        fnames = data_summary[((data_summary.Ch1 == '0-3') | (data_summary.Ch1 == '3-0')) &
                              (data_summary.Notes != 'Test') & (data_summary.Freq == 422) &
                              (~data_summary.Notes.str.lower().str.startswith('Gain'.lower())) &
                              (data_summary.Notes.str.contains(strings))]['File name'].values
        tasks = ['no_task'] * len(fnames)
        round = ['no_round'] * len(fnames)
        channels = ['same_channel'] * len(fnames)

    all_data = nested_dict()

    # load data from xml files and append to matrix
    for ix, fname in enumerate(fnames):
        if fname == 'ocdbd7_2017_03_07_08_07_24__MR_0.xml':  # this one was empty
            continue
        if fname == 'ocdbd10_2019_06_13_13_31_38__MR_0.xml' or fname == 'ocdbd10_2019_06_13_13_36_53__MR_1.xml':  # dbs on
            continue
        if 'ocdb11' in fname:
            fname = fname.replace('ocdb11', 'ocdbd11')
        txt_file = opj(session_dir, (fname.split('.')[0] + '.txt'))
        data = np.loadtxt(txt_file, dtype=float, delimiter=',')
        data = np.delete(data, [1, 3, 4, 5], axis=1)  # delete empty channels
        data = data[1000:, :]  # remove artifact at beginning which affects about 2 seconds
        all_data[tasks[ix]][round[ix]][channels[ix]] = data

    attribute_dict = {'summary_file': data_summary_file}

    return all_data, attribute_dict, data_summary


def nested_dict():
    return defaultdict(nested_dict)


def clean_hdf5(data):
    """Clean hdf5 file, remove dbs sessions and empty sessions"""
    dbs_list = f'{os.environ.get("HOME")}/deep_LFP/data/dbs_visits.csv'
    df = pd.read_csv(dbs_list)
    dset_names = [(subject + '/' + session) for (subject, session) in zip(df.subjects.values, df.dbs_sessions.values)]

    # remove sessions with dbs
    for i in range(len(dset_names)):
        print(dset_names[i])
        sub = dset_names[i].split('/')[0]
        sess = dset_names[i].split('/')[1]
        del data[sub][sess]

    # remove sessions with other issues.
    # del data['ocdbd3']['Session_2015_11_17_Tuesday']
    # del data['ocdbd7']['Session_2017_03_06_Monday']
    # del data['ocdbd8']['Session_2017_11_08_Wednesday']
    # del data['ocdbd9']['Session_2018_03_07_Wednesday']
    # del data['tt']['Session_2015_12_01_Tuesday']


def save_dict_to_hdf5(dic, filename):
    """
    Saves nested dictionary to hdf5 file
    """
    with h5py.File(filename, 'w') as h5file:
        save_nested_dict_to_group(h5file, '/', dic)


def save_nested_dict_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, defaultdict):
            save_nested_dict_to_group(h5file, path + key + '/', dict(item))
        elif isinstance(item, dict):
            save_nested_dict_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    """
    Loads nested dictionary from hdf5
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def main():
    """
    Goes through the raw data and loads into a nested dictionary which is then saved as an hdf5 file
    Returns:

    """
    data_dir = 'PATH_TO_DATA'

    subjects = [name for name in os.listdir(data_dir) if os.path.isdir(opj(data_dir, name))]
    subjects = [name for name in subjects if name != 'Datasummary leeg']

    visit_file = f'{os.environ.get("HOME")}/deep_LFP/data/visits.csv'
    visits = pd.read_csv(visit_file)
    all_data = nested_dict()
    for subject in subjects:
        subject_dir = opj(data_dir, subject)
        sessions = [name for name in os.listdir(subject_dir) if os.path.isdir(opj(subject_dir, name))]
        sessions.sort()
        for session in sessions:
            print(subject, session)
            session_dir = opj(subject_dir, session)
            if visits[visits.subjects == subject].Visit_1.values[0] == session:
                session_name = 'visit_1'
            else:
                session_name = session
            session_data, session_attributes, summary_data = load_session_data(session_dir, session_name, subject)
            print(session_dir)
            if session_data:
                all_data[subject][session_name] = dict(session_data)

    clean_hdf5(all_data)  # don't need if not using resting state

    # save all_data dictionary
    save_dict_to_hdf5(all_data, 'data/nested_dictionary_11subjects22112020.hdf5')


if __name__ == "__main__":
    main()
