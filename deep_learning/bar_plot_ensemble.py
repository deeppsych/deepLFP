import os

import pandas as pd
import seaborn as sns
import pathlib
import numpy as np

from statannot import add_stat_annotation


def load_values(permutation_directory):
    p_values = {}
    for sub_dir in sorted(permutation_directory.iterdir()):

        p_value = np.loadtxt(sub_dir.joinpath('p_corrected_difference.txt'))
        subject_name = sub_dir.name
        p_values[subject_name] = p_value.item()

    return p_values


if __name__ == '__main__':

    ensemble_acc = pd.read_csv(f'{os.environ.get("HOME")}/deep_LFP/experiments/temporal_ensemble/5/results_df.csv')

    original_accuracy = pd.read_csv(f'{os.eviron.get("HOME")}/deep_LFP/results/deep_lfp_results_subject_specific.csv')
    original_accuracy.rename(columns={'Unnamed: 0': 'Subjects', 'Total Accuracy': 'Accuracy'}, inplace=True)
    original_accuracy.loc[:, 'Source'] = 'Original'

    ensemble_acc.loc[:, 'Subjects'] = [f'Patient {i}' for i in ensemble_acc.loc[:, 'Subjects'].values]
    original_accuracy.loc[:,'Subjects'] = [f'Patient {i.split(" ")[-1]}' for i in original_accuracy.loc[:, 'Subjects'].values]

    ensemble_acc = ensemble_acc[ensemble_acc.Measure == 'Accuracy']
    ensemble_acc.rename(columns={'Value': 'Accuracy'}, inplace=True)

    included_columns = ['Subjects', 'Source', 'Accuracy']
    total_accuracy = pd.concat((original_accuracy[included_columns], ensemble_acc[included_columns]))
    total_accuracy.rename(columns={'Subjects': 'Patients'}, inplace=True)
    total_accuracy['Patients'] = [s.split(' ')[1] for s in total_accuracy['Patients'].values]

    ax = sns.barplot(x='Patients', y=included_columns[-1], hue='Source', data=total_accuracy)
    perm_dir = pathlib.Path('/data/eaxfjord/deep_LFP/experiments/state_prediction_subject_temporal_ensemble_permutations/')
    p_values = load_values(permutation_directory=perm_dir)

    subs = ['tt', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7', 'ocdbd8', 'ocdbd9', 'ocdbd10', 'ocdbd11']
    p = []
    for s in subs:
        p.append(p_values[s])

    box_pairs = [(('1', 'Original'), ('1', 'Temporal ensemble')),
                 (('2', 'Original'), ('2', 'Temporal ensemble')),
                 (('3', 'Original'), ('3', 'Temporal ensemble')),
                 (('4', 'Original'), ('4', 'Temporal ensemble')),
                 (('5', 'Original'), ('5', 'Temporal ensemble')),
                 (('6', 'Original'), ('6', 'Temporal ensemble')),
                 (('7', 'Original'), ('7', 'Temporal ensemble')),
                 (('8', 'Original'), ('8', 'Temporal ensemble')),
                 (('9', 'Original'), ('9', 'Temporal ensemble')),
                 (('10', 'Original'),('10', 'Temporal ensemble')),
                 (('11', 'Original'), ('11', 'Temporal ensemble'))]

    add_stat_annotation(ax, pvalues=p, perform_stat_test=False, box_pairs=box_pairs,
                        x='Patients', y=included_columns[-1], hue='Source', data=total_accuracy)
