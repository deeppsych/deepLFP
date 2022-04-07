import pathlib
import os

import joblib
from boosted_trees.classic_machine_learning import prepare_subject_data, MyLabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics._ranking import _binary_clf_curve
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_theme()

import numpy as np
import scipy

if __name__ == '__main__':

    # load models for subject 2 and get predictions for 1 vs all for obsessions
    best_model = joblib.load(f'{os.getenv("HOME")}/deep_LFP/saved_models/ocdbd2_gridsearch.pkl')
    bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'low_gamma': [30, 50],
             'high_gamma': [50, 100]}
    band_names = list(bands.keys())
    df_test = prepare_subject_data(train=False, bands=bands)
    df_train = prepare_subject_data(train=True, bands=bands)

    subject_test = df_test[df_test.subjects == 'ocdbd2']
    subject_train = df_train[df_train.subjects == 'ocdbd2']

    le = MyLabelEncoder()
    y_train = le.fit(subject_train['state'].values).transform(subject_train['state'].values)
    y_test = le.transform(subject_test['state'].values)

    obs_index = y_train == 2
    not_obs_index = y_train != 2
    y_train[obs_index] = 1
    y_train[not_obs_index] = 0

    X_test = subject_test[band_names].values
    X_train = subject_train[band_names].values

    test_scores_all = best_model.predict_proba(X_test)

    auc = roc_auc_score(y_test==2, test_scores_all[:, 2])

    fps, tps, thresholds = _binary_clf_curve(y_test==2, test_scores_all[:, 2])

    n = test_scores_all.shape[0]
    net_benefit = (tps/n) - (fps/n) * thresholds/(1-thresholds)

    sns.lineplot(x=thresholds, y=net_benefit, label='Boosted trees', ci=None)

    prevalence = (y_test==2).sum() / n
    treat_all = prevalence - ((1-prevalence) * (np.linspace(0, 1, 100)/(1-np.linspace(0, 1, 100))))
    treat_all[-1] = np.nan
    sns.lineplot(x=np.linspace(0, 1, 100), y=treat_all, label='Always stimulate', linestyle='--', ci=None)
    sns.lineplot(x=[0,1], y=0, label='Never stimulate', linestyle='--', ci=None)

    deep_test = np.fromfile(f'{os.environ.get("HOME")}/deep_LFP/saved_models/ocdbd2_deeplearning_test.npy', dtype='int64')

    # load ensemble predictions
    folder = pathlib.Path(f'{os.environ.get("HOME")}/deep_LFP/experiments/InceptionTime_3sec_state_prediction_final_subject_level_final_correct_filter_11subs/ocdbd2/')
    all_preds = np.empty((deep_test.shape[0], 5))
    for i in range(5):
        preds_file = folder.joinpath(f'{i}/predictions.npy')
        preds = scipy.special.softmax(np.load(preds_file), axis=1)[:, 2]
        all_preds[:,i] = preds
    predictions = np.mean(all_preds, axis=1)

    deep_auc = roc_auc_score(deep_test == 2, predictions)

    fps, tps, thresholds = _binary_clf_curve(deep_test==2, predictions)

    n = predictions.shape[0]
    net_benefit = (tps/n) - (fps/n) * thresholds/(1-thresholds)
    sns.lineplot(x=thresholds, y=net_benefit, label='InceptionTime', ci=None)

    # load temporal ensemble predictions
    temporal_ensemble_path = f'{os.getenv("HOME")}/deep_LFP/experiments/temporal_ensemble/5/ocdbd2/ensemble_predictions.npy'
    temporal_ensemble_preds = scipy.special.softmax(np.load(temporal_ensemble_path), axis=1)[:, 2]
    temporal_ensemble_preds = temporal_ensemble_preds.reshape(48,5).mean(axis=1)

    true_labels = (deep_test==2).reshape(48,5)[:, 0]
    temporal_ensemble_auc = roc_auc_score(true_labels, temporal_ensemble_preds)
    fps, tps, thresholds = _binary_clf_curve(true_labels, temporal_ensemble_preds)
    n = temporal_ensemble_preds.shape[0]
    net_benefit = (tps/n) - (fps/n) * thresholds/(1-thresholds)
    sns.lineplot(x=thresholds, y=net_benefit, label='InceptionTime temporal ensemble', ci=None)

    plt.ylim([-0.1, 0.3])
    plt.show()