import os
import random
import warnings
import pathlib
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, permutation_test_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import column_or_1d
from catboost import CatBoostClassifier
import catboost
import joblib
import mne
import shap
from scipy.integrate import simps
import seaborn as sns
from matplotlib import pyplot as plt

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def reshape_df(df):
    df['sample_id'] = df.groupby(['subjects', 'state', 'rounds', 'channel', 'side', 'band']).cumcount()
    df = df.pivot(index=['subjects', 'state', 'rounds', 'channel', 'side', 'sample_id'], columns='band',
                  values='bandpower')
    df = df.reset_index()
    return df


def get_model(model='logistic'):
    if model == 'logistic':
        param_grid = [
            {'classifier': [LogisticRegression()],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': np.logspace(-4, 4, 20),
             'classifier__solver': ['saga']},
        ]
    elif model == 'catboost':
        param_grid = [
            {
                'classifier': [CatBoostClassifier(thread_count=3)],
                'classifier__iterations': [250, 100, 500, 1000],
                'classifier__depth': [3, 1, 6],
                'classifier__loss_function': ['MultiClass'],
                'classifier__l2_leaf_reg': [3, 1, 5, 10],
                'classifier__border_count': [32, 5, 20, 50],
                'classifier__logging_level': ['Silent'],
                'classifier__random_seed': [42]},
        ]

    return param_grid


def nested_cv(power, band_names, model_name='logistic'):
    """
    Nested cross validation

    Args:
        power:          Spectral power in bands
        band_names:     names of bands to include as features
        model_name:     Which model to use, catboost or a logistic regression

    Returns:

    """
    # train test split
    subjects = power['subjects'].unique()

    seed = 42
    rng = random.Random(seed)  # my random number generator
    rng.shuffle(subjects)

    results = pd.DataFrame(index=subjects, columns=['accuracy'])
    for i, name in enumerate(subjects):
        print(f'Outer fold {i}')
        # one test subject
        test_subject = subjects[i]

        train_subjects = [s for s in subjects if s not in [test_subject]]

        test_df = power[power['subjects'] == test_subject]
        train_df = power[power['subjects'].isin(train_subjects)]

        train_df, test_df = preprocess_data(band_names, test_df, train_df)

        X_train, y_train = train_df[band_names].values, train_df['state'].values
        X_test, y_test = test_df[band_names].values, test_df['state'].values
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        pipe = Pipeline([('classifier', LogisticRegression())])

        param_grid = get_model(model_name)

        cv_iterator = extract_cv_iterator(train_df, train_subjects)

        if model_name == 'catboost':
            n_jobs = 24
        else:
            n_jobs = 20
        clf = GridSearchCV(pipe, param_grid=param_grid, cv=cv_iterator, verbose=10, n_jobs=n_jobs,
                           scoring='balanced_accuracy', error_score=np.nan)
        # Fit on data
        print('Searching for hyperparameters')
        best_clf = clf.fit(X_train, y_train)
        print('Hyperparameters found')

        # predict on test subject
        y_test_preds = best_clf.predict(X_test)

        accuracy = balanced_accuracy_score(y_test, y_test_preds)
        print(f"Accuracy of fold {i} was {accuracy:.3f}")

        results.loc[name, 'accuracy'] = accuracy
        params = [(k.split('__')[-1], v) for k, v in best_clf.best_params_.items()]
        params[0] = ('classifier', model_name)

        for k, v in params:
            results.loc[name, k] = v

    return results


def preprocess_data(band_names, test_df, train_df):
    train_df = reshape_df(train_df)
    test_df = reshape_df(test_df)
    # standardize power using mean and std from training set
    power_means = train_df[band_names].mean()
    power_std = train_df[band_names].std()
    train_df[band_names] = (train_df[band_names] - power_means) / power_std
    test_df[band_names] = (test_df[band_names] - power_means) / power_std
    return train_df, test_df


def extract_cv_iterator(df, train_subjects):
    # generate iterator for leave one subject out cv split for cross validation
    cv_iterator = []
    for j in range(len(train_subjects)):
        val_subject = train_subjects[j]
        inner_train_subjects = [s for s in train_subjects if s != val_subject]
        test_indices = df[df['subjects'] == val_subject].index.values.astype(int)
        train_indices = df[df['subjects'].isin(inner_train_subjects)].index.values.astype(int)
        cv_iterator.append((train_indices, test_indices))
    return cv_iterator


def load_data(path):
    df = pd.read_pickle(path)
    array = np.load(path.with_suffix('.npy'))
    return array, df


def get_power(f, p, band):
    low, high = band
    # Frequency resolution
    freq_res = f[1] - f[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(f >= low, f <= high)
    bp = simps(p[idx_band].squeeze(), dx=freq_res)

    return bp


def calculate_power(bands, df):
    power_df = pd.DataFrame(columns=['subjects', 'side', 'band', 'bandpower', 'sample_id'])
    subjects = df['subjects'].unique()

    power_data = []
    for subject in subjects:
        subject_data = df[df.subjects == subject]
        sample_numbers = subject_data.sample_no
        for sample_id in sample_numbers:
            sample_data = subject_data[subject_data.sample_no == sample_id]
            for side in ['left', 'right']:
                sample = sample_data[f'{side}_data'].values[0]
                if sample.shape[0] < 3 * 422:
                    print(f'Lenght of sample was {sample.shape[0]} which is less than {3 * 422}')
                    break
                else:
                    p, f = mne.time_frequency.psd_array_welch(sample, 422, fmin=0.5, fmax=100, n_fft=3 * 422,
                                                              verbose=0,
                                                              average=None)
                    for band in bands:
                        power = get_power(f, p, bands[band])
                        power_data.append({'subjects': subject,
                                           'sample_id': sample_id,
                                           'side': side,
                                           'band': band,
                                           'bandpower': power})
    power_df = power_df.append(power_data)
    return power_df


def calculate_power_array(bands, df, array):
    sf = 211
    power_df = pd.DataFrame(columns=['subjects', 'state', 'side', 'band', 'bandpower', 'sample_id'])
    subjects = df['subjects'].unique()

    power_data = []
    for subject in subjects:
        subject_index = df.subjects == subject
        subject_df_data = df[subject_index]
        subject_array_data = array[subject_index]
        sample_numbers = subject_df_data.sample_no
        for sample_id in sample_numbers:
            sample_index = subject_df_data.sample_no == sample_id
            sample_array_data = subject_array_data[sample_index]
            sample_data = subject_df_data[sample_index]
            state = sample_data['state'].values[0]
            for ix, side in enumerate(['left', 'right']):
                sample = sample_array_data.squeeze()[ix, :]
                if sample.shape[0] < 3 * sf:
                    print(f'Lenght of sample was {sample.shape[0]} which is less than {3 * sf}')
                    break
                else:
                    p, f = mne.time_frequency.psd_array_welch(sample, sf, fmin=0.5, fmax=100, n_fft=3 * sf,
                                                              verbose=0,
                                                              average=None)
                    for band in bands:
                        power = get_power(f, p, bands[band])
                        power_data.append({'subjects': subject,
                                           'state': state,
                                           'sample_id': sample_id,
                                           'side': side,
                                           'band': band,
                                           'bandpower': power})
    power_df = power_df.append(power_data)
    return power_df


class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


def prepare_subject_data(train=True, bands=None):
    if train:
        path = pathlib.Path(f'{os.getenv("HOME")}/deep_LFP/data/states/training_sample11_non_overlapping_states_3_sec_resampled_rounds.pkl')
    else:
        path = pathlib.Path(f'{os.getenv("HOME")}/deep_LFP/data/states/test_samples11_non_overlapping_states_3sec_resampled.pkl')
    array, df = load_data(path)
    df['sample_no'] = df.groupby(['subjects']).cumcount()
    df_train = calculate_power_array(bands, df, array)
    df_train['sample_no'] = df_train.groupby(['subjects', 'state', 'side', 'band']).cumcount()
    df_train = df_train.pivot(index=['subjects', 'state', 'side', 'sample_no'], columns='band',
                              values='bandpower').reset_index()
    df_train['state'] = pd.Categorical(df_train['state'], df.state.unique())

    return df_train


if __name__ == '__main__':
    subjects = ['tt', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7',
                'ocdbd8', 'ocdbd9', 'ocdbd10', 'ocdbd11']
    analysis_type = 'subject_level_state_prediction'  # either group, subject_prediction or subject_level_state_prediction
    model_name = 'catboost'
    bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'low_gamma': [30, 50],
             'high_gamma': [50, 100]}
    band_names = list(bands.keys())
    feature_names = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
    states = ['baseline', 'compulsions', 'obsessions', 'relief']

    #  group level analysis
    if analysis_type == 'group':
        power = pd.read_csv('power_samples.csv')

        results = nested_cv(power, band_names, model_name)
        results.to_csv(f'{model_name}_group_level.csv')

    # subject prediction
    if analysis_type == 'subject_prediction':
        path = pathlib.Path('./data/resting_state/training_sample11_non_overlapping_resting_state_3_sec.pkl')
        array, df = load_data(path)

        df_train = calculate_power(bands, df)
        df_train['sample_id'] = df_train.groupby(['subjects', 'side', 'band']).cumcount()
        df_train = df_train.pivot(index=['subjects', 'side', 'sample_id'], columns='band',
                                  values='bandpower').reset_index()
        df_train['subjects'] = pd.Categorical(df_train['subjects'], subjects)
        df_train = df_train.sort_values(by='subjects')

        test_path = pathlib.Path('./data/resting_state/test_samples11_non_overlapping_resting_state_3sec.pkl')
        _, test_df = load_data(test_path)

        df_test = calculate_power(bands, test_df)
        df_test['sample_id'] = df_test.groupby(['subjects', 'side', 'band']).cumcount()
        df_test = df_test.pivot(index=['subjects', 'side', 'sample_id'], columns='band',
                                values='bandpower').reset_index()
        df_test['subjects'] = pd.Categorical(df_test['subjects'], subjects)
        df_test = df_test.sort_values(by='subjects')

        le = MyLabelEncoder()
        y_train = le.fit(df_train['subjects'].values).transform(df_train['subjects'].values)
        y_test = le.transform(df_test['subjects'].values)

        model = CatBoostClassifier(thread_count=2, logging_level='Silent', loss_function='MultiClass',
                                   random_seed=42)
        pipe = Pipeline([('standardize', StandardScaler()),
                         ('classifier', model)])

        parameters = {'classifier__iterations': [500, 1000, 1500, 2000],
                      'classifier__depth': [3, 6, 10],
                      'classifier__l2_leaf_reg': [1],
                      'classifier__border_count': [50, 100, 150, 250]}
        best_clf = GridSearchCV(pipe, param_grid=parameters, scoring='balanced_accuracy', n_jobs=30,
                                cv=3, verbose=10)

        X_train = df_train[band_names].values
        X_test = df_test[band_names].values

        best_clf.fit(X_train, y_train)
        joblib.dump(best_clf, f'./saved_models/subject_prediction_gridsearch.pkl')


        y_test_preds = best_clf.predict(X_test)

        accuracy = balanced_accuracy_score(y_test, y_test_preds)

        # confusion matrix
        cmatrix = confusion_matrix(y_test, y_test_preds)
        df_cm = pd.DataFrame(
            cmatrix, index=le.classes_, columns=le.classes_
        )

        heatmap = sns.heatmap(df_cm, annot=True, fmt='g')
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        # permutation test
        train_indices = np.arange(0, X_train.shape[0])
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        test_indices = np.arange(X_train.shape[0], X.shape[0])
        cv_iterator = [(train_indices, test_indices)]
        score, permutation_score, p_value = permutation_test_score(estimator=best_clf, X=X, y=y, cv=cv_iterator,
                                                                   n_permutations=120, n_jobs=30,
                                                                   scoring='balanced_accuracy',
                                                                   verbose=1)

    #  subject level models
    if analysis_type == 'subject_level_state_prediction':
        df_train = prepare_subject_data(train=True, bands=bands)
        df_test = prepare_subject_data(train=False, bands=bands)
        subjects_paper = [str(i + 1) for i in range(len(subjects))]
        subject_transform_dict = dict(zip(subjects, subjects_paper))

        # phase is the task
        results_df = pd.DataFrame(columns=['Subjects', 'Source', 'Phase', 'Measure', 'Value'])
        source = 'original'

        for subject in subjects:
            # subject = 'ocdbd2'

            subject_train = df_train[df_train.subjects == subject]
            subject_test = df_test[df_test.subjects == subject]

            le = MyLabelEncoder()
            y_train = le.fit(subject_train['state'].values).transform(subject_train['state'].values)
            y_test = le.transform(subject_test['state'].values)

            X_train = subject_train[band_names].values
            X_test = subject_test[band_names].values

            model = CatBoostClassifier(thread_count=2, logging_level='Silent', loss_function='MultiClass',
                                       random_seed=42)
            pipe = Pipeline([('standardize', StandardScaler()),
                             ('classifier', model)])

            parameters = {'classifier__iterations': [100, 200, 500, 1000],
                          'classifier__learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
                          'classifier__depth': [1, 2, 3, 5, 10],
                          'classifier__colsample_bylevel': [0.8, 0.9, 1.0],
                          'classifier__l2_leaf_reg': [0.1, 1, 3, 5, 20],
                          'classifier__border_count': [50, 100, 150, 250]}
            best_clf = GridSearchCV(pipe, param_grid=parameters, scoring='balanced_accuracy', n_jobs=15,
                                    cv=3, verbose=10)

            best_clf.fit(X_train, y_train)
            joblib.dump(best_clf, f'./saved_models/{subject}_gridsearch.pkl')

            y_test_preds = best_clf.predict(X_test)

            accuracy = balanced_accuracy_score(y_test, y_test_preds)
            matrix = confusion_matrix(y_test_preds, y_test)
            accuracy_per_class = matrix.diagonal() / matrix.sum(axis=1)

            train_indices = np.arange(0, X_train.shape[0])
            X = np.concatenate([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            test_indices = np.arange(X_train.shape[0], X.shape[0])
            cv_iterator = [(train_indices, test_indices)]
            score, permutation_score, p_value = permutation_test_score(best_clf.best_estimator_, X, y,
                                                                       cv=cv_iterator,
                                                                       n_permutations=120,
                                                                       n_jobs=30,
                                                                       scoring='balanced_accuracy',
                                                                       verbose=1)
            perm_results = {'score': score,
                            'permutation_score': permutation_score,
                            'p_value': p_value}
            with open(f"./saved_models/perm_scores/{subject}.pkl", "wb") as f:
                pickle.dump(perm_results, f)

            results_df = results_df.append({'Subjects': subject_transform_dict[subject],
                                            'Source': source,
                                            'Phase': 'Total',
                                            'Measure': 'Accuracy',
                                            'Value': accuracy}, ignore_index=True)

            # feature importance
            # prepare data
            X_test_scaled = best_clf.best_estimator_['standardize'].transform(X_test)
            test_pool = catboost.Pool(data=X_test_scaled, label=y_test)
            # SHAP
            shap_values = best_clf.best_estimator_['classifier'].get_feature_importance(data=test_pool,
                                                                                        type='ShapValues')

            for ix, s in enumerate(states):
                shap.summary_plot(shap_values[:, ix, :-1], X_test_scaled, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f'./saved_models/plots/shap/{subject}_{s}.svg')
                plt.close()

            # specificity and sensitivty
            mask = np.ones(le.classes_.shape[0], bool)
            TP = np.diag(matrix)
            TN = np.empty(le.classes_.shape[0])
            FP = np.empty(le.classes_.shape[0])
            FN = np.empty(le.classes_.shape[0])
            sens = np.empty(le.classes_.shape[0])
            spec = np.empty(le.classes_.shape[0])

            for i in range(le.classes_.shape[0]):
                new_mask = mask.copy()
                new_mask[i] = 0
                TN[i] = matrix[new_mask, :][:, new_mask].sum()
                FP[i] = matrix[:, ~new_mask][new_mask].sum()
                FN[i] = matrix[:, new_mask][~new_mask, :].sum()
                sens[i] = TP[i] / (TP[i] + FN[i])
                spec[i] = TN[i] / (TN[i] + FP[i])

                results_df = results_df.append({'Subjects': subject_transform_dict[subject],
                                                'Source': source,
                                                'Phase': le.classes_[i], 'Measure': 'Sensitivity',
                                                'Value': sens[i]}, ignore_index=True)
                results_df = results_df.append({'Subjects': subject_transform_dict[subject], 'Source': source,
                                                'Phase': le.classes_[i], 'Measure': 'Specificity',
                                                'Value': spec[i]}, ignore_index=True)
