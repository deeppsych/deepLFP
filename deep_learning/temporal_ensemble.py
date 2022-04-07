import numpy as np
import os
import pathlib
import torch
import pandas as pd
import pytorch_lightning as pl
import scipy.stats as stats

from models.inceptiontime import InceptionTime
from deep_learning.lightning_model import LitModel
from lightning_model_filter import LitDataModuleFiltered
from deep_learning.main_lightning import ensemble_predictions, get_options
from filter_ablation import extract_filtered_data


def sample_non_overlapping(timeseries):
    samples = []
    ix = 0
    while ix < timeseries.shape[1] - 633:
        samples.append(timeseries[:, ix:ix+633])
        ix += 633

    samples = np.array(samples)
    means = samples.mean(axis=2)
    stds = samples.std(axis=2)

    samples = (samples - np.repeat(means[..., np.newaxis], 633, axis=2)) / (
        np.repeat(stds[..., np.newaxis], 633, axis=2))

    return samples


def sample_randomly(timeseries, n_samples, sample_length, sampling_frequency, random=None, samples_split=True):
    """
    Takes in a timeseries and draws n_samples samples of a specific length and returns them

    :param timeseries:
    :param n_samples:
    :param sample_length
    :param sampling_frequency
    :param random               random seed
    :param samples_split        if sample should be split up in 3 second chunks for temporal ensembling
    :return:
    """
    sample_length_samples = sample_length * sampling_frequency

    if random is None:
        random_indices = np.random.randint(0, timeseries.shape[1] - sample_length_samples, n_samples)
    else:
        random_indices =random

    if samples_split:
        n_consecutive = sample_length // 3
        samples = np.empty((n_samples, n_consecutive, timeseries.shape[0], sample_length_samples // n_consecutive))
    else:
        samples = np.empty((n_samples, timeseries.shape[0], sample_length_samples))

    for i, ix in enumerate(random_indices):
        current_sample = timeseries[:, ix:ix+sample_length_samples]

        if samples_split:
            split_samples = np.array(np.split(current_sample, n_consecutive, axis=1))
            means = split_samples.mean(axis=2)
            stds = split_samples.std(axis=2)

            samples[i, ...] = (split_samples - np.repeat(means[..., np.newaxis], 633, axis=2)) / (
                np.repeat(stds[..., np.newaxis], 633, axis=2))


        else:
            means = current_sample.mean(axis=1)
            stds = current_sample.std(axis=1)

            samples[i, ...] = (current_sample - np.repeat(means[..., np.newaxis], 633, axis=1)) / (
                                np.repeat(stds[..., np.newaxis], 633, axis=1))
    return samples


def load_models(model_folder, n_classes):

    all_models = []
    for f in model_folder.iterdir():
        if f.is_dir():
            model = InceptionTime(num_classes=n_classes, num_blocks=3, num_filters=64, kernel=60, input_block=False)
            # load previous weights
            model_file = list(f.glob('*model_best.pth.tar'))[0]
            state_dict = torch.load(model_file)['state_dict']
            model.load_state_dict(state_dict)
            all_models.append(model)
            del model
    return all_models


if __name__=="__main__":
    """
    Here I look at accuracy if I take 3 or 5 consecutive samples and use majority voting"""

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    n_classes = 4
    subjects = ['tt', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7', 'ocdbd8', 'ocdbd9', 'ocdbd10',
                'ocdbd11']
    subjects_paper = [str(i+1) for i in range(len(subjects))]
    subject_transform_dict  = dict(zip(subjects, subjects_paper))
    my_index = pd.MultiIndex(names=['subject', 'num_consecutive'], levels=[subjects, ['1','3', '5']],
                             codes=[[], []])
    accuracy = pd.DataFrame(index=my_index)
    num_consecutive_samples = 5
    test_df = pd.read_pickle(f'{os.getenv("HOME")}/deep_LFP/data/states/test_data_states_42.pkl')

    acc = pd.DataFrame(columns=['Subjects', 'Source', 'Accuracy', 'Baseline', 'Obsessions', 'Compulsions', 'Relief'])
    acc.loc[:, 'Subjects'] = subjects
    acc.loc[:, 'Source'] = 'Temporal ensemble'

    results_df = pd.DataFrame(columns=['Subjects', 'Source', 'Phase', 'Measure', 'Value'])
    for sub in subjects:

        test_data, test_labels = extract_filtered_data(test_df, train=False, subject=sub, sample_length=3*num_consecutive_samples,
                                                       samples_split=True)
        model_dir = pathlib.Path(f'{os.getenv("HOME")}/deep_LFP/experiments/state_prediction_subject_lightning/{sub}/')
        options = get_options()
        options['current_subject'] = sub
        current_data, current_labels = None, None
        current_test_data, current_test_labels = test_data['original'].reshape((-1, 2, 633)), test_labels['original']
        lightning_data_fil = LitDataModuleFiltered(current_data, current_labels, current_test_data, current_test_labels,
                                                   batch_size=5)

        test_acc = []
        it = 0
        main_experiment_dir = pathlib.Path.cwd().joinpath('experiments', 'temporal_ensemble', f'{num_consecutive_samples}', sub)
        for f in model_dir.iterdir():
            if f.is_dir():
                experiment_dir = main_experiment_dir.joinpath(str(it))
                best_model = list(f.glob('best_model*.ckpt'))[0]
                model = LitModel(num_classes=4, kernel=60, num_filters=64, num_blocks=3,
                                 input_block=False, learning_rate=options['learning_rate'])
                model = model.load_from_checkpoint(str(best_model))
                tester = pl.Trainer(default_root_dir=str(experiment_dir), gpus=[0])
                results = tester.test(model, datamodule=lightning_data_fil)
                test_acc.append(results[0]['test_acc'])
                it += 1

        options = {}
        options['binary'] = False
        options['num_classes'] = 4
        options['class_names'] = ['baseline', 'compulsions', 'obsessions', 'relief']
        ensemble_predictions(main_experiment_dir, lightning_data_fil, options)

        ensemble_preds = np.load(main_experiment_dir.joinpath(('ensemble_predictions.npy')))
        predicted_classes = np.argmax(ensemble_preds, axis=1)
        mode, _ = stats.mode(predicted_classes.reshape(48, 5), axis=1)
        mode = mode.squeeze()
        true_labels = test_labels['original'].reshape(48, 5)[:,0]


        # confusion matrix
        confusion_matrix = np.zeros((options['num_classes'], options['num_classes']))
        for true, pred in zip(true_labels, mode):
            confusion_matrix[true, pred] += 1

        # specificity and sensitivty
        mask = np.ones(options['num_classes'], bool)
        TP = np.diag(confusion_matrix)
        TN = np.empty(options['num_classes'])
        FP = np.empty(options['num_classes'])
        FN = np.empty(options['num_classes'])
        sens = np.empty(options['num_classes'])
        spec = np.empty(options['num_classes'])

        for i in range(options['num_classes']):
            new_mask = mask.copy()
            new_mask[i] = 0
            TN[i] = confusion_matrix[new_mask, :][:, new_mask].sum()
            FP[i] = confusion_matrix[:, ~new_mask][new_mask].sum()
            FN[i] = confusion_matrix[:, new_mask][~new_mask, :].sum()
            sens[i] = TP[i] / (TP[i] + FN[i])
            spec[i] = TN[i] / (TN[i] + FP[i])

            results_df = results_df.append({'Subjects': subject_transform_dict[sub], 'Source': 'Temporal ensemble',
                                            'Phase': options['class_names'][i], 'Measure': 'Sensitivity',
                                            'Value': sens[i]}, ignore_index=True)
            results_df = results_df.append({'Subjects': subject_transform_dict[sub], 'Source': 'Temporal ensemble',
                                            'Phase': options['class_names'][i], 'Measure': 'Specificity',
                                            'Value': spec[i]}, ignore_index=True)

        ix_sub = acc.Subjects == sub
        accuracy = (mode == true_labels).sum() / mode.shape[0]
        acc.loc[ix_sub, 'Accuracy'] = accuracy
        results_df = results_df.append({'Subjects': subject_transform_dict[sub], 'Source': 'Temporal ensemble',
                                        'Phase': 'Total', 'Measure': 'Accuracy', 'Value': accuracy}, ignore_index=True)
        for ix, band in enumerate(['Baseline', 'Compulsions', 'Obsessions', 'Relief']):
            index = true_labels == ix
            acc.loc[ix_sub, band] = (true_labels[index] == mode[index]).sum() / mode[index].shape[0]  # sensitivity

    if save:
        results_df.to_csv(main_experiment_dir.parent.joinpath('results_df.csv'))


