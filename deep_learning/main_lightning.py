import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import pathlib
import torch.nn as nn
import gc
import numpy as np
import pandas as pd

from utils.utils import calculate_accuracy, calculate_confusion_matrix, plot_confusion_matrix
from deep_learning.lightning_model import LitModel, LitDataModule


def train_and_test(options, experiment_dir, data):
    if options['transfer_learning']:
        options['learning_rate'] = 0.004
    model = LitModel(num_classes=options['num_classes'], kernel=options['kernel'],
                     num_filters=options['num_filters'], num_blocks=options['blocks'],
                     input_block=options['input_block'], learning_rate=options['learning_rate'])
    if options['transfer_learning']:
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.load_from_checkpoint(str(options['transfer_model']))
        if options["fixed_weights"]:
            for param in model.parameters():
                param.requires_grad = False  # this turns off the updating of weights when backpropagating
        model.fc = nn.Linear(model.fc.in_features, options['num_classes'])

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=experiment_dir, mode='min',
                                          filename='best_model_{epoch:02d}_{val_loss:.2f}', save_last=True)
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=6, mode='min', min_delta=1e-4)
    trainer = pl.Trainer(default_root_dir=experiment_dir, callbacks=[checkpoint_callback, early_stopping_cb],
                         max_epochs=options['num_epochs'], gpus=options['gpu'], gradient_clip_val=1.0)

    trainer.fit(model, datamodule=data)
    best_model = checkpoint_callback.best_model_path
    model = model.load_from_checkpoint(best_model)
    tester = pl.Trainer(default_root_dir=experiment_dir, gpus=options['gpu'])
    results = tester.test(model, datamodule=data)
    tester.evaluation_loop.predictions.to_disk()


def load_predictors(folder):
    predictors = {}
    ix = 0
    for f in sorted(folder.iterdir()):
        if f.is_dir():
            prediction_file = f.joinpath('predictions.pt')
            my_dict = torch.load(prediction_file)
            array = np.vstack([np.array(my_dict[i]['predictions']) for i in range(len(my_dict))])
            predictors[f"iter_{ix}"] = array
            ix += 1
    return predictors


def ensemble_predictions(directory, data, options):

    data.prepare_data()
    data.setup(stage='test')
    true_labels = data.test_set.labels
    predictors = load_predictors(directory)
    n_iter = len(predictors.keys())
    accuracy = {}
    predictor_matrix = np.empty((predictors['iter_0'].shape[0], options['num_classes'], n_iter))
    for key, value in predictors.items():
        accuracy[key] = calculate_accuracy(torch.from_numpy(value), true_labels, binary=options['binary'])
        print(f'accuracy is: {accuracy[key]} for iteration {key}')
        predictor_matrix[..., int(key.split('_')[1])] = value
    predictor_average = np.mean(predictor_matrix, -1)
    np.save(directory.joinpath('ensemble_predictions.npy'), predictor_average)

    ensemble_accuracy = calculate_accuracy(torch.from_numpy(predictor_average), true_labels, binary=options['binary']).item()

    average_accuracy = np.mean([accuracy[k] for k in accuracy.keys()])

    print(f'ensemble accuracy is: {ensemble_accuracy}')
    print(f'average accuracy is: {average_accuracy}')

    c_matrix = calculate_confusion_matrix(torch.Tensor(predictor_average), true_labels, binary=options['binary'])
    fig_cm = plot_confusion_matrix(c_matrix, options)
    pd.DataFrame(c_matrix.numpy()).to_csv(directory.joinpath('ensemble_confusion_matrix.csv'),index=False, header=False)
    fig_cm.savefig(f'{directory}/ensemble_confusion_matrix.png')
    df = pd.DataFrame()
    df.loc['0', 'ensemble_accuracy'] = ensemble_accuracy
    df.loc['0', 'average_accuracy'] = average_accuracy
    df.to_csv(directory.joinpath('ensemble_accuracy.csv'))


def gather_ensembles(experiment_dir):
    results = pd.DataFrame()
    for f in experiment_dir.iterdir():
        if f.is_dir():
            fname = f.joinpath('ensemble_accuracy.csv')
            df = pd.read_csv(fname)
            results.loc[f.name, 'ensemble_accuracy'] = df['ensemble_accuracy'].values
    results.to_csv(experiment_dir.joinpath('results.csv'))


def state_prediction(main_experiment_dir, options):
    for subject in options['subjects']:
        if not subject == options['subject']:
            continue
        options['current_subject'] = subject
        if options['permute']:
            options['permute_it'] = int(main_experiment_dir.name.split('_')[2])
        data = LitDataModule(options, test_subject=subject)
        for it in range(options['iterations']):
            if options['permute']:
                experiment_dir = main_experiment_dir.joinpath(f'{it}')
            else:
                experiment_dir = main_experiment_dir.joinpath(f'{subject}/{it}')
            if not experiment_dir.exists():
                pathlib.Path.mkdir(experiment_dir, parents=True)
            if options['transfer_learning']:
                options['transfer_model'] = next(
                    options['transfer_folder'].joinpath(f'{it}/').glob('best_model*.ckpt'))

            train_and_test(options, experiment_dir, data)
            gc.collect()
            torch.cuda.empty_cache()

        if options['permute']:
            ensemble_predictions(main_experiment_dir, data, options)
        else:
            ensemble_predictions(main_experiment_dir.joinpath(f'{subject}'), data, options)

    if not options['permute']:
        gather_ensembles(main_experiment_dir)


def subject_prediction(main_experiment_dir, options):
    data = LitDataModule(options, test_subject=None)
    for it in range(options['iterations']):
        experiment_dir = main_experiment_dir.joinpath(f'{it}/')
        if not experiment_dir.exists():
            pathlib.Path.mkdir(experiment_dir, parents=True)
        train_and_test(options, experiment_dir, data)
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_predictions(main_experiment_dir, data, options)


def build_null_distribution(dir, truth_dir):
    """Gather accuracies from permutations and builds null distribution"""
    p_values = []
    for d in dir.iterdir():
        if d.is_dir():
            df = pd.read_csv(d.joinpath('ensemble_accuracy.csv'))
            p_values.append(df['ensemble_accuracy'].values[0])

    truth_df = pd.read_csv(truth_dir.joinpath('ensemble_accuracy.csv'))
    p_truth = truth_df['ensemble_accuracy'].values[0]
    p_values.append(p_truth)
    p_values.sort()

    index_truth = np.argwhere(p_values == p_truth)[0][0]
    perm_count = len(p_values) - index_truth

    p_corrected = perm_count / len(p_values)

    np.savetxt(dir.joinpath('null_dist.csv'), p_values)
    np.savetxt(dir.joinpath('p_corrected.txt'), [p_corrected])


def get_options():
    options = {
        "transfer_learning": False,
        "fixed_weights": False,
        "temporal_ensemble": True,
        "num_consecutive": 5, # for temporal ensemble
        "batch_size": 224,
        "sample_length": 3,
        "num_epochs": 1000,
        "task": 'state_prediction_subject',
        "root_path": pathlib.Path.cwd(),
        "iterations": 5, # how many models in ensemble
        "transform": None,
        "blocks": 3,
        "kernel": 60,
        "num_filters": 64,
        "learning_rate": 0.0004,
        "one_side": False,
        "input_block": False,
        "binary": False,
        "permute": True,
        "perm_iter": 120,
        "gpu": [0],
        "subject": 'ocdbd2' # to limit analysis only to a certain subject
    }
    return options


def main():

    options = get_options()

    experiment_title = options['task'] + "_temporal_ensemble_permutations"

    if 'state_prediction' in options['task']:
        options['subjects'] = ['tt', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7', 'ocdbd8', 'ocdbd9',
                               'ocdbd10',
                               'ocdbd11']
        train_func = state_prediction
        if 'group' in options['task']:
            options['learning_rate'] = 0.0002
    else:
        options['subjects'] = ['tt', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7', 'ocdbd8', 'ocdbd9',
                               'ocdbd10',
                               'ocdbd11']
        train_func = subject_prediction
        options['num_classes'] = 11

    if 'state_prediction' in options['task']:
        options['num_classes'] = 4
        options['class_names'] = ['baseline', 'compulsions', 'obsessions', 'relief']
    else:
        options['num_classes'] = 11
        options['class_names'] = [f'Patient {i+1}' for i in range(11)]

    if options['transfer_learning']:
        experiment_title = f'{experiment_title}_transfer_learning'
        options['transfer_folder'] = pathlib.Path('/data/eaxfjord/deep_LFP/experiments/subject_prediction_lightning/')

    main_experiment_dir = options['root_path'].joinpath('experiments', experiment_title)
    if not main_experiment_dir.exists():
        pathlib.Path.mkdir(main_experiment_dir, parents=True)

    if not options['permute']:
        train_func(main_experiment_dir, options)
    else:
        if options['task'] == 'state_prediction_subject':
            main_experiment_dir = main_experiment_dir.joinpath(options['subject'])
            if not main_experiment_dir.exists():
               main_experiment_dir.mkdir()
        already_done = 0
        for d in main_experiment_dir.iterdir():
            if d.is_dir():
                already_done += 1
        for i in range(options['perm_iter']):
            perm_dir = main_experiment_dir.joinpath(f'main_iter_{i+already_done}/')
            if not perm_dir.exists():
                perm_dir.mkdir()
            train_func(perm_dir, options)
        truth_dir = pathlib.Path(f'{os.environ.get("HOME")}/deep_LFP/experiments/temporal_ensemble/{options["num_consecutive"]}/{options["subject"]}')
        # build_null_distribution(main_experiment_dir, truth_dir=truth_dir)


if __name__ == "__main__":
    main()
