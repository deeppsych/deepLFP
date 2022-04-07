import csv
import pandas as pd
from sklearn import metrics
import torch
import numpy as np
import nitime.algorithms as tsa
import shutil
import warnings
import visdom
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools as it
import seaborn as sns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class EarlyStopping():
    def __init__(self):
        self.nsteps_similar_loss = 0
        self.best_loss = 9999.0
        self.delta_loss = 1e-3

    def _increment_step(self):
        self.nsteps_similar_loss += 1

    def _reset(self):
        self.nsteps_similar_loss = 0

    def eval_loss(self, loss):
        if (self.best_loss - loss) <= self.delta_loss:
            self._increment_step()
        else:
            self._reset()
            self.best_loss = loss

    def get_nsteps(self):
        return self.nsteps_similar_loss


def generate_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`
    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def plot_visdom(meterlogger, log_dir):
    """recreate visdom plots using matplotlib and save as png in log folder"""

    viz = visdom.Visdom(port=8097)
    window_json = viz.get_window_data(env=meterlogger.env)
    window_data = json.loads(window_json)

    for plot in window_data.keys():
        if plot in ['loss', 'accuracy']:
            plot_content = window_data[plot]['content']['data']
            training_data = plot_content[1]
            test_data = plot_content[2]

            x_train, y_train = training_data['x'], training_data['y']
            x_test, y_test = test_data['x'], test_data['y']

            plt.ioff()
            plt.figure()
            plt.plot(x_train, y_train, label=training_data['name'])
            plt.plot(x_test, y_test, label=test_data['name'])
            plt.legend()
            plt.title(plot)
            plt.savefig(f'{log_dir}/{plot}.png')
            plt.close()
        elif 'confusion' in plot:
            plot_content = window_data[plot]['content']['data']
            fig, ax = plt.subplots()
            im = ax.imshow(plot_content[0]['z'], origin='lower')
            ax.set_xticks(plot_content[0]['x'])
            ax.set_yticks(plot_content[0]['y'])
            ax.set_xlabel('predicted classes')
            ax.set_ylabel('true classes')
            ax.set_title(plot)
            fig.colorbar(im)
            fig.savefig(f'{log_dir}/{plot}.png')
            plt.close()
        else:
            warnings.warn(f'Unknown plot type: {plot}')


def sample_randomly(timeseries, n_samples, sample_length, sampling_frequency, random=None,
                    samples_split=False):
    """
    Takes in a timeseries and draws n_samples samples of a specific length and returns them

    :param timeseries:
    :param n_samples:
    :param sample_length
    :param sampling_frequency
    :return:
    """
    sample_length_samples = sample_length * sampling_frequency

    if random is None:
        np.random.seed(42)
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

            samples[i, ...] = (current_sample - np.repeat(means[..., np.newaxis], sample_length_samples, axis=1)) / (
                                np.repeat(stds[..., np.newaxis], sample_length_samples, axis=1))
    return samples


def calculate_confusion_matrix(logits, labels, binary=False):

    if binary:
        num_classes = 2
        pred = (logits > 0.5).float()
    else:
        num_classes = logits.shape[1]
        pred = torch.argmax(logits, 1)

    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(labels.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1  # rows for true labels, columns for predictions

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, options, figsize=(8, 5), fontsize=12):
    df_cm = pd.DataFrame(
        confusion_matrix.numpy(), index=options['class_names'], columns=options['class_names'],
    )
    plt.ioff()
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='g')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.tight_layout()
    plt.close()
    plt.ion()
    return fig


def plot_from_log(log_file):
    """plot data from pickled log file"""
    df = pd.read_csv(log_file, delimiter='\t')

    epochs = df.epoch.values
    loss = df.loss.values
    acc = df.acc.values

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(epochs, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('acc', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, acc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def calculate_accuracy(outputs, targets, binary=False, ensemble=False):
    batch_size = targets.size(0)

    if binary:
        pred = (outputs.detach() > 0.5).float()
        n_correct_elems = (pred == targets).float().sum()
    else:
        if ensemble:
            _, pred = outputs.topk(1, 1, True)
            pred = pred.reshape((pred.shape[0]//5, 5))
            pred, _ = torch.mode(pred, 1)
            targets = targets.reshape((targets.shape[0]//5, 5))
            targets, _ = torch.mode(targets, 1)
            correct = pred.eq(targets)
            batch_size = batch_size // 5
        else:
            _, pred = outputs.topk(1, 1, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1))
            n_correct_elems = correct.float().sum()

    return n_correct_elems / batch_size


def calc_multiclass_metrics(outputs, targets):

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    _, predicted = torch.max(outputs, 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        accuracy_score = metrics.accuracy_score(targets, predicted)
        precision = metrics.precision_score(targets, predicted, average='macro', labels=labels)
        recall = metrics.recall_score(targets, predicted, average='macro', labels=labels)
        f1_score = metrics.f1_score(targets, predicted, average='macro', labels=labels)
        fbeta = metrics.fbeta_score(targets, predicted, average='macro', beta=0.5, labels=labels)

    return accuracy_score, precision, recall, f1_score, fbeta


def plot_frequency_spectrum(matrix):
    sampling_freq = 422

    for i in range(10):
        # rows = np.where(training_matrix[:, 0] == 8.0)
        rand_row = np.random.randint(0, matrix.shape[0])
        print(rand_row)
        timeseries = matrix[rand_row, 1:,1:]
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(timeseries)
        f, fx, _ = tsa.spectral.multi_taper_psd(timeseries, Fs=sampling_freq)
        ax2.plot(f, fx)
        plt.show()


def save_checkpoint(title, state, model_type='best'):
    dst = f'{str(title)}_model_{model_type}.pth.tar'
    torch.save(state, dst)


def load_checkpoint(title, model_type='best'):
    filename = f'{str(title)}_model_{model_type}.pth.tar'
    dict = torch.load(filename)
    return dict


def plot_kernel(model):
    kernel = model.conv1.weight.data.clone()

    nrows=8
    ncols=8
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                           top= 1.-0.5/(nrows+1), bottom=0.5/(nrows+1),
                           left=0.5/(ncols+1), right=1.-0.5/(ncols+1))
    ix = 0
    for i,j in it.product(range(nrows),range(ncols)):
        ax = plt.subplot(gs[i, j])
        ax.plot(kernel[ix, ...].cpu().numpy().T)
        ix += 1