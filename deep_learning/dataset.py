from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import periodogram
import pandas as pd


class LFPSubjectPrediction(Dataset):
    """Dataset class for subject prediction (fingerprinting) where the test set is a separate session"""
    def __init__(self, data_file, split='train', standardize=None, augment=False, aug_type=None, permute=False):
        data_frame = pd.read_pickle(data_file)
        array = np.load(data_file.with_suffix('.npy'))

        # sort data so class labels are in same orders as OCDBD1 to OCDBD 11
        data_frame.loc[data_frame.subjects == 'tt', 'subjects'] = 'ocdbd1' # change tt into OCDBD1
        classes = ['ocdbd1', 'ocdbd2', 'ocdbd3', 'ocdbd4', 'ocdbd5', 'ocdbd6', 'ocdbd7', 'ocdbd8', 'ocdbd9', 'ocdbd10',
                   'ocdbd11']
        data_frame['subjects'] = pd.Categorical(data_frame['subjects'], classes)
        data_frame = data_frame.sort_values(by='subjects')

        self.classes = classes
        y = pd.factorize(data_frame.subjects)[0]
        if permute:
            y = np.random.permutation(y)

        X = array
        if split == 'train' or split == 'valid':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            if split == 'train':
                data = X_train
                labels = y_train

            if split == 'valid':
                data = X_val
                labels = y_val

        elif split == 'test':
            data = X
            labels = y

        if standardize:
            n_samples = data.shape[0]
            for i_sample in range(n_samples):
                data_sample = data[i_sample]
                mean_sample = data_sample.mean(axis=1, keepdims=True)
                std_sample = data_sample.std(axis=1, keepdims=True)
                if np.any(std_sample == 0) or np.any(np.isnan(mean_sample)) or np.any(np.isnan(std_sample)):
                    assert (std_sample == 0)
                data[i_sample] = (data_sample - mean_sample) / std_sample

        self.data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = {'data': self.data[item], 'labels': self.labels[item]}
        return sample


class LFPDataStates(Dataset):
    """This class uses subject wise split.  One subject for test, one for validation and the rest for training.
    Used for the group model
    """

    def __init__(self, data=None, split='train', standardize=None, test_subject=None):

        df = data['dataframe']
        array = data['array']

        subject_id = df.subjects
        subjects = np.unique(subject_id)

        # pick a random subject for use as test subject
        if test_subject:
            self.test_subject = test_subject
            test_subject_idx = np.argwhere(subjects==test_subject)[0][0]
            random_generator = np.random.RandomState(test_subject_idx)
        elif not test_subject:
            random_generator = np.random.RandomState(42)
            test_subject_idx = random_generator.randint(0, len(subjects))
            self.test_subject = subjects[test_subject_idx]
        training_subjects = [x for x in subjects if x != self.test_subject]

        # pick a random validation subject
        val_idx = random_generator.randint(0, len(training_subjects))
        self.val_subject = subjects[val_idx]
        self.training_subjects = [x for x in training_subjects if x != self.val_subject]

        if split == 'train':
            train_indices = (df.subjects.isin(self.training_subjects))
            data = array[train_indices]
            labels = pd.factorize(df.state[train_indices])[0]
        if split == 'valid':
            val_indices = (df.subjects.isin([self.val_subject]))
            data = array[val_indices]
            labels = pd.factorize(df.state[val_indices])[0]
        if split == 'test':
            test_indices = (df.subjects.isin([self.test_subject]))
            data = array[test_indices]
            labels = pd.factorize(df.state[test_indices])[0]

        if standardize:
            n_samples = data.shape[0]
            for i_sample in range(n_samples):
                data_sample = data[i_sample]
                mean_sample = data_sample.mean(axis=1, keepdims=True)
                std_sample = data_sample.std(axis=1, keepdims=True)
                if np.any(std_sample == 0) or np.any(np.isnan(mean_sample)) or np.any(np.isnan(std_sample)):
                    assert(std_sample == 0)
                data[i_sample] = (data_sample - mean_sample) / std_sample

        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = {'data': self.data[item], 'labels': self.labels[item]}
        return sample


class LFPDataStatesPercentSplit(Dataset):
    """Here use a 80/20 % split for training/validation and use last round as test,
    Used for subject-wise state prediction models"""

    def __init__(self, data_file, split, subject=None, channel=None, transform=None, standardize=False,
                 two_class=False, augment=False, aug_type=None, fs=422, one_side=False, permute=False,
                 permute_it=0):

        if type(data_file) is dict:
            array = data_file['samples']
            labels = data_file['labels']
        else:
            data_frame = pd.read_pickle(data_file)
            array = np.load(data_file.with_suffix('.npy'))

        if subject and channel:
            index = (data_frame.subjects == subject) & (data_frame.channel_pair == channel)
            data_frame = data_frame[index]
            array = array[index.values]
        elif subject:
            if type(data_file) is not dict:
                index = (data_frame.subjects == subject)
                data_frame = data_frame[index]
                array = array[index.values]

        self.states = ['baseline', 'compulsions', 'obsessions', 'relief']
        if type(data_file) is not dict:
            data_frame.state = pd.Categorical(data_frame.state, categories=self.states)
        if type(data_file) is not dict:
            y = pd.factorize(data_frame.state)[0]
        else:
            y = labels
        if permute:
            y = np.random.RandomState(seed=permute_it).permutation(y)
        X = array
        self.fs = fs
        self.transform = transform

        if split == 'train' or split == 'valid':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                              random_state=42)
            if split == 'train':
                data = X_train
                labels = y_train

            if split == 'valid':
                data = X_val
                labels = y_val

        elif split == 'test':
                data = X
                labels = y

        if transform is not None:
            standardize = False
        if standardize:
            n_samples = data.shape[0]
            for i_sample in range(n_samples):
                data_sample = data[i_sample]
                mean_sample = data_sample.mean(axis=1, keepdims=True)
                std_sample = data_sample.std(axis=1, keepdims=True)
                if np.any(std_sample == 0) or np.any(np.isnan(mean_sample)) or np.any(np.isnan(std_sample)):
                    assert(std_sample == 0)
                data[i_sample] = (data_sample - mean_sample) / std_sample

        self.data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = {'data': self.data[item], 'labels': self.labels[item]}
        return sample




