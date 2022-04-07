import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import pathlib
import pandas as pd
import numpy as np

from models.inceptiontime import ResidualBlock
from dataset import LFPDataStatesPercentSplit, LFPDataStates, LFPSubjectPrediction
from torch.utils.data import DataLoader
from utils.utils import sample_randomly


class LitModel(LightningModule):
    def __init__(self, num_classes=4, kernel=60, num_blocks=3, num_filters=32, input_block=False, learning_rate=1e-4):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.kernel_sizes = [kernel // (2 ** i) for i in range(3)]
        self.num_filters = num_filters
        self.input_block = input_block

        if input_block:
            self.input_block = nn.Sequential(nn.Conv2d(in_channels=1, kernel_size=[1, 5],
                                                       out_channels=64, bias=False, padding=[0, 2]),
                                             nn.BatchNorm2d(num_features=64))
            self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, bias=False,
                                                      kernel_size=1), nn.BatchNorm2d(1))

        self.residual_blocks = nn.ModuleList([ResidualBlock(input_size=2, kernel_sizes=self.kernel_sizes,
                                                            num_filters=self.num_filters, is_first=True)])
        self.residual_blocks.extend([ResidualBlock(input_size=4 * self.num_filters, kernel_sizes=self.kernel_sizes,
                                                   num_filters=self.num_filters) for i in range(num_blocks - 1)])
        self.fc = nn.Linear(self.num_filters * 4, num_classes)
        self.relu = nn.ReLU(inplace=True)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, my_input):
        if self.input_block:
            my_input = self.relu(self.input_block(my_input[:, None, ...]))
            my_input = self.relu(self.bottleneck(my_input)).squeeze()

        for block in self.residual_blocks:
            my_input = block(my_input)
        features = my_input
        my_input = features.mean(axis=2)
        my_input = self.fc(my_input).squeeze()

        return my_input, features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-4)
        monitor = 'val_loss'
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor}

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['labels']
        output, _ = self(x)
        loss = self.loss(output, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['labels']
        output, _ = self(x)
        val_loss = self.loss(output, y)
        val_acc = self.accuracy(output, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss, val_acc

    def validation_epoch_end(self, outputs):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['data'], batch['labels']
        output, _ = self(x)
        test_loss = self.loss(output, y)
        test_acc = self.accuracy(output, y)
        self.log('test_loss', test_loss, sync_dist=True)
        self.log('test_acc', test_acc, sync_dist=True)

        return test_loss, test_acc, output

    def test_epoch_end(self, outputs):
        all_predictions = torch.cat([outputs[i][2] for i in range(len(outputs))], 0)
        path = pathlib.Path(self.trainer.default_root_dir, 'predictions.pt')
        self.write_prediction('predictions', all_predictions, filename=str(path))


class LitDataModule(pl.LightningDataModule):
    def __init__(self, options, test_subject=None):
        super(LitDataModule, self).__init__()
        self.batch_size = options['batch_size']
        self.options = options

        if self.options['task'] == 'state_prediction_group':
            self.test_subject = test_subject

    def prepare_data(self):
        pass

    def get_samples(self, dataframe, train=True):
        train_samples = []
        train_labels = []
        for state in self.options['class_names']:
            label_train = self.options['class_names'].index(state)
            index_train = (dataframe['state'] == state) & (dataframe['subjects'] == self.options['current_subject'])
            ts_right = np.concatenate(dataframe[index_train].resampled_right.values)
            ts_left = np.concatenate(dataframe[index_train].resampled_left.values)
            ts_train = np.array([ts_left, ts_right])
            if train:
                num_samples_train = (180 * self.options['sample_length']) // self.options['num_consecutive']
            else:
                num_samples_train = 180//(self.options['num_consecutive'] * self.options['sample_length'])
            samples = sample_randomly(ts_train, num_samples_train,
                                      sample_length=self.options['sample_length']*self.options['num_consecutive'],
                                      sampling_frequency=211, samples_split=True)
            train_labels.append([label_train] * num_samples_train * self.options['num_consecutive'])
            train_samples.append(samples.reshape(-1, 2, self.options['sample_length'] * 211))
        train_matrix = np.concatenate(train_samples)
        train_label_matrix = np.concatenate(train_labels)
        return train_matrix, train_label_matrix

    def setup(self, stage=None):
        if 'state_prediction' in self.options['task']:
            if self.options['temporal_ensemble']:
                train_df = pd.read_pickle(self.options['root_path'].joinpath('data', 'states',
                                                              f'training_data_states_42.pkl'))
                test_df = pd.read_pickle(self.options['root_path'].joinpath('data', 'states',
                                                                            f'test_data_states_42.pkl'))
                train_samples, train_labels = self.get_samples(train_df, train=True)
                test_samples, test_labels = self.get_samples(test_df, train=False)
                self.train_matrix = {'samples': train_samples, 'labels': train_labels}
                self.test_matrix = {'samples': test_samples, 'labels': test_labels}

            else:
                self.train_matrix = self.options["root_path"].joinpath('data', 'states',
                                                                   f'training_samples11_states_42_{self.options["sample_length"]}sec.pkl')
                self.test_matrix = self.options["root_path"].joinpath('data', 'states',
                                                                  f'test_samples11_states_42_{self.options["sample_length"]}sec.pkl')
            if 'group' in self.options['task']:
                self.combined_data = self.combine_data([self.train_matrix, self.test_matrix])
        elif self.options['task'] == 'subject_prediction':
            self.train_matrix = self.options['root_path'].joinpath('data', 'resting_state/'
                                                                   f'training_sample11_random_resting_state_'
                                                                   f'{self.options["sample_length"]}_sec_resampled.pkl')
            self.test_matrix = self.options['root_path'].joinpath('data', 'resting_state/'
                                                                 f'test_samples11_random_resting_state_'
                                                                 f'{self.options["sample_length"]}sec_resampled.pkl')
        if stage == 'fit' or stage is None:
            if self.options['task'] == 'state_prediction_subject':
                self.train_set = LFPDataStatesPercentSplit(data_file=self.train_matrix, split='train', standardize=True,
                                                           transform=self.options['transform'],
                                                           two_class=self.options['two_class'],
                                                           augment=self.options['augment'],
                                                           aug_type=self.options['augment_type'],
                                                           subject=self.options['current_subject'],
                                                           one_side=self.options['one_side'],
                                                           permute=self.options['permute'],
                                                           permute_it=self.options['permute_it'])
                self.val_set = LFPDataStatesPercentSplit(data_file=self.train_matrix, split='valid', standardize=True,
                                                         transform=self.options['transform'],
                                                         two_class=self.options['two_class'],
                                                         subject=self.options['current_subject'],
                                                         one_side=self.options['one_side'],
                                                         permute=self.options['permute'],
                                                         permute_it=self.options['permute_it'])
            elif self.options['task'] == 'state_prediction_group':
                self.train_set = LFPDataStates(data=self.combined_data, split='train', standardize=True,
                                               test_subject=self.test_subject)
                self.val_set = LFPDataStates(data=self.combined_data, split='valid', standardize=True,
                                             test_subject=self.test_subject)
            elif self.options['task'] == 'subject_prediction':
                self.train_set = LFPSubjectPrediction(data_file=self.train_matrix, split='train', standardize=True,
                                                      augment=self.options['augment'], permute=self.options['permute'],
                                                      aug_type=self.options['augment_type'])
                self.val_set = LFPSubjectPrediction(data_file=self.train_matrix, split='valid', standardize=True,
                                                    permute=self.options['permute'])

        if stage == 'test' or stage is None:
            if self.options['task'] == 'state_prediction_subject':
                self.test_set = LFPDataStatesPercentSplit(data_file=self.test_matrix, split='test', standardize=True,
                                                          transform=self.options['transform'],
                                                          two_class=self.options['two_class'],
                                                          subject=self.options['current_subject'],
                                                          one_side=self.options['one_side'],
                                                          permute=self.options['permute'])
            elif self.options['task'] == 'state_prediction_group':
                self.test_set = LFPDataStates(data=self.combined_data, split='test', standardize=True,
                                              test_subject=self.test_subject)
            elif self.options['task'] == 'subject_prediction':
                self.test_set = LFPSubjectPrediction(data_file=self.test_matrix, split='test', standardize=True,
                                                     permute=self.options['permute'])

    def combine_data(self, data_files):
        """
        Combine training and test data files for subject wise split
        :param data_files:
        :return:
        """
        dfs = []
        arrays = []
        for file in data_files:
            data_frame = pd.read_pickle(file)
            array = np.load(file.with_suffix('.npy'))
            dfs.append(data_frame)
            arrays.append(array)
        combined_df = pd.concat(dfs, axis=0)
        combined_arrays = np.concatenate(arrays, 0)
        all_data = {}
        all_data['dataframe'] = combined_df
        all_data['array'] = combined_arrays
        return all_data

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                          num_workers=10)
