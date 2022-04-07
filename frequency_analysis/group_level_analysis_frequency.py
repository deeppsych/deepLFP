import pathlib
import random
import os

import numpy as np
import pandas as pd
import mne
from scipy.integrate import simps
from sklearn.model_selection import train_test_split

from utils.load_data_hdf5 import load_dict_from_hdf5


def flatten_dict(nested_dict):
    """
    Flattends the nested structure in the nested dictionary

    Args:
        nested_dict:

    Returns:

    """
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    """
    Convert nested dictionary to a dataframe

    Args:
        values_dict:

    Returns:

    """
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.columns = ['data']
    return df


def preprocess_data(data, sf=422):
    """
    Bandpass filters the data from 3-99 HZ and bandstops from 47-53Hz

    Args:
        data:
        sf:         Sampling frequency

    Returns:

    """
    h = mne.filter.create_filter(data, sfreq=sf, l_freq=3, h_freq=99,
                                 method='fir',
                                 phase='zero', verbose=False, h_trans_bandwidth=1)
    h_notch = mne.filter.create_filter(data, sfreq=sf, l_freq=53,
                                       h_freq=47, method='fir', phase='zero', verbose=False,
                                       h_trans_bandwidth=1, l_trans_bandwidth=1)

    processed_data = np.convolve(data, h, 'same')
    processed_data = np.convolve(processed_data, h_notch, 'same')
    return processed_data


def power_spectrum(data, sf=422, multitaper=True, aggregrate='mean',
                   sample_length=3):
    """
    Computes the power spectrum of the data

    Args:
        data:
        sf:             sampling frequency
        multitaper:     Whether to use multitaper or welch
        aggregrate:     How to aggregate
        sample_length:  Length of each sample in seconds

    Returns:

    """
    if multitaper:
        psd, freqs = mne.time_frequency.psd_array_multitaper(data, sf, adaptive=True, normalization='full',
                                                             verbose=0, fmin=0.5, fmax=100)
    else:
        psd, freqs = mne.time_frequency.psd_array_welch(data, sf, fmin=0.5, fmax=100, n_fft=sample_length * sf,
                                                        verbose=0,
                                                        average=aggregrate)
    return freqs, psd


def bandpower(band, psd, freqs, relative=False):
    """
    Computes the power

    Args:
        band:       Which band to compute power in [low, high]
        psd:        The power spectral density
        freqs:      Frequencies, x-axis of power spectral density
        relative:   Whether to calculate relative power

    Returns: Bandpower

    """
    low, high = band
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    if len(psd.shape) > 1:
        bp = np.array([simps(psd[idx_band, i], dx=freq_res) for i in range(psd.shape[1])])
    else:
        bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def calculate_power(df, bands, **kwargs):
    """
    Calculates power per band for each subject

    Args:
        df:         Dataframe with my data
        bands:      information about the bands, ie. what frequencies each band is in
        **kwargs:

    Returns:

    """
    new_data_df = pd.DataFrame(columns=['subjects', 'state', 'rounds', 'channel', 'side', 'band', 'bandpower'])
    ix = 0
    for index, row in df.iterrows():
        data = row.values[0]
        if data.size == 0:
            continue
        left_data = data[:, 0]
        right_data = data[:, 1]
        index = list(index)
        names_dict = {k: v for k, v in zip(['subjects', 'state', 'rounds', 'channel'], index)}
        for data, name in zip([left_data, right_data], ['left', 'right']):
            processed = preprocess_data(data)
            f, p = power_spectrum(processed, **kwargs)
            names_dict['side'] = name
            for band in bands:
                bp = bandpower(bands[band], p, f, relative=False)
                names_dict['band'] = band
                names_dict['bandpower'] = bp
                df1 = pd.DataFrame(names_dict, index=list(range(ix, ix + bp.shape[0])))
                new_data_df = new_data_df.append(df1)
                ix = ix + bp.shape[0]
    return new_data_df


def preprocess_df(df):
    """
    Modifies dataframe to be in right format for analysis

    Args:
        df:

    Returns:

    """
    df['sample_id'] = df.groupby(['subjects', 'state', 'rounds', 'channel', 'side', 'band']).cumcount()
    df = df.pivot(index=['subjects', 'state', 'rounds', 'channel', 'side', 'sample_id'], columns='band',
                  values='bandpower')
    df = df.reset_index()
    return df


if __name__ == '__main__':
    main_effect_analysis = True
    data_dictionary = f'{os.environ.get("HOME")}/deep_LFP/data/nested_dictionary_11subjects22112020.hdf5'
    data_dictionary = load_dict_from_hdf5(data_dictionary)

    visit1_dict = {}
    resting_dict = {}
    all_data_total = []
    # extract first visit for each subject
    for subject in list(data_dictionary.keys()):
        if 'visit_1' in data_dictionary[subject].keys():
            visit1_dict[subject] = data_dictionary[subject]['visit_1']

    # calculate power in prespecified bands
    bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'low_gamma': [30, 50],
             'high_gamma': [50, 100]}

    data_df = nested_dict_to_df(visit1_dict)

    # remove visits with to little data
    sizes = np.array([d.shape[0] for d in data_df['data'].values])
    ix = np.where(sizes < 5000)[0]
    data_df = data_df.drop(data_df.index[ix])

    data_df.index = data_df.index.set_names(['subjects', 'states', 'rounds', 'channels'])

    # calculates power
    power = calculate_power(data_df, bands, multitaper=False, aggregrate=None, sample_length=3)

    if main_effect_analysis:
        # average across channels
        df = power.groupby(list(power.columns[power.columns != 'channel'][:-1]), as_index=False)['bandpower'].mean()

        df.to_csv('bandpower_with_baseline.csv')
        # run a statistical model (anova) on the results to see if power is changing significanlty compared to baseline
        # I did that in R.
