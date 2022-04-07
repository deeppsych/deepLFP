import os
import itertools

import pandas as pd
import seaborn as sns
from natsort import natsorted
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pingouin as pg
import scipy


if __name__ == "__main__":

    # path to raw data with VAS scores
    vas_fname = f'{os.environ.get("HOME")}/deep_LFP/info/Data Visit 1 VAS v1.0.xlsx'

    vas_df = pd.read_excel(vas_fname, sheet_name=1, engine='openpyxl')
    vas_df = vas_df.drop('Report Parent',axis=1)
    df = pd.melt(vas_df, id_vars=['Record Id', 'Conditie_VAS', 'Ronde_VAS'], var_name='VAS_score')
    df.columns = ['Subject', 'Time', 'Rounds', 'Emotion', 'Value']
    df = df.set_index(['Subject', 'Rounds', 'Time', 'Emotion'])
    # df = df.groupby(level=['Subject', 'Emotion', 'Rounds']).sum() # compute area under the curve
    df = df.groupby(level=['Subject', 'Emotion', 'Time']).mean() # average across rounds
    df.reset_index(inplace=True)
    emotions = df.Emotion.unique()

    # compute anova
    results = pg.rm_anova(data=df, dv='Value', subject='Subject', within=['Emotion', 'Time'],
                          detailed=True)

    # post hoc one way anovas for each symptom
    all_results = {}
    emotions = df.Emotion.unique()
    for em in emotions:
        temp_df = df[df.Emotion == em]
        temp_results = pg.rm_anova(data=temp_df, dv='Value', subject='Subject',
                                   within='Time', detailed=True)
        all_results[em] = temp_results
    [print(key, all_results[key].loc[0, 'p-GG-corr']) for key in all_results.keys()]

    #  Figure from from paper
    plt.style.use('seaborn')
    mean_vas = vas_df.groupby(by='Conditie_VAS').mean().loc[:, 'VAS_angst':]
    mean_vas.columns = ['Anxiety', 'Agitation', 'Mood', 'Obsessions', 'Compulsions', 'Avoidance']

    fig, ax = plt.subplots()
    mean_vas.plot(ax=ax)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5, 6]))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))

    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(['Baseline', 'Obsessions \n induced', 'Obsessions', 'Compulsions',
                                                        'Compulsions \n until relief', 'Relief']))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    plt.xlabel('')
    plt.ylabel('VAS score')
    plt.title('Visual Analog Scales during the experiment')
    plt.tight_layout()