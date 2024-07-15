""" Contains the functions needed for analyzing learning curves. """

import math
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_surprisals(words:List[str], surprisals_df, neg_samples=False, first_step=True):
    num_words = len(words)
    cols = min(num_words, 3)
    rows = math.ceil(num_words / cols)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axs = np.atleast_2d(axs)

    correlations = {}

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        ax = axs[i//cols, i%cols]
        if neg_samples:
            ax.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o', label='Positive Samples')
            ax.plot(word_data['Steps'], word_data['MeanNegSurprisal'], marker='o', label='Negative Samples')

            # Calculate correlation
            if not first_step:
                word_data = word_data[word_data['Steps'] != 0]
                
            corr, _ = pearsonr(word_data['MeanSurprisal'], word_data['MeanNegSurprisal'])
            correlations[word] = corr
            ax.set_title(f'"{word}"', pad=18)
            ax.text(0.5, 1.02, f'Correlation: {corr:.2f}', fontsize=10, ha='center', transform=ax.transAxes)

        else:
            ax.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o')
            ax.set_title(f'"{word}"')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean surprisal')
        ax.invert_yaxis()

    # Remove empty subplots
    for j in range(i+1, rows*cols):
        fig.delaxes(axs.flatten()[j])
    
    plt.tight_layout()

    # Legend
    last_ax = axs.flatten()[i]
    handles, labels = last_ax.get_legend_handles_labels()
    last_ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, (i % cols) / cols + 0.5 / cols), title="Legend")   
    
    plt.show()

    return correlations


def plot_all_in_one(words:List[str], surprisals_df):
    plt.style.use('ggplot')
    plt.figure(figsize=(4, 3.5))

    max_x = 0

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        line, = plt.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o', alpha=0.7)

        # annotate the end of the line
        x = word_data['Steps'].iloc[-1]
        y = word_data['MeanSurprisal'].iloc[-1]
        plt.annotate(word, (x, y), textcoords="offset points", xytext=(+10,+0), color=line.get_color())

        max_x = max(max_x, x)

    xlim = plt.gca().get_xlim()
    plt.gca().set_xlim(xlim[0], max_x * 1.3)  # increase the maximum x value by 30%

    # plt.title('All Words')
    plt.xlabel('Steps')
    plt.ylabel('Mean surprisal')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def get_avg_df(dfs: List[pd.DataFrame], column: str):
    avg_dfs = []
    for df in dfs:
        avg = (df.groupby('Steps')
                 .agg({column: 'mean'})
                 .reset_index()
                 .assign(Diffs=lambda x: x[column].diff().fillna(0)))
        avg_dfs.append(avg)
    return avg_dfs


def plot_freq_infreq_full(freq, infreq, full, column: str):
    plt.figure()

    plt.plot(full.Steps, full[column], linestyle='--', alpha=0.6, label='Full Corpus')
    plt.plot(freq.Steps, freq[column], marker='o', label='Frequent Words')
    plt.plot(infreq.Steps, infreq[column], marker='o', label='Infrequent Words')

    plt.ylim(max(freq[column].max(), infreq[column].max()), 0)
    plt.xlabel('Steps')
    plt.ylabel(column)
    plt.legend()
    plt.show()


def plot_avg_pos_neg(positives, negatives):
    plt.figure()
    pos_labels = ['Full Corpus', 'Frequent Words', 'Infrequent Words']
    neg_labels = ['Full Corpus (negatives)', 'Frequent Words (negatives)', 'Infrequent Words (negatives)']
    colors = ['purple', 'green', 'red']

    for i, df in enumerate(positives):
        plt.plot(df.Steps, df['MeanSurprisal'], marker='o', color=colors[i], label=pos_labels[i])
    
    for i, df in enumerate(negatives):
        plt.plot(df.Steps, df['MeanNegSurprisal'], marker='o', color=colors[i], alpha= 0.3, label=neg_labels[i])

    plt.gca().invert_yaxis()
    plt.xlabel('Steps')
    plt.ylabel('Mean Surprisal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()