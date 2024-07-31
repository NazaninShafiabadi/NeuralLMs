""" Contains the functions needed for analyzing learning curves. """

import math
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def fit_linear_model(X:np.ndarray, y:np.ndarray, first_step=True) -> Dict[str, float]:
    if not first_step:
        X, y = X[1:], y[1:]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    alpha = model.coef_[0]  # slope
    beta = model.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return X.flatten(), y_pred, {'alpha': alpha, 'beta': beta, 'R2': r2, 'MSE': mse}


def plot_surprisals(words:List[str], surprisals_df, neg_samples=False, first_step=True, fit_line=False, return_outputs=False):
    """ 
    If first_step is set to False, neither the correlations nor the linear model will consider 
    the first step, but the first step will still be shown on the plot.
    """
    num_words = len(words)
    cols = min(num_words, 3)
    rows = math.ceil(num_words / cols)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axs = np.atleast_2d(axs)

    correlations = {}
    all_metrics = {}

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        ax = axs[i//cols, i%cols]

        metrics = {}

        # Fit linear model for positive samples
        X = word_data['Steps'].values.reshape(-1, 1)
        y_pos = word_data['MeanSurprisal'].values
        X_flat, y_pred_pos, metrics['positive'] = fit_linear_model(X, y_pos, first_step=first_step)
        
        # Plot positive surprisals & fitted line
        ax.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o', color='darkseagreen', label='Positive Samples')
        if fit_line:
            ax.plot(X_flat, y_pred_pos, linestyle='--', color='#043927', label='Positive Fit')

        if neg_samples:
            y_neg = word_data['MeanNegSurprisal'].values
            X_flat, y_pred_neg, metrics['negative'] = fit_linear_model(X, y_neg, first_step=first_step)
            
            # Plot negative surprisals & fitted line
            ax.plot(word_data['Steps'], word_data['MeanNegSurprisal'], marker='o', color='indianred', label='Negative Samples')
            if fit_line:
                ax.plot(X_flat, y_pred_neg, linestyle='--', color='#8D021F', label='Negative Fit')

            # Calculate correlation
            if not first_step:
                word_data = word_data[word_data['Steps'] != 0]
                
            corr, _ = pearsonr(word_data['MeanSurprisal'], word_data['MeanNegSurprisal'])
            correlations[word] = corr
            ax.set_title(f'"{word}"', pad=18)
            ax.text(0.5, 1.02, f'Correlation: {corr:.2f}', fontsize=10, ha='center', transform=ax.transAxes)

        else:
            ax.set_title(f'"{word}"')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean surprisal')
        ax.invert_yaxis()

        all_metrics[word] = metrics
    
    # Remove empty subplots
    for j in range(i+1, rows*cols):
        fig.delaxes(axs.flatten()[j])
    
    plt.tight_layout()

    # Legend
    last_ax = axs.flatten()[i]
    handles, labels = last_ax.get_legend_handles_labels()
    last_ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, (i % cols) / cols + 0.5 / cols), title="Legend")   
    
    plt.show()

    if return_outputs:
        return correlations, all_metrics
    else: 
        return


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


def corr_plot(words:List[str], df, first_step=True) -> None:
    """ The df needs to have individual surprisals """
    for w in words:
        w_data = df[df.Token == w].reset_index(drop=True)
        w_count = w_data.groupby('Steps')['Token'].count().values[0]
        X = w_data['Steps'].values.reshape(-1, 1)
        y = w_data['Surprisal'].values
        X_flat, y_pred, metrics = fit_linear_model(X, y, first_step=first_step)
        g = sns.lineplot(data=w_data, x='Steps', y='Surprisal', errorbar=('ci', 100))
        g.plot(X_flat, y_pred, linestyle='--', color='dimgray', label='Positive Fit')
        g.figure.suptitle(f"{w} ({w_count} samples)")
        g.text(0.5, 1.02, f'α: {metrics['alpha']:.2e}', fontsize=10, ha='center', transform=g.transAxes)
        plt.show()