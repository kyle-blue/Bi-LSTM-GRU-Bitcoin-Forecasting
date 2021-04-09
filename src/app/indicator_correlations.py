import os
from typing import List, Type
from app.DataPreprocesser import DataPreprocesser
import ta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_INDICATORS = 15 # Number of indicators to reduce to (find x number of indicators with lowest corrlations)
def indicator_correlations(symbol: str):
    max_df_len = 1000
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{symbol}.parquet",
        max_dataset_size=max_df_len,
        should_ask_load=False # Don't load previously generated sequences (and don't ask)
    )
    # We don't need to preprocessor.preprocess() since we don't want the sequences

    df = preprocessor.get_df_original()
    df = add_all_indicators(df, symbol)
    df = remove_non_indicators(df)

    correlations = df.corr()
    plot_cor_heatmap(correlations, title="Correlation Heatmap", figsize=30, fontsize=15)

    correlations = reduce_correlation_matrix(correlations, NUM_INDICATORS)
    print("Reduced to 10 indicators with low correlations!")
    
    plot_cor_heatmap(correlations, title="Correlation Heatmap", figsize=15, fontsize=20)




def plot_cor_heatmap(correlations: pd.DataFrame, *, title:str, figsize: int, fontsize: int):
    figure = plt.figure(figsize=(figsize, figsize))
    ax = figure.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations, interpolation="nearest")
    cb = figure.colorbar(cax)
    cb.ax.tick_params(labelsize=figsize)
    plt.title(title, fontsize=int(figsize*2))

    ax.set_xticks(list(range(len(correlations.columns))))
    ax.set_xticklabels(correlations.columns,fontsize=fontsize)
    ax.set_yticks(list(range(len(correlations.columns))))
    ax.set_yticklabels(correlations.columns, fontsize=fontsize)
    plt.xticks(rotation=90)
    
    plt.show()

def remove_non_indicators(df: pd.DataFrame):
    ## Remove non-indicators
    for col in df.columns:
        is_indicator = "_ind_" in col
        if not is_indicator:
            del df[col]

    print("Removed all non-indicators!")
    print(df)
    return df

def add_all_indicators(df: pd.DataFrame, symbol: str):
    for col in df.columns:
        cur_symbol = col.split("_")[0]
        if cur_symbol == symbol:
            df = ta.add_all_ta_features(
                df, f"{cur_symbol}_open", f"{cur_symbol}_high", 
                f"{cur_symbol}_low", f"{cur_symbol}_close",
                f"{cur_symbol}_volume", fillna=True, colprefix=f"{cur_symbol}_ind_"
            )
            df.dropna(inplace=True)

    print("Added all indicators!")
    return df


def reduce_correlation_matrix(correlations: pd.DataFrame, reduction_size: int):
    best_indicators: List[str] = []
    
    correlations = correlations.abs()
    correlations_original = correlations.copy()

    cor = correlations.to_numpy()
    row_sums = np.sum(cor, axis=1)
    min_row = np.argmin(row_sums)
    best_indicators.append(correlations.columns[min_row])

    correlations.drop(correlations.index[min_row], axis="index", inplace=True)
    correlations.drop(correlations.columns[min_row], axis="columns", inplace=True)

    while len(best_indicators) < reduction_size:
        row_sums = []
        print(best_indicators)
        for index, row in correlations.iterrows():
            row_sums.append(correlations_original.loc[index, best_indicators].sum())
        min_row = np.argmin(row_sums)
        ind = correlations.columns[min_row]
        if ind not in best_indicators:
            best_indicators.append(ind)
            correlations.drop(correlations.index[min_row], axis="index", inplace=True)
            correlations.drop(correlations.columns[min_row], axis="columns", inplace=True)
                
    ret = correlations_original.loc[best_indicators, best_indicators]
    return ret