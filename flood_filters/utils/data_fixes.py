import tqdm
import numpy as np
import pandas as pd



def interp(x):
    xx = x.fillna(method='ffill')
    xx[(xx.diff() == 0) & (x != 0)] = np.nan
    if pd.isna(xx).any():
        xx = xx.interpolate().round(1)
    return xx

def interpolate_discretized_data(df, col='depth_proc_mm', out_col=None, labels=('flood',)):
    '''Interpolate'''
    for e, dfi in df[df.label.isin(labels)].groupby('event_id'):
        x = dfi[col]
        xx = x.fillna(method='ffill')
        xx[(xx.diff() == 0) & (x != 0)] = np.nan
        if pd.isna(xx).any():
            tqdm.tqdm.write(f'smoothing flood event {e}')#.reset_index(drop=True)
            xx = xx.interpolate().round(1)#pd.Series(.values, x.index)
        df.loc[x.index, out_col or col] = xx[x.index].values
    return df
