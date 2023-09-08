from datetime import timedelta

import tqdm
import numpy as np
import pandas as pd
from .misc import rolling_apply


def get_alert_signal(df, col, threshold, mins=5, mins2=20):
    '''Compute a sliding window alert signal like how grafana works.
    
    '''
    def alert_fn(x, x2):
        return {
            # 'signal': np.max([x[col].mean() > threshold, x2[col].mean() > threshold]),
            # 'label': (x.label.mode().tolist() or [pd.NA])[0],
            # 'event_id': (x.event_id.mode().tolist() or [pd.NA])[0],
            'signal': x[col].mean() > threshold,
            'label': (x.label.dropna().unique().tolist() or [pd.NA])[0],
            'event_id': (x.event_id.dropna().unique().tolist() or [pd.NA])[0],
            'start': x.start.dropna().min(),
        }

    # y = df.groupby(pd.Grouper(freq=f'{mins}min')).apply(alert_fn)
    # y = df.rolling(f'{mins}min', method='table').apply(alert_fn)
    y = rolling_apply(
        df[[col, 'label', 'event_id', 'start']], 
        alert_fn, 
        timedelta(minutes=mins), 
        timedelta(minutes=mins2 or mins*3), 
        desc=f'{col} > {threshold}')
    y['trigger'] = y['signal'].astype(int).diff().clip(0)

    df['alert_signal'] = y.signal
    df['alert_trigger'] = y.trigger
    df[f'{col}_{threshold}_alert_signal'] = y.signal
    df[f'{col}_{threshold}_alert_trigger'] = y.trigger
    return y




def get_alert_metrics(alert):
    # get true/false positives for alert triggers
    print("getting tp/fp/fn")
    P = alert.loc[alert.label == 'flood']
    N = alert.loc[alert.label != 'flood'].copy()
    N['label'] = N.label.fillna("none")

    TP = P.groupby('event_id').trigger.any().sum()
    TP_ALERT_COUNT = P.groupby('event_id').trigger.sum().mean()
    FP = N.groupby('label').trigger.sum()
    FN = len(P.event_id.unique()) - TP

    TP_ids = P.groupby('event_id').trigger.any().loc[lambda x: x].index.unique()
    FP_ids = N.groupby('event_id').trigger.any().loc[lambda x: x].index.unique()
    FN_ids = P.groupby('event_id').trigger.any().loc[lambda x: ~x].index.unique()
    
    print('TP', TP, TP_ALERT_COUNT, TP_ids)
    print('FP', FP.sum(), FP_ids)
    print('FN', FN, FN_ids)
    print('FP', FP)

    return {
        'TP': TP, "TP_ALERT_COUNT": TP_ALERT_COUNT, 'FP': FP.sum(), 'FN': FN,
        **({f'FP_{k}': v for k, v in FP.items()}),
        "TP_ids": TP_ids.tolist(),
        "FP_ids": FP_ids.tolist(),
        "FN_ids": FN_ids.tolist(),
    }

def get_alert_stats(df, thresholds, mins=5):
    results = []
    for col, threshold in tqdm.tqdm([
        (col, th) for col, ths in thresholds.items() for th in ths
    ]):
        # simulate alert rules
        alert = get_alert_signal(df, col, threshold, mins=mins)
        results.append({
            **get_alert_metrics(alert),
            'column': col,
            'threshold': threshold,
        })
    return results
