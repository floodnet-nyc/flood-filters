import os
import glob
import json
import tqdm
from datetime import timedelta
import numpy as np
import pandas as pd
from flood_filters import *
from flood_filters.utils.load_data import load_data
from flood_filters.models import FloodDetector

CLS_NAMES = np.array(['flood', 'blip', 'pulse-chain', 'box', 'snow', 'something'])


# load

def load_csvs(filenames, event_fname):
    df = pd.concat([pd.read_csv(f) for f in tqdm.tqdm(filenames, desc='loading...')])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()

    event_df = load_events(event_fname).iloc[:100]

    df = df[df.deployment_id.isin(event_df.Deployment_id.unique())]
    df = df.loc[event_df.start.min():event_df.end.max()]

    df['event_id'] = pd.NA
    df['class_id'] = pd.NA
    df['label'] = pd.NA

    for deployment_id, edf in tqdm.tqdm(event_df.groupby('Deployment_id'), desc='adding events...'):
        is_dep = df.deployment_id == deployment_id
        for eid, row in edf.iterrows():
            mask = (
                is_dep &
                (df.index >= row.start - timedelta(minutes=1)) &
                (df.index < row.end + timedelta(minutes=1))
            )
            df.loc[mask, 'event_id'] = eid
            df.loc[mask, 'class_id'] = row.Class
            df.loc[mask, 'label'] = row.label
            df.loc[mask, 'duration'] = row.duration
            df.loc[mask, 'start'] = row.start
            df.loc[mask, 'end'] = row.end

    print(df.shape)
    print(df.head())

    return df


def load_events(event_fname):
    # load all events
    event_df = pd.read_csv(event_fname)
    event_df['start'] = pd.to_datetime(event_df.Start_time, format='%Y-%m-%d %H:%M:%S', utc=True)
    event_df['end'] = pd.to_datetime(event_df.End_time, format='%Y-%m-%d %H:%M:%S', utc=True)
    event_df['duration'] = (event_df.end - event_df.start).dt.total_seconds()
    event_df['label'] = CLS_NAMES[event_df.Class]
    event_df = event_df.sort_values('start')
    event_df = event_df[~event_df.label.isin(['snow', 'something'])]
    return event_df




def get_alert_signal(df, col, threshold, mins=5):
    y = df.groupby(pd.Grouper(freq=f'{mins}min')).apply(lambda x: pd.Series({
        'signal': x[col].mean() > threshold,
        'label': (x.label.mode().tolist() or [pd.NA])[0],
        'event_id': (x.event_id.mode().tolist() or [pd.NA])[0],
        'start': x.start.min(),
    }))
    y['trigger'] = y['signal'].diff().clip(0)
    return y

def get_alert_delay(P):
    if not len(P):
        return []

    def get_alert_delay(df):
        on = df[df.trigger == 1]
        return (on.index.min() - df.index.min()).total_seconds() if len(on) else pd.NA
    return P.groupby('event_id').apply(get_alert_delay).tolist()

def get_alert_metrics(alert):
    # get true/false positives for alert triggers
    print("getting tp/fp/fn")
    P = alert.loc[alert.label == 'flood']
    N = alert.loc[alert.label != 'flood']
    TP = P.groupby('event_id').trigger.any().sum()
    FP = N.trigger.sum()
    FN = len(P.event_id.unique()) - TP

    # get alert delay
    print("getting alert delay")
    
    return TP, FP, FN, get_alert_delay(P)

def integrate(x):
    return (x * x.index.diff()).sum()


def calc_statistics():
    pass

# 

def run(data_dir, event_fname='data/Events.csv'):
    data_df = load_data(
        glob.glob(os.path.join(data_dir, '*.csv')), 
        event_fname)

    is_raining = None

    filters = [
        # ('none', lambda: FilterBank([])),
        # ('grad', lambda: FilterBank([
        #     RangeFilter(),
        #     GradFilter(),
        # ])),
        # ('grad+blip+box', lambda: FilterBank([
        #     RangeFilter(),
        #     GradFilter(),
        #     BlipFilter(is_raining=is_raining),
        #     BoxFilter(is_raining=is_raining),
        # ])),
        ('grad+blip+box+blip', lambda: FilterBank([
            RangeFilter(),
            GradFilter(),
            BlipFilter(is_raining=is_raining),
            BoxFilter(is_raining=is_raining),
            BlipFilter(is_raining=is_raining),
        ])),
    ]
    mlfbank = FilterBank([FloodDetector()])

    for filt_name, get_filter in filters:
        TP = 0
        FP = 0
        FN = 0
        delays = []
        results = []
        print(filt_name)
        for deployment_id, ddf in tqdm.tqdm(data_df.groupby('deployment_id'), total=len(data_df.deployment_id.unique())):
            print(deployment_id)
            print('floods:', ddf.event_id[ddf.label == 'flood'].unique())
            print(ddf.label.value_counts())
            # filter data
            fbank = get_filter()
            ddf_filtered = fbank.apply(ddf, desc=deployment_id, leave=True)
            ddf_filtered = mlfbank.apply(ddf_filtered, desc='ml', leave=True)
            print()

            col = 'depth_proc_mm'

            # simulate alert rules
            print(col, "getting alert rules")
            alert = get_alert_signal(ddf_filtered, col, 20, mins=5)

            # get true/false positives for alert triggers
            TP, FP, FN, delays = get_alert_metrics(alert)

            print('TP', TP, 'FP', FP, 'FN', FN, pd.Series(delays).mean())
            # if len(delays):
            #     break

            results.append({
                'TP': int(TP), 'FP': int(FP), 'FN': int(FN),
                'delays': pd.Series(delays).mean(),
                'column': col,
            })

            col = 'flood_detected'

            # simulate alert rules
            print(col, "getting alert rules")
            alert = get_alert_signal(ddf_filtered, col, 0.5, mins=5)

            # get true/false positives for alert triggers
            TP, FP, FN, delays = get_alert_metrics(alert)

            print('TP', TP, 'FP', FP, 'FN', FN, pd.Series(delays).mean())
            # if len(delays):
            #     break

            results.append({
                'TP': int(TP), 'FP': int(FP), 'FN': int(FN),
                'delays': pd.Series(delays).mean(),
                'column': col,
            })

        results_df = pd.DataFrame(results).T
        results_df.to_csv(f'results_{filt_name}.csv')
        print_block(filt_name, results_df[['TP', 'FP', 'FN']].sum(), results_df.delays.mean())



def print_block(*xs, c='#', nc=40):
    print()
    print()
    print(c*nc)
    for x in xs:
        if isinstance(x, (list, tuple)):
            print(*x)
        elif isinstance(x, dict):
            print(*(x for kv in x.items() for x in kv))
        else:
            print(x)
    print(c*nc)
    print()
    print()


if __name__ == '__main__':
    import fire
    fire.Fire(run)