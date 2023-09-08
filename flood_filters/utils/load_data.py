import os
import tqdm
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .misc import rolling_apply
from .profiler import profile_func

CLS_NAMES = np.array(['flood', 'blip', 'pulse-chain', 'box', 'snow', 'something'])


# ---------------------------------------------------------------------------- #
#                                     Load                                     #
# ---------------------------------------------------------------------------- #

# @profile_func
def load_data(
        depth_fs, event_fname=None, weather_fs=None, 
        drop_label=['snow'], cache_file='data/eval.pkl', 
):
    if cache_file and os.path.isfile(cache_file):
        df = pd.read_pickle(cache_file)
        print('loaded from', cache_file)
        return df
    df = load_depth(depth_fs)

    if event_fname is not None:
        event_df = load_events(event_fname)
        df = join_events(df, event_df)
        if drop_label:
            df = df[~df.label.isin(drop_label)]

    if weather_fs is not None:
        weather_df = load_weather(weather_fs)
        df = join_weather(df, weather_df)

    if cache_file:
        df.to_pickle(cache_file)
    return df


def load_depth(filenames):
    df = pd.concat([
        pd.read_csv(f)#, parse_dates=['time'] 
        for f in tqdm.tqdm(filenames, desc='loading...')
    ])
    # df = df[df.deployment_id.isin(['daily_happy_satyr'])]
    print(df.time.dtype)
    df['time'] = pd.to_datetime(df['time'], utc=True, format='%Y-%m-%d %H:%M:%S.%f%z')
    print(df.time.dtype)
    df = df.sort_values(['deployment_id', 'time'])
    df = df.set_index(['deployment_id', 'time'])
    print('loaded depth data:', df.shape, df.columns)
    return df


def load_weather(filenames, last_hour=False):
    df = pd.concat([
        pd.read_csv(f)
        for f in tqdm.tqdm(filenames, desc='reading weather...')
    ])
    df['time'] = pd.to_datetime(df.time, utc=True, format='ISO8601')
    # df = (
    #     df.groupby(pd.Grouper(key='time', freq='10Min'))
    #       .apply(lambda x: x.groupby('sensor_id').agg({
    #           "max_precip_last_5min_mm_per_min": "sum",
    #       }).max()))
    df = df.groupby(['sensor_id', pd.Grouper(key='time', freq='10Min')])[['max_precip_last_5min_mm_per_min']].sum()
    df = df.groupby(level=1).max()
    if last_hour:
        df['precip_last_hour'] = rolling_apply(
            df.max_precip_last_5min_mm_per_min,
            lambda x: x.max() > 0,
            timedelta(minutes=60),
            as_series=True,
            desc='last hour',
        )
    return df

def load_tide(filenames):
    df = pd.concat([
        pd.read_csv(f, parse_dates=['time'])
        for f in tqdm.tqdm(filenames, desc='reading tide...')
    ])
    df = df.drop(columns=['lat', 'lon'])
    df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
    # df['time'] = df.time.dt.tz_localize(None)
    df = (
        df.set_index('time')
          .sort_index()
          .groupby(['sensor_id', pd.Grouper(freq='5Min')])
          .mean().dropna())
    return df


def load_events(event_fname):
    # load all events
    event_df = pd.read_csv(event_fname)
    event_df = event_df.set_index(pd.Index(np.arange(len(event_df)), name='event_id'))
    event_df['start'] = pd.to_datetime(event_df.Start_time.str.strip(), format='%Y-%m-%d %H:%M:%S', utc=True)
    event_df['end'] = pd.to_datetime(event_df.End_time.str.strip(), format='%Y-%m-%d %H:%M:%S', utc=True)
    event_df['duration'] = (event_df.end - event_df.start).dt.total_seconds()
    event_df['label'] = CLS_NAMES[event_df.Class]
    event_df = event_df.sort_values('start')
    # event_df = event_df[event_df.index.isin([800])]
    print('loaded event data:', event_df.shape, event_df.columns)
    return event_df



# ---------------------------------------------------------------------------- #
#                                     Join                                     #
# ---------------------------------------------------------------------------- #

def join_events(df, event_df, trim_to_events=True):
    if trim_to_events:
        event_df = event_df[event_df.Deployment_id.isin(df.index.get_level_values(0).unique())]
        df = df[df.index.get_level_values(0).isin(event_df.Deployment_id.unique())]
        df = df.loc[pd.IndexSlice[:, event_df.start.min():event_df.end.max()], :]
        # df = df.groupby(level=0).apply(lambda d: d[event_df.start.min():event_df.end.max()])
        # t=df.index.get_level_values(1)
        # df = df.loc[(t >= event_df.start.min()) & (t <= event_df.end.max())]

    df['event_id'] = pd.NA
    df['class_id'] = pd.NA
    df['label'] = pd.NA

    df = df.reset_index(drop=False)
    for deployment_id, edf in tqdm.tqdm(event_df.groupby('Deployment_id'), desc='adding events...'):
        ddf = df[df.deployment_id == deployment_id]
        t = ddf.time
        idx_d = ddf.index
        # idx_d = df.loc[deployment_id:deployment_id].index
        # t = idx_d.get_level_values(1)
        for eid, row in tqdm.tqdm(edf.iterrows(), total=len(edf), desc=f'{deployment_id}'):
            idx = idx_d[(
                (t >= row.start - timedelta(minutes=1)) &
                (t <= row.end + timedelta(minutes=1))
            )]
            if not len(idx):
                tqdm.tqdm.write(f'no data available for {row.label} {row.start}')
                continue
            df.loc[idx, ['event_id', 'label', 'start', 'end']] = (
                eid, row.label, row.start, row.end)
            # if row.label == 'flood':
            #     df.loc[idx].depth_proc_mm.plot()
            #     plt.show()
    df = df.set_index(['deployment_id', 'time'])
    return df


def join_weather(df, weather_df):
    print("joining weather")
    t = df.index.get_level_values(1).to_series().dt.round('10min')
    # df = df.drop(columns=['max_precip_last_5min_mm_per_min_x', 'max_precip_last_5min_mm_per_min_y'])
    df = df.merge(weather_df.reset_index().set_index(['time']), how='left', left_on=t, right_index=True, suffixes=['', ''])
    print("joined")
    return df