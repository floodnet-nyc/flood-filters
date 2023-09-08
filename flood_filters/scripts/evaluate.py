import os
import glob
import json
import tqdm
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flood_filters import *
from flood_filters.models import FloodDetector

from flood_filters.utils.profiler import profile
from flood_filters.utils.load_data import load_data, load_events
from flood_filters.utils.alert import get_alert_signal, get_alert_metrics, get_alert_stats
from flood_filters.utils.visualize import plot_alert, event_plotter
from flood_filters.utils.filter_eval import get_filter_statistics

CLS_NAMES = np.array(['flood', 'blip', 'pulse-chain', 'box', 'snow', 'something'])


# load

def get_filters():
    filters = [
        # ('asis', None),
        # # ('none', lambda: FilterBank([])),
        # ('grad', lambda: FilterBank([
        #     RangeFilter(),
        #     GradFilter(),
        # ])),
        # # ('grad+blip+box', lambda: FilterBank([
        # #     RangeFilter(),
        # #     GradFilter(),
        # #     BlipFilter(is_raining=is_raining),
        # #     BoxFilter(is_raining=is_raining),
        # # ])),
        ('grad+blip+box+blip|undefined', lambda: FilterBank([
            RangeFilter(),
            GradFilter(),
            BlipFilter(),
            BoxFilter(),
            BlipFilter(),
        ])),
    ]
    
    return filters


def run_alert(data_dir='data', event_fname=None, ml_fname=None, limit=None):
    data_df, events_df = load_data(
        glob.glob(os.path.join(data_dir, 'depth', '*.csv')), 
        event_fname or os.path.join(data_dir, 'Events_523.csv'),
        glob.glob(os.path.join(data_dir, 'weather', '*.csv')), 
        return_events=True,
    )
    
    # downselecting only the info we need
    # data_df = data_df.reset_index().set_index('time')
    data_df = data_df[[
        # 'deployment_id', #'time', # in index
        'event_id', 'label',
        'depth_filt_mm', 'depth_proc_mm',
        'start', 'end', 
    ]]

    counts = (data_df
        .groupby(['deployment_id', 'label'])
            .apply(lambda x: len(x.event_id.unique()))
            .unstack().fillna(0).astype(int).sort_values('flood', ascending=False))
    print(counts)

    filters = get_filters()
    mlfbank = FilterBank([FloodDetector(ml_fname)])

    # from IPython import embed
    # embed()

    with profile():
        for filt_name, get_filter in filters:
            results = []
            try:
                for deployment_id in tqdm.tqdm(counts.index):
                    ddf = data_df.loc[deployment_id]
                    ddf = ddf.iloc[:limit] if limit else ddf
                    ddf = ddf.copy()

                    tqdm.tqdm.write(f"{deployment_id}")
                    tqdm.tqdm.write(f"flood events: {ddf[ddf.label == 'flood'].event_id.dropna().unique()}")

                    tqdm.tqdm.write("Performing filters...")
                    if get_filter is not None:
                        fbank = get_filter()
                        print(fbank)
                        ddf['depth_proc_mm'] = ddf['depth_filt_mm']
                        ddf = fbank.apply(ddf, desc=deployment_id, leave=True)
                    # ddf = mlfbank.apply(ddf, desc='ml', leave=True)

                    tqdm.tqdm.write("Calculating alert signal...")
                    column = "flood_detected"
                    threshold = 0.5
                    alert = get_alert_signal(ddf, column, threshold, mins=5)
                    event_plotter(ddf, plot_alert, deployment_id, 'alert_plots')

                    tqdm.tqdm.write("Calculating alert metrics...")
                    metrics = get_alert_metrics(alert)

                    results.append({
                        **metrics, 
                        'deployment_id': deployment_id,
                        'column': column,
                        'threshold': threshold,
                        'model_id': mlfbank.filters[0].model_id,
                    })
                    print_block(results[-1])
            finally:
                result_fname = f'results/alert_results_{filt_name}.csv'
                results_df = pd.DataFrame.from_records(results)
                # if os.path.isfile(result_fname):
                #     results_df = pd.concat([
                #         pd.read_csv(result_fname),
                #         results_df,
                #     ])
                results_df.to_csv(result_fname)



def run_filter(data_dir='data', event_fname=None, limit=None):
    event_fname = event_fname or os.path.join(data_dir, 'Events_523.csv')
    data_df = load_data(
        glob.glob(os.path.join(data_dir, 'depth', '*.csv')), 
        event_fname,
        glob.glob(os.path.join(data_dir, 'weather', '*.csv')), 
    )
    events_df = load_events(event_fname)
    
    # downselecting only the info we need
    # data_df = data_df.reset_index().set_index('time')
    data_df = data_df[[
        # 'deployment_id', #'time', # in index
        'event_id', 'label',
        'depth_filt_mm', 'depth_proc_mm',
        'start', 'end', 
    ]]

    counts = (data_df
        .groupby(['deployment_id', 'label'])
            .apply(lambda x: len(x.event_id.unique()))
            .unstack().fillna(0).astype(int).sort_values('flood', ascending=False))
    print(counts)

    filters = get_filters()

    # from IPython import embed
    # embed()

    with profile():
        for filt_name, get_filter in filters:
            results = []
            try:
                for deployment_id in tqdm.tqdm(counts.index):
                    ddf = data_df.loc[deployment_id]
                    ddf = ddf.iloc[:limit] if limit else ddf
                    ddf = ddf.copy()

                    tqdm.tqdm.write(f"{deployment_id}")
                    tqdm.tqdm.write(f"flood events: {ddf[ddf.label == 'flood'].event_id.dropna().unique()}")

                    tqdm.tqdm.write("Performing filters...")
                    if get_filter is not None:
                        fbank = get_filter()
                        ddf['depth_proc_mm'] = ddf['depth_filt_mm']
                        ddf = fbank.apply(ddf, desc=deployment_id, leave=True)

                    stats = get_filter_statistics(ddf)
                    results.append({
                        **stats, 
                        'deployment_id': deployment_id,
                    })
                    print_block(results[-1])
            finally:
                result_fname = f'results/filter_results_{filt_name}.csv'
                results_df = pd.DataFrame.from_records(results)
                results_df = results_df.sort_values('total_proportion_of_noise', ascending=False)
                # if os.path.isfile(result_fname):
                #     results_df = pd.concat([
                #         pd.read_csv(result_fname),
                #         results_df,
                #     ])
                results_df.to_csv(result_fname)
                plot_filter_results(results_df, events_df, filt_name)

                results_df



# def filt_stats(df):
#     results = []

#     df = df.copy()
#     df['depth_filt_mm'] = df.depth_raw_mm
#     fbank = FilterBank([RangeFilter()])
#     x_range = fbank.apply(x)


#     fbank = FilterBank([RangeFilter(), GradFilter()])
#     x_grad = fbank.apply(x)

#     return {
#         'range': x_range.groupby('label').depth_proc_mm.count(),
#         'gradient': x_grad.groupby('label').depth_proc_mm,
#     }
    





def plot_filter_results(results_df, events_df, filt_name):
    plt.figure(figsize=(16, 5))
    (results_df
        .sort_values('total_proportion_of_noise', ascending=False)
        .rename(columns={
        'total_proportion_of_noise': 'before',
        'total_proportion_of_noise_after_filter': 'after',
    }).plot(x="deployment_id", y=["before", "after"], kind="bar"))
    plt.title('proportion of noise before and after filters applied')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/filter_results_noise_prop_{filt_name}.png')
    plt.close()

    plt.figure(figsize=(16, 5))
    (results_df
        .sort_values('total_events', ascending=False)
        .rename(columns={
        'total_events': 'before',
        'total_events_after_filter': 'after',
    }).plot(x="deployment_id", y=["before", "after"], kind="bar"))
    plt.title('count of noise events before and after filters applied')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/filter_results_events_{filt_name}.png')
    plt.close()

    plt.figure(figsize=(16, 5))
    ax = plt.gca()
    before_cols = {c: c.removeprefix('before_') for c in results_df.columns if c.startswith('before_')}
    after_cols = {c: c.removeprefix('after_') for c in results_df.columns if c.startswith('after_')}
    d=pd.concat([
        (results_df[list(before_cols)]
            .rename(columns=before_cols)
            .sum() / results_df.total_number_of_points.sum())
            .to_frame().T
            .assign(kind='before filter'),
        (results_df[list(after_cols)]
            .rename(columns=after_cols)
            .sum() / results_df.total_number_of_points.sum())
            .to_frame().T
            .assign(kind='after filter'),
    ]).set_index('kind').T
    orig = d['before filter']
    new = d['after filter']
    # orig.plot.bar(ax=ax)
    # (new - orig).plot.bar(ax=ax, bottom=orig, label='after filter')
    plot_delta(orig, new)

    before_cols = {c: c.removeprefix('eventbefore_') for c in results_df.columns if c.startswith('eventbefore_')}
    after_cols = {c: c.removeprefix('eventafter_') for c in results_df.columns if c.startswith('eventafter_')}
    d = pd.concat([
        results_df[before_cols].rename(columns=before_cols).fillna(0).sum().astype(int).to_frame().T.assign(kind='before filter'), 
        results_df[after_cols].rename(columns=after_cols).fillna(0).sum().astype(int).to_frame().T.assign(kind='after filter'), 
    ]).set_index('kind')
    print(d)
    d.to_csv(f'results/noise_type_results_{filt_name}.csv')

    # pd.DataFrame({
    #     'before filter': d['before filter'],
    #     '': d['before filter']
    # }).plot(kind='bar', ax=ax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/noise_type_results_{filt_name}.png')
    plt.close()


def plot_delta(orig, new):
    plt.barh(orig.index, orig.values, color='#ff384c')
    plt.barh(orig.index, (new - orig).values, color='#dbdbdb', left=orig.values)
    import matplotlib.ticker as mtick
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 2))


def print_block(*xs, c='#', nc=40):
    print()
    print()
    print(c*nc)
    for x in xs:
        if isinstance(x, (list, tuple)):
            print(*x)
        elif isinstance(x, dict):
            for k, v in x.items():
                print(k, v)
        else:
            print(x)
    print(c*nc)
    print()
    print()





if __name__ == '__main__':
    import fire
    fire.Fire({'filter': run_filter, 'alert': run_alert})