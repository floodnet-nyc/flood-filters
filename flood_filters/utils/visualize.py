import os
import tqdm
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from .misc import ignores
from .data_fixes import interpolate_discretized_data
from .. import get_filterbank

opj = os.path.join


def _plot_event(df):
    df.depth_filt_mm.plot(marker='.', c='k', label='filt')
    df.depth_proc_mm.plot(marker='.', c='r', label='proc')
    start, end = df.start.dropna().min(), df.end.dropna().max()
    plt.axvline(start, c='k', linestyle='--', alpha=0.4)
    plt.axvline(end, c='k', linestyle='--', alpha=0.4)
    ax = plt.gca()
    axb = ax.twinx()
    df.max_precip_last_5min_mm_per_min.plot(c='b', label='precip')
    axb.set_ylim([0, None])
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)
    plt.title(str(start))



def _plot_ml_alert(df):
    with ignores():
        df.flood_detected.plot(c='b', label='ml')
    with ignores():
        df.alert_signal.plot(marker='.', c='k', label='alert')
    for i, x in df.alert_trigger[df.alert_trigger>0].items():
        plt.axvline(i, c='r')
    plt.legend()


def plot_event(df):
    plt.figure(figsize=(15, 5))
    _plot_event(df)

def plot_alert(df):
    plt.figure(figsize=(15, 6))
    plt.subplot(211)
    _plot_event(df)

    plt.subplot(212)
    _plot_ml_alert(df)


def isolate_event(df, event_id):
    i = df.event_id == event_id
    df = df.copy()
    df.loc[~i, ['start', 'end']] = np.nan
    return df


def event_plotter(df, plot, dep_id='unknown', out_dir='event_plots', mins=30):
    for i in tqdm.tqdm(df.event_id.dropna().unique(), desc=f'plotting {dep_id}...'):
        dfe = df[df.event_id == i]
        dfi = df.loc[
            dfe.index.min() - timedelta(minutes=mins):
            dfe.index.max() + timedelta(minutes=mins)]
        if not len(dfi):
            tqdm.tqdm.write(f'no data for event {i}')
            continue
        lbl, *_ = [*dfe.label.dropna().unique()] or ['none']
        start = dfe.start.min()

        if lbl == 'flood':
            interpolate_discretized_data(dfi)

        # save plot
        # dfi = isolate_event(dfi, i)
        plot(dfi)
        fname = opj(out_dir, lbl, f'{i}_{dep_id}_{start}.jpg')
        tqdm.tqdm.write(fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)
        plt.close()
        # input()

# from IPython import embed
def deployments_event_plotter(df, plot, out_dir='event_plots', fbank=None, mins=30):
    df = df.reset_index().set_index('time')
    for dep_id, dfd in df.groupby('deployment_id'):
        if fbank is not None:
            fbank = get_filterbank(fbank)
            print("Using fbank", fbank)
            dfd = dfd.copy()
            dfd['depth_proc_mm'] = dfd['depth_filt_mm']
            dfd = fbank.apply(dfd, desc=dep_id, leave=True)
        event_plotter(dfd, plot, dep_id, out_dir, mins)
        # embed()

def load_and_plot(
        data_dir='data', 
        out_dir='event_plots',
        **kw
):
    import glob
    from .load_data import load_data
    out_dir = os.path.join(out_dir, datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

    df = load_data(
        glob.glob(opj(data_dir, 'depth/*.csv')), 
        glob.glob(opj(data_dir, 'Events_523.csv')), 
        glob.glob(opj(data_dir, 'weather/*.csv')), 
    )
    print(df.shape)
    print(df.columns)
    deployments_event_plotter(df, plot_event, out_dir, **kw)


if __name__ == '__main__':
    import fire
    fire.Fire(load_and_plot)