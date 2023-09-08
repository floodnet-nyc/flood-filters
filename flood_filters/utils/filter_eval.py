import numpy as np



def get_filter_statistics(df):
    total_number_of_points = len(df)

    # df['label'] = df.label.replace('something', 'complex')

    event_points = df[~df.label.isin([np.nan]) & (df.depth_filt_mm > 0)].dropna(subset=['depth_filt_mm'])
    filtered_points = event_points[(event_points.depth_proc_mm > 0)].dropna(subset=['depth_proc_mm'])

    noise_points = event_points[~event_points.label.isin(['flood', 'snow', 'something'])]
    filtered_noise_points = noise_points[noise_points.depth_proc_mm > 0].dropna(subset=['depth_proc_mm'])

    total_noise_points = len(noise_points)
    total_noise_points_after_filter = len(filtered_noise_points)
    before_count_by_label = event_points.groupby('label').depth_proc_mm.count()
    after_count_by_label = filtered_points.groupby('label').depth_proc_mm.count()

    before_event_count_by_label = event_points.groupby('label').event_id.nunique()
    after_event_count_by_label = filtered_points.groupby('label').event_id.nunique()
    before_event_count_by_label['total_flood'] = before_event_count_by_label[before_event_count_by_label.index.isin(['flood', 'snow'])].sum()
    after_event_count_by_label['total_flood'] = after_event_count_by_label[after_event_count_by_label.index.isin(['flood', 'snow'])].sum()
    before_event_count_by_label['total_noise'] = before_event_count_by_label[~before_event_count_by_label.index.isin(['flood', 'snow'])].sum()
    after_event_count_by_label['total_noise'] = after_event_count_by_label[~after_event_count_by_label.index.isin(['flood', 'snow'])].sum()

    total_events = len(noise_points.event_id.unique())
    total_events_after_filter = len(filtered_noise_points.event_id.unique())
    return {
        "total_number_of_points": total_number_of_points,
        "total_noise_points": total_noise_points,
        "total_noise_points_after_filter": total_noise_points_after_filter,
        "total_proportion_of_noise": total_noise_points / total_number_of_points,
        "total_proportion_of_noise_after_filter": total_noise_points_after_filter / total_number_of_points,

        "total_events": total_events,
        "total_events_after_filter": total_events_after_filter,
        "total_proportion_of_noise_events_after_filter": total_events_after_filter / total_events,

        **({f'before_{k}': v for k, v in before_count_by_label.items()}),
        **({f'after_{k}': v for k, v in after_count_by_label.items()}),
        **({f'eventbefore_{k}': v for k, v in before_event_count_by_label.items()}),
        **({f'eventafter_{k}': v for k, v in after_event_count_by_label.items()}),
        # "before": before_count_by_label,
        # "after": after_count_by_label,
    }
