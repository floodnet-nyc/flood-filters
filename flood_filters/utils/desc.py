


def event_counts(df):
    return df.groupby(['deployment_id', 'label']).index.count().unstack().fillna(0).astype(int)