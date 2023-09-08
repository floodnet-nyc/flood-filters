import tqdm
import contextlib
import numpy as np
import pandas as pd

@contextlib.contextmanager
def ignores():
    try:
        yield
    except Exception:
        import traceback
        traceback.print_exc()

def rolling_apply(df, func, *xs, desc=None, as_series=None):
    '''Rolling (multi-)window apply.'''
    idx = df.index
    idxs = [idx - xi for xi in xs]
    result = [
        func(*(df.loc[ix[i]:idx[i]] for ix in idxs))
        for i in tqdm.tqdm(np.arange(len(df)), desc=desc)
    ]
    return (
        (
            pd.DataFrame(index=idx)
            if not len(result) else
            pd.DataFrame(result, index=idx)
            if isinstance(result[0], (dict, pd.Series, pd.DataFrame))
            else pd.Series(result, index=idx)
        ) 
        if as_series is None else
        pd.Series(result, index=idx)
        if as_series else
        pd.DataFrame(result, index=idx)
    )
