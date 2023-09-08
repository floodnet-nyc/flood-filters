from datetime import datetime, timezone

import tqdm
import pandas as pd



class FilterBank:
    def __init__(self, filters):
        self.filters = filters

    def __str__(self):
        fs = ','.join(f'\n  {f}' for f in self.filters)
        return f'{self.__class__.__name__}({fs})'

    def clear(self):
        for f in self.filters:
            if hasattr(f, 'clear'):
                f.clear()

    def __call__(self, *msgs):
        for f in self.filters:
            msgs = [(m, ti) for msg, t in msgs for m, ti in f(msg, t)]
        return msgs

    def filter(self, *msgs):
        return self(*msgs)

    def apply(self, X, desc=None, progress=True, clear=True, leave=False):
        if clear: self.clear()
        it = X.iterrows()
        result = {}
        if desc is None:
            desc = '>'.join(f.name for f in self.filters if hasattr(f, 'name'))
        it = tqdm.tqdm(it, total=len(X), desc=f'{desc or ""} filtering...', leave=leave) if progress else it
        for t, row in it:
            for rowi, ti in self.filter((row, t.timestamp())):
                result[datetime.fromtimestamp(ti, tz=timezone.utc)] = rowi
        return pd.DataFrame.from_dict(result, orient='index').sort_index()


def apply_filters(X, filters, **kw):
    return FilterBank(filters).apply(X, **kw)
