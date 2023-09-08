import collections
import numpy as np
from ..filter import Filter, log


class ShortFilter(Filter):
    '''Buffer 3 points, if the middle point is >0 and the other two are zeros, set all to zero.
    Only valid if it's not raining or the first and last points are within 5 mins.

    Delay: 1 entry, maximum 5 minutes if rain else equal to time till the latest entry

    Cases:
    _   send
    -   send

    __  send
    _-  hold
    -_  send

    ___ send
    __- hold
    _-- unhold send
    --- send
    --_ send
    -__ send
    _-_ filter
    -_- send

    '''
    def __init__(self, maxlen=2, name='short', **kw):
        super().__init__(name, buffer_size=2 + maxlen, min_samples=2, **kw)
        self.maxlen = maxlen

    def __get_desc__(self):
        return {}

    def _should_hold_point(self, notnull):
        # read buffer
        buffer = self.buffer
        # Decide if the current point should be held
        for i in range(len(buffer)-1):
            (m1, d1, t1) = buffer[i]
            (m2, d2, t2) = buffer[i+1]
            if d1 == 0 and d2 > 0:
                return True

    def _should_release_points(self, notnull):
        # read buffer
        buffer = self.buffer
        (m1, d1, t1), *middle, (m4, d4, t4) = buffer

        if d1 == 0 and d4 == 0:
            for mi, di, ti in middle:
                if di > 0:
                    self.override(mi, np.nan)

        release = d4 == 0
        if release:
            self.buffer.clear()
        return release
