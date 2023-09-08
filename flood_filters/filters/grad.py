from datetime import datetime
import numpy as np
from ..filter import Filter, log


class GradFilter(Filter):
    '''Compare this point with the previous point and calculate gradient. If gradient greater than 5 inches/min, 
    set its value with the previous point
    '''
    def __init__(self, inchmin=10, spotty_coverage_height_ratio=0.05, replace_method='nan', name='gradient', **kw):
        super().__init__(name, buffer_size=2, min_samples=2, **kw)
        self.min_mmps = inchmin * 25.4 / 60
        self.filtering = False
        self.spotty_coverage_height_ratio = spotty_coverage_height_ratio
        self.replace_method = replace_method
        self.ref_depth = None

    def __get_desc__(self):
        return {'min_mmps': self.min_mmps, 'last': self.buffer[0][1] if self.buffer else None}
    
    def clear(self):
        super().clear()
        self.filtering = False
        self.true_last_depth = None
        self.ref_depth = None

    # def _filter(self, msg, depth, t):
    #     m1, d1,  t1 = self.buffer[0]
    #     # check that the change in depth is under some threshold
    #     dddt = (depth-d1) / max(0.1, (t - t1))
    #     if (
    #         dddt > self.min_mmps or 
    #         self.filtering and 
    #         self.true_last_depth and 
    #         abs(depth-self.true_last_depth) / self.true_last_depth < self.spotty_coverage_height_ratio
    #     ):  
    #         log.info(
    #             '%s gradient filtered %.0f -> %.0f in %.0fs: %.2fmm/s > %.2fmm/s', 
    #             datetime.fromtimestamp(t).isoformat(), 
    #             d1, depth, t-t1, dddt, self.min_mmps)

    #         # copy over depth
    #         if self.replace_method == 'nan':
    #             dn = np.nan
    #         else:
    #             dn = d1
    #         self.override(msg, dn, self.name)
    #         self.buffer[-1] = (msg, d1, t)
    #         self.filtering = True
    #     else:
    #         self.filtering = False
    #     #     log.info('%s no gradient %.0f -> %.0f in %.0fs: %.2fmm/s <= %.2fmm/s', datetime.fromtimestamp(t).isoformat(), d1, depth, t-t1, dddt, self.min_mmps)

    #     self.true_last_depth = depth
    #     yield msg, t

    def _on_new_point(self, notnull):
        # read the buffer
        buffer = self.buffer
        (_, d1, t1), (m2, d2, t2) = buffer

        # initialize the reference depth
        if self.ref_depth is None:
            self.ref_depth = d1

        # calculate the gradient
        dddt = (d2-self.ref_depth) / max(0.1, (t2 - t1))

        # filter where there's a high gradient
        self.filtering = (
            # high gradient
            dddt > self.min_mmps or 
            # or if there is a time gap, but the height is about the same
            self.filtering and 
            abs(d2-d1) / d1 < self.spotty_coverage_height_ratio
        )

        if self.filtering:
            # drop point with high gradient
            self.override(m2, np.nan)
        else:
            # ref_depth should be the last unfiltered point
            self.ref_depth = d2
