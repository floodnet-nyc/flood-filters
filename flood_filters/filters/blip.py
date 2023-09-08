import collections
import numpy as np
from ..filter import Filter, log


class BlipFilter(Filter):
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
    def __init__(self, blip_ratio=0.1, blip_min_dist=2, downward_blips=False, replace_method='nan', name='blip', **kw):
        super().__init__(name, buffer_size=3, min_samples=2, **kw)
        # how much higher does the distance between 
        self.blip_ratio = blip_ratio
        # 
        self.blip_min_dist = blip_min_dist
        # should we consider blips downwards?
        self.downward_blips = downward_blips
        self.replace_method = replace_method

    def __get_desc__(self):
        return {
            'ratio': self.blip_ratio, 'min_dist': self.blip_min_dist, 
            'holding': self.holding, 'n': len(self.buffer)
        }

    # def _filter(self, msg, depth, t):
    #     if self.holding:
    #         # we held a buffer, let's see what happened
    #         (_, d1, t1), (m2, d2, t2), (_, d3, t3) = self.buffer
    #         base_dist = abs(d1 - d3)
    #         signed_blip_dist = d2 - d1
    #         blip_dist = abs(signed_blip_dist) if self.downward_blips else signed_blip_dist
    #         if (
    #             blip_dist > self.blip_min_dist and 
    #             base_dist / abs(blip_dist) < self.blip_ratio and 
    #             not self.is_invalid(msg, t, t2)
    #         ):
    #             log.info('blip filtered [%f,%f,%f] @ %s', d1, d2, d3, t)
    #             if self.replace_method == 'nan':
    #                 dn = np.nan
    #             else:
    #                 dn = (d1+d3)/2
    #             self.override(m2, dn, self.name)
    #         self.holding = False
    #         yield m2, t2

    #     # check for blip pre-condition: __-
    #     else:
    #         if depth - self.buffer[-2][1] > self.blip_min_dist:
    #             self.holding = True
    #             log.debug('blip hold %s %s', self.buffer[-2][1], depth)
    #             return

    #     # otherwise we're good
    #     yield msg, t

    def _should_hold_point(self, notnull):
        # read buffer
        buffer = self.buffer
        *_, (_, d1, _), (m2, d2, t2) = buffer

        # Decide if the current point should be held based on the blip_threshold
        return d2 - d1 > self.blip_min_dist

    def _should_release_points(self, notnull):
        # read buffer
        buffer = self.buffer
        (_, d1, _), (m2, d2, t2), (m3, d3, t3) = buffer

        # check blip conditions
        base_dist = abs(d1 - d3)
        signed_blip_dist = d2 - d1
        blip_dist = abs(signed_blip_dist) if self.downward_blips else signed_blip_dist

        filtering = (
            # the jump up was high enough
            blip_dist > self.blip_min_dist and 
            # the blip height vs the base difference was high enough
            base_dist / abs(blip_dist) < self.blip_ratio and 
            # the time range was small enough, or it wasn't raining
            not self.is_invalid(m3, t3, t2)
        )

        # null out the point
        if filtering: 
            self.override(m2, np.nan)
        return True
