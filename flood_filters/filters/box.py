import numpy as np
from ..filter import Filter, log

class BoxFilter(Filter):
    '''Filter out boxes from the flood depth data. e.g. __---------__ -> __        __
    
    Buffer 3 points, if the first point is zero, and then the latest two are the same/ std dev is close to zero, make them zero. 
    
    Delay: 1 entry, max delay is equal to time till the latest entry

    Cases:
    _   send
    -   send

    __  send
    _-  hold
    -_  send

    ___ send
    __- hold
    _-- filter ___
    --- filter ___
    --_ send
    -__ send
    _-_ send
    -_- send
    

    '''
    def __init__(self, box_ratio=0.05, name='box', replace_method='nan', **kw):
        super().__init__(name, buffer_size=3, min_samples=3, **kw)
        self.box_diff = 0
        self.box_jump = 0
        self.box_ratio = box_ratio
        self.filtering = False
        self.replace_method = replace_method

    def clear(self):
        super().clear()
        self.box_diff = 0
        self.filtering = False

    def __get_desc__(self):
        return {
            'ratio': self.box_ratio,
            'holding': self.holding, 'filtering': self.filtering, 'n': len(self.buffer),
        }

    # def _filter(self, msg, depth, t):
    #     (m2, d2, t2) = self.buffer[-2]
    #     if self.filtering:
    #         dd = depth-d2
    #         self.box_diff += dd
    #         if (
    #             # the relative distance change from the previous point
    #             abs(dd / self.box_jump) < self.box_ratio and
    #             # the relative distance change from the first point
    #             abs(self.box_diff / self.box_jump) < self.box_ratio and
    #             not self.is_invalid(msg, t, t2)
    #         ):
    #             log.info('box filtered %f %f %f %s', depth, dd, self.box_diff, t)
    #             if self.replace_method == 'nan':
    #                 dn = np.nan
    #             else:
    #                 dn = 0
    #             if self.holding:
    #                 self.override(m2, dn, self.name)
    #             self.override(msg, dn, self.name)
    #         else:
    #             if not self.holding:
    #                 log.info('box jump down %f < %f %s', depth / self.box_jump, self.box_ratio / 2, t)
    #                 if self.replace_method != 'nan' and depth / self.box_jump < self.box_ratio / 2:
    #                     log.info('box jump down clamped %s', t)
    #                     self.override(msg, 0, self.name)
    #                 log.info('. box filter done')
    #             self.filtering = False

    #     # check for box pre-condition: __-
    #     else:
    #         if d2 == 0 and depth > 0:
    #             log.debug('? box hold %f > %f', depth, d2)
    #             self.holding = True
    #             self.filtering = True
    #             self.box_diff = 0
    #             self.initial_box = [d2, depth]
    #             self.box_start = t
    #             self.box_jump = depth - d2
    #             return

    #     if self.holding:
    #         log.debug('~ box hold release %s %s %s  ', self.filtering, m2[self.depth_field], d2)
    #         self.holding = False
    #         yield m2, t2

    #     # otherwise we're good
    #     yield msg, t



    # v2

    def _should_hold_point(self, notnull):
        # read the buffer
        buffer = self.buffer
        *_, (m2, d2, t2), (m3, d3, t3) = buffer

        # box hold detection
        if not self.filtering:
            if d2 == 0 and d3 > 0:
                self.filtering = True
                # initial box statistics
                self.box_diff = 0
                self.initial_box = [d2, d3]
                self.box_start = t3
                self.box_jump = d3 - d2
                return True
            
            # nothing to see here
            return 
        
        # we're filtering a box
        self._filter_points()

        # sometimes the ends of boxes can leave behind a blip
        if not self.filtering:
            self._end_box()

        return False

    def _should_release_points(self, notnull):
        if not notnull:
            return 
        # we're filtering a box
        self._filter_points()
        # always release points
        return True

    def _filter_points(self):
        # read the buffer
        buffer = self.buffer
        (_, d1, _), (m2, d2, t2), (m3, d3, t3) = buffer

        # check the box statistics
        dd = d3-d2
        self.box_diff += dd
        if (
            # the relative distance change from the previous point
            abs(dd / self.box_jump) < self.box_ratio and
            # the relative distance change from the first point
            abs(self.box_diff / self.box_jump) < self.box_ratio and
            not self.is_invalid(m3, t3, t2)
        ):
            log.info('box filtered %f %f %f %s', d3, dd, self.box_diff, t3)
            if self.holding:
                self.override(m2, np.nan)
            self.override(m3, np.nan)
            return 

        # the box statistics didn't match. Stop filtering
        self.filtering = False

    def _end_box(self):
        # special handling for the end of a box
        # sometimes it leaves behind a single point
        (m3, d3, t3) = self.buffer[-1]

        if d3 / self.box_jump < self.box_ratio / 2:
            log.info('box jump down clamped %s', t3)
            self.override(m3, np.nan)
        log.info('. box filter done')