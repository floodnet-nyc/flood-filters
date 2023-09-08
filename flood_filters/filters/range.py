from ..filter import Filter, log


class RangeFilter(Filter):
    '''Depth values outside the range [10 mm, d_max mm] are zeroed'''
    def __init__(self, height=None, noise_floor=10, name='range'):
        super().__init__(name)
        self.noise = noise_floor
        self.height = height

    def __get_desc__(self):
        return {'noise': self.noise, 'height': self.height}

    # def _filter(self, msg, depth, t):
    #     night_median = msg.get('night_median')
    #     if depth > 0 and depth <= self.noise:
    #         self.override(msg, 0, f'{self.name}:noise-floor')
    #     if self.height and depth > self.height:
    #         self.override(msg, None, f'{self.name}:max-height')
    #     if night_median and depth > night_median:
    #         self.override(msg, None, f'{self.name}:night-median')
    #     yield msg, t


    def _should_hold_point(self, notnull):
        # read the buffer
        (m, d, t) = self.buffer[-1]

        # min height
        if d > 0 and d <= self.noise:
            self.override(m, 0, f'{self.name}:noise-floor')
        
        # max height
        max_height = m.get('median_height_mm')
        if self.height and d > self.height:
            self.override(m, None, f'{self.name}:max-height')
        if max_height and d > max_height:
            self.override(m, None, f'{self.name}:max-height')

        # we never need to hold a point
        return False

    def _should_release_points(self):
        return True  # we never need to hold a point
