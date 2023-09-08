# import numpy as np
# from ..filter import Filter, log


# class DistFilter(Filter):
#     '''Depth values outside the range [10 mm, d_max mm] are zeroed'''
#     def __init__(self, max_reading=[5000, 9999], min_reading=300, name='dist'):
#         super().__init__(name)
#         self._max_readings = list(max_reading)
#         self.min_reading = min_reading
#         self.clear()

#     def clear(self):
#         super().clear()
#         self.max_readings = max_reading = list(self._max_readings)
#         self.max_reading = max_reading.pop(0)

#     def __get_desc__(self):
#         return {'max': self.max_reading}

#     def _filter(self, msg, depth, t):
#         # recognize if we have a 9999 sensor instead of 5000
#         if depth > self.max_reading and self.max_readings:
#             self.max_reading = self.max_readings.pop(0)

#         # no return
#         if depth >= self.max_reading:
#             self.override(msg, np.nan, f'{self.name}:no-return')
#         # below minimum reading capable from sensor
#         if depth < self.min_reading:
#             self.override(msg, np.nan, f'{self.name}:below-min-reading')
#         yield msg, t
