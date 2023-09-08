import os
import sys
import logging
import functools
import collections
import numpy as np
import pandas as pd
from IPython import embed

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.StreamHandler(sys.stderr))
log.setLevel(logging.WARNING)



BUFFER_SIZE = 2
MAX_DEPTH_RATE_CHANGE = 2.11667

@functools.lru_cache(10)
def warn_once(message):
    log.warning(message)


class Filter:
    def __init__(
            self, name, 
            depth_field='depth_proc_mm', 
            filter_reason_field='depth_filt_stages_applied', 
            max_mins=6, 
            buffer_size=1, 
            min_samples=0, 
            max_nan_qsize=200,
            max_hold_count=200,
            call_on_null=False,
            is_raining='precip_last_hour',
        ):
        self.buffer = collections.deque(maxlen=buffer_size)  # short time history
        # self.nan_buffer = collections.deque(maxlen=max_nan_qsize)
        self.hold_buffer = collections.deque(maxlen=max_hold_count)
        self._min_samples = min_samples or 0
        self.call_on_null = call_on_null
        self.name = name
        self.depth_field = depth_field
        self.filter_reason_field = filter_reason_field
        
        self._is_raining = is_raining
        self._precip_column = is_raining if isinstance(is_raining, str) else None

        self.max_s = max_mins*60
        self.holding = False
        # self.timestamp = 0

    def __str__(self):
        return f'{self.name}({", ".join(f"{k}={v}" for k, v in self.__get_desc__().items())})'

    def __get_desc__(self):
        return {}

    def clear(self):
        self.buffer.clear()
        self.holding = False
        # self.timestamp = 0


    # check rain

    # def is_raining(self, msg, t):
    #     if self._is_raining is True: return True
    #     if self._is_raining is False: return False
    #     if self._precip_column:
    #         if self._precip_column not in msg:
    #             warn_once(f'{self._precip_column} not in data. Assuming rain...')
    #             return True
    #         return msg[self._precip_column]
    #     return self._is_raining is None or self._is_raining(t)

    def is_invalid(self, msg, t, t1):
        invalid = self.max_s and (t-t1) > self.max_s
        if invalid:
            rain = is_raining(self._is_raining, self._precip_column, msg, t)
            invalid = invalid and rain
            self._set_reason(msg, f'?{self.name}:cancelled-rain-time-diff' if rain else f'?{self.name}:no-rain-time-diff')


    # def __call__(self, msg, t):
    #     depth, t = self._parse_message(msg, t)
    #     if self.buffer and self.buffer[-1][-1] == t:
    #         # self.buffer[-1] = (msg, depth, t)
    #         return
    #     if t == pd.Timestamp('2023-07-01 03:03:30.830000+0000', tz='UTC'):
    #         embed()
    #     # if we're holding points, buffer nans, otherwise just pass them on
    #     if np.isnan(depth):
    #         if self.holding:
    #             self.nan_buffer.append((msg, t))
    #         else:
    #             yield msg, t
    #         return
    #     self.buffer.append((msg, depth, t))
    #     # first one
    #     if self._min_samples and len(self.buffer) < self._min_samples:
    #         yield msg, t
    #         return
        
    #     for mf, tf in self._filter(msg, depth, t):
    #         # yield nans in order
    #         while self.nan_buffer:
    #             mn, tn = self.nan_buffer[0]
    #             if tn > tf:
    #                 break
    #             yield mn, tn
    #             self.nan_buffer.popleft()
    #         yield mf, tf
    #     while len(self.nan_buffer) > self.nan_buffer.maxlen - 10:
    #         yield self.nan_buffer.popleft()
            

    #     # yield from self._filter(msg, depth, t)

    def __call__(self, message, timestamp):
        # Extract the depth value from the message
        depth, timestamp = self._parse_message(message, timestamp)

        # skip redundant messages
        if self.buffer and self.buffer[-1][-1] == timestamp:
            log.warning('Duplicate message timestamp (%s: %s) received. Skipping...', depth, timestamp)
            return

        # Append the current message to the buffer
        nonnull = not pd.isna(depth)
        if nonnull: #np.isnan(depth):
            self.buffer.append((message, depth, timestamp))
        
        should_call = (nonnull or not self.call_on_null) and len(self.buffer) >= self._min_samples

        if should_call:
            self._on_new_point(nonnull)


        if self.holding:
            # Append the current message to the hold buffer
            self.hold_buffer.append((message, depth, timestamp))
            # Check if we should release the held points
            if should_call and self._should_release_points(nonnull):
                self.holding = False
                log.debug('%s release %s', self.name, str([d for _,d,_ in self.buffer]))
                # Yield the held messages with updated depth if needed
                for msg, d, t in self.hold_buffer:
                    # if t < self.timestamp:
                    #     continue
                    yield msg, t
                # remove out-dated points
                self.hold_buffer.clear()

            # at some point, we should release some points to free up resources
            elif len(self.hold_buffer) >= self.hold_buffer.maxlen - 2:
                for i in range(min(len(self.hold_buffer), 10)):
                    m, _, t = self.hold_buffer.popleft()
                    yield m, t
            return

        # Check if the filter's decision allows the point to pass or be modified
        if (
            should_call and 
            not pd.isna(depth) and
            # not np.isnan(depth) and
            len(self.buffer) >= self._min_samples and
            self._should_hold_point(nonnull)
        ):
            log.debug('%s hold %s', self.name, str([d for _,d,_ in self.buffer]))
            self.holding = True
            self.hold_buffer.append((message, depth, timestamp))
        else:
            # Yield the non-held message with the original depth value
            yield message, timestamp

    def _filter(self, msg, depth, t):
        yield msg, t


    def _on_new_point(self, nonnull):
        pass

    def _should_hold_point(self, nonnull):
        return False

    def _should_release_points(self, nonnull):
        return True


    # getters

    def _parse_message(self, msg, t):
        return msg[self.depth_field], t


    # modifications

    def override(self, msg, depth, reason=None):
        # print("overriding", msg[self.depth_field], depth, reason)
        # input()
        msg[self.depth_field] = depth;
        self._set_reason(msg, reason or self.name)
        return msg

    def _set_reason(self, msg, reason):
        if reason:
            reasons = (msg.get(self.filter_reason_field) or '').split('|')
            reasons = unique_list(reasons + [reason], {''})
            msg[self.filter_reason_field] = '|'.join(reasons)
        return msg
    

    # utils

    def log(self, *a):
        log.info(*a)


def unique_list(seq, seen=None):
    seen = seen or set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def is_raining(is_raining, precip_column, msg, t):
    # hardcoded rain answers
    if is_raining is True: return True
    if is_raining is False: return False
    # rain status is inside the message
    if precip_column:
        if precip_column not in msg:
            warn_once(f'{precip_column} not in data. Assuming rain...')
            return True
        return msg[precip_column]
    # call an external function
    return is_raining is None or is_raining(t)