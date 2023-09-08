import functools
import contextlib
from pyinstrument import Profiler

PROFILER = None

@contextlib.contextmanager
def profile():
    global PROFILER
    prev = PROFILER
    if prev is not None:
        prev.stop()

    PROFILER = Profiler()
    PROFILER.start()
    try:
        yield
    finally:
        PROFILER.stop()
        PROFILER.print()
        if prev is not None:
            prev.start()
        PROFILER = prev

def profile_func(func):
    @functools.wraps(func)
    def wrapper(*a, **kw):
        with profile():
            return func(*a, **kw)
    return wrapper
