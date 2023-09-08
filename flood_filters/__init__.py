import logging
log = logging.getLogger(__name__)
from .filter import *
from .filters import *
from .filter_bank import *


FILTER_BANKS = {
    'none': lambda: FilterBank([]),
    'grad': lambda: FilterBank([
        RangeFilter(),
        GradFilter(),
    ]),
    'grad+blip+box': lambda: FilterBank([
        RangeFilter(),
        GradFilter(),
        BlipFilter(is_raining=is_raining),
        BoxFilter(is_raining=is_raining),
    ]),
    'grad+blip+box+blip': lambda: FilterBank([
        RangeFilter(),
        GradFilter(),
        BlipFilter(),
        BoxFilter(),
        BlipFilter(),
    ]),
}

DEFAULT_FBANK = 'grad+blip+box+blip'

def get_filterbank(name=DEFAULT_FBANK):
    if name is True:
        name = DEFAULT_FBANK
    if name is None: 
        return None
    if isinstance(name, FilterBank): 
        return name
    if isinstance(name, Filter): 
        return FilterBank([Filter])
    return FILTER_BANKS[name]()