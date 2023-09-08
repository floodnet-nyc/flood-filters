

def test(dep_id, start, end):
    pass

import fire
from tqdm.contrib.logging import logging_redirect_tqdm
with logging_redirect_tqdm():
    # log.setLevel(logging.INFO)
    fire.Fire(test)
