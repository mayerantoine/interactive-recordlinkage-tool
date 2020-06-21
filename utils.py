import time
import logging
from functools import wraps

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        logging.info("@timefn: %s took  %s  seconds",fn.__name__,str(t2 - t1))
        return result
    return measure_time
