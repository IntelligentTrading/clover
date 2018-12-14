import datetime
import logging
import time
from apps.backtesting.config import POOL_SIZE
from pathos.multiprocessing import Pool
import tqdm
from dateutil import parser


def datetime_from_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y/%m/%d %H:%M:%S UTC')


def datetime_to_timestamp(datetime_str):
    return parser.parse(datetime_str).timestamp()


def get_distinct_signal_types(signals):
    return set([x.signal_signature if x is not None else '(no signals)' for x in signals])


def time_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end-start:.4f} seconds")
        return result

    return wrapper


def parallel_run(func, param_list, pool_size=POOL_SIZE):
    with Pool(pool_size) as pool:
        #results = pool.map(func, param_list)
        results = list(tqdm.tqdm(pool.imap(func, param_list), total=len(param_list)))
        pool.close()
        pool.join()
        pool.terminate() # needed for Pathos,
#        pool.restart()   # see https://stackoverflow.com/questions/49888485/pathos-multiprocessings-pool-appears-to-be-nonlocal
    return results

def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class LogDuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv