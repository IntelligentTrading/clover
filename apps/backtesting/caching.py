import inspect

import cloudpickle

from apps.backtesting.config import CACHE_MODE, CACHE_MODE_DICTIONARY, ENABLE_BACKTEST_CACHE, CACHE_MODE_REDIS
from apps.backtesting.evaluation import Evaluation


if CACHE_MODE_REDIS:
    from apps.backtesting.config import redis_instance


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class memoize(object):
    def __init__(self, cls):
        self.cls = cls

        self.default_args = get_default_args(cls.__init__)
        for base in cls.__bases__:
            self.default_args = dict(self.default_args, **get_default_args(base))

        # This bit allows staticmethods to work as you would expect. (code adapted from StackOverflow)
        for attr, val in cls.__dict__.items():
            if type(val) is staticmethod:
                self.__dict__[attr] = val.__func__
        if CACHE_MODE == CACHE_MODE_DICTIONARY:
            self.data = {}

    def _process_redis(self, key, *args, **kwargs):
        if not redis_instance.exists(key):
            evaluation = self.cls(*args, **kwargs)
            redis_instance.set(key, cloudpickle.dumps(evaluation))
            return evaluation
        return cloudpickle.loads(redis_instance.get(key))

    def _process_dict(self, key, *args, **kwargs):
        if not key in self.data:
            evaluation = self.cls(*args, **kwargs)
            self.data[key] = evaluation
            return evaluation
        return self.data[key]

    def __call__(self, *args, **kwargs):
        if not ENABLE_BACKTEST_CACHE:
            return self.cls(*args, **kwargs)

        kwargs = dict(self.default_args, **kwargs)

        key = Evaluation.signature_key(**kwargs)
        if CACHE_MODE == CACHE_MODE_REDIS:
            return self._process_redis(key, *args, **kwargs)
        elif CACHE_MODE == CACHE_MODE_DICTIONARY:
            return self._process_dict(key, *args, **kwargs)