import json
import logging
from datetime import datetime
import numpy as np
from apps.TA import TAException, JAN_1_2017_TIMESTAMP
from apps.TA.storages.abstract.key_value import KeyValueStorage
from settings.redis_db import database

logger = logging.getLogger(__name__)


class StorageException(TAException):
    pass


class TimeseriesException(TAException):
    pass


class TimeseriesStorage(KeyValueStorage):
    """
    stores things in a sorted set unique to each ticker and exchange
    """
    class_describer = "timeseries"
    value = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 'timestamp' REQUIRED, VALIDATE
        try:
            self.unix_timestamp = int(kwargs['timestamp'])  # int eg. 1483228800
        except KeyError:
            raise TimeseriesException("timestamp required for TimeseriesStorage objects")
        except ValueError:
            raise TimeseriesException(
                "timestamp must be castable as integer, received {ts}".format(
                    ts=kwargs.get('timestamp')))
        except Exception as e:
            raise StorageException(str(e))

        if self.unix_timestamp < JAN_1_2017_TIMESTAMP:
            raise TimeseriesException("timestamp before January 1st, 2017")

    def save_own_existance(self, describer_key=""):
        self.describer_key = describer_key or f'{self.__class__.class_describer}:{self.get_db_key()}'

    @classmethod
    def score_from_timestamp(cls, timestamp) -> float:
        return round((float(timestamp) - JAN_1_2017_TIMESTAMP) / 300, 3)

    @classmethod
    def timestamp_from_score(cls, score) -> int:
        return int(float(score) * 300) + JAN_1_2017_TIMESTAMP

    @classmethod
    def datetime_from_score(cls, score) -> datetime:
        return datetime.fromtimestamp(cls.timestamp_from_score(score))

    @classmethod
    def periods_from_seconds(cls, seconds) -> float:
        return float(seconds) / 300

    @classmethod
    def seconds_from_periods(cls, periods) -> int:
        return int(float(periods) * 300)

    @classmethod
    def query(cls, key: str = "", key_suffix: str = "", key_prefix: str = "",
              timestamp: int = None,
              timestamp_tolerance: int = 299,
              periods_range: float = 0.01,
              *args, **kwargs) -> dict:
        """
        :param key: the exact redis sortedset key (optional)
        :param key_suffix: suffix on the key  (optional)
        :param key_prefix: prefix on the key (optional)
        :param timestamp: timestamp for most recent value returned  (optional, default returns latest)
        :param periods_range: number of periods desired in results (optional, default 0, so only return 1 value)
        :param timestamp_tolerance: tolerance in seconds on results within timestamp and period range (optional, defualt=299)
        :return: dict(values=[], ...)
        """

        sorted_set_key = cls.compile_db_key(key=key, key_prefix=key_prefix, key_suffix=key_suffix)
        # logger.debug(f'query for sorted set key {sorted_set_key}')
        # example key f'{key_prefix}:{cls.__name__}:{key_suffix}'

        # do a quick check to make sure this is a class of things we know is in existence
        describer_key = f'{cls.class_describer}:{sorted_set_key}'
        # if no timestamp, assume query to find the most recent, the last one

        if not timestamp:
            query_response = database.zrange(sorted_set_key, -1, -1)
            try:
                [value, score] = query_response[0].decode("utf-8").split(":")
                timestamp = cls.timestamp_from_score(score)
            except:
                timestamp = JAN_1_2017_TIMESTAMP

            min_timestamp = max_timestamp = timestamp


        else:
            # compress timestamps to scores
            target_score = cls.score_from_timestamp(timestamp)
            score_tolerance = cls.periods_from_seconds(timestamp_tolerance)

            # logger.debug(f"querying for key {sorted_set_key} \
            # with score {target_score} and back {periods_range} periods")

            min_score, max_score = (target_score - score_tolerance - periods_range), (target_score + score_tolerance)
            min_timestamp,  max_timestamp = cls.timestamp_from_score(min_score), cls.timestamp_from_score(max_score)

            query_response = database.zrangebyscore(sorted_set_key, min_score, max_score)
            # example query_response = [b'0.06288:210156.0']
            # which came from f'{self.value}:{str(score)}' where score = (self.timestamp-JAN_1_2017_TIMESTAMP)/300

        return_dict = {
            'values': [],
            'values_count': 0,
            'timestamp': timestamp,
            'earliest_timestamp': min_timestamp,
            'latest_timestamp': max_timestamp,
            'periods_range': periods_range,
            'period_size': 300,
            'scores': []
        }

        if not len(query_response):
            return return_dict

        try:
            return_dict['values_count'] = len(query_response)

            if len(query_response) < periods_range + 1:
                return_dict["warning"] = "fewer values than query's periods_range"

            values = [value_score.decode("utf-8").rsplit(":", 1)[0] for value_score in query_response]
            scores = [value_score.decode("utf-8").rsplit(":", 1)[1] for value_score in query_response]

            # todo: double check that [-1] in list is most recent timestamp

            values = [value_score.decode("utf-8")[:value_score.decode("utf-8").rfind(':')] for value_score in query_response]
            scores = [value_score.decode("utf-8")[value_score.decode("utf-8").rfind(':')+1:] for value_score in query_response]

            return_dict.update({
                'values': values,
                'scores': scores,
                'earliest_timestamp': cls.timestamp_from_score(float(scores[0])),
                'latest_timestamp': cls.timestamp_from_score(float(scores[-1])),
            })
            return return_dict

        except IndexError:
            return return_dict

        except Exception as e:
            logger.error("redis query problem: " + str(e))
            return {'error': "redis query problem: " + str(e),  # wtf happened?
                    'values': []}

    @staticmethod
    def get_values_array_from_query(query_results: dict, limit: int = 0):

        value_array = [float(v) for v in query_results['values']]

        if limit:
            if not isinstance(limit, int) or limit < 1:
                raise TimeseriesException(f"bad limit: {limit}")

            elif len(value_array) > limit:
                value_array = value_array[-limit:]

        return np.array(value_array)

    def get_z_add_data(self):
        z_add_key = f'{self.get_db_key()}'  # set key name
        z_add_score = f'{self.score_from_timestamp(self.unix_timestamp)}'  # timestamp as score (int or float)
        z_add_name = f'{self.value}:{z_add_score}'  # item unique value
        z_add_data = {"key": z_add_key, "name": z_add_name, "score": z_add_score}  # key, score, name
        return z_add_data

    def save(self, publish=False, pipeline=None, *args, **kwargs):

        if self.value is None:
            logger.warning("no value was set, nothing to save, but will save empty string anyway")

        self.value = self.value or ""  # use empty string for None, False, etc

        if not self.force_save:
            # validate some rules here?
            pass

        self.save_own_existance()  # todo: is this still necessary?

        z_add_data = self.get_z_add_data()
        # # logger.debug(f'savingdata with args {z_add_data}')

        if pipeline is not None:
            pipeline = pipeline.zadd(*z_add_data.values())
            # logger.debug("added command to redis pipeline")
            if publish: pipeline = self.publish(pipeline)
            return pipeline
        else:
            response = database.zadd(*z_add_data.values())
            # logger.debug("no pipeline, executing zadd command immediately.")
            if publish: self.publish()
            return response

    def publish(self, pipeline=None):
        if pipeline:
            return pipeline.publish(self.__class__.__name__, json.dumps(self.get_z_add_data()))
        else:
            return database.publish(self.__class__.__name__, json.dumps(self.get_z_add_data()))

    def get_value(self, *args, **kwargs):
        TimeseriesException("function not yet implemented! ¯\_(ツ)_/¯ ")
        pass


"""
We can scan the newest or oldest event ids with ZRANGE 4,
maybe later pulling the events themselves for analysis.

We can get the 10 or even 100 events immediately
before or after a timestamp with ZRANGEBYSCORE
combined with the LIMIT argument.

We can count the number of events that occurred
in a specific time period with ZCOUNT.

https://www.infoq.com/articles/redis-time-series
"""
