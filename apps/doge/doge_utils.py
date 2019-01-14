from settings.redis_db import database
from settings import logger
from apps.doge.doge_TA_actors import CommitteeStorage
from apps.backtesting.utils import datetime_from_timestamp
from apps.backtesting.data_sources import db_interface
from apps.portfolio.services.doge_votes import get_allocations_from_doge
from apps.genetic_algorithms.gp_artemis import ExperimentManager
import datetime
import json
from apps.doge.doge_train_test import DogeTrainer, GP_TRAINING_CONFIG

def clean_redis():
    for key in database.keys('*Doge*'):
        database.delete(key)
    for key in database.keys('*Committee*'):
        database.delete(key)


def view_keys(pattern):
    return database.keys(pattern)


def get_key_values(key):
    return database.zrange(key, 0, -1)


def list_all_committees(ticker='BTC_USDT', exchange='binance'):
    values = database.zrange(f'{ticker}:{exchange}:CommitteeStorage', 0, -1)
    logger.info('Existing committees:')
    for item in values:
        item = item.decode('UTF8').split(':')
        timestamp = CommitteeStorage.timestamp_from_score(item[-1])
        logger.info(f'  ->  at timestamp {datetime_from_timestamp(timestamp)}')

        logger.info(f'This committee produced the following allocations: '
                    f'{get_allocations_from_doge(at_datetime=datetime.datetime.utcfromtimestamp(timestamp))}')


def get_indicator_status(indicator_key='Willr', ticker='BTC_USDT', exchange='binance'):
    indicator_keys = view_keys(f'{ticker}:{exchange}:{indicator_key}*')
    for key in indicator_keys:
        last_entry = database.zrange(key, -1, -1)[0]
        timestamp = CommitteeStorage.timestamp_from_score(last_entry.decode('UTF8').split(':')[-1])
        logger.info(f'For key {key}, last entry is at {datetime_from_timestamp(timestamp)}')


class DogePerformanceTimer:

    def __init__(self):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()
        self.time_doge_performance()

    def _fill_experiment_params(self, **params):
        gp_training_config_json = self.gp_training_config_json.format(
            start_time=datetime_from_timestamp(params['start_time']),
            end_time=datetime_from_timestamp(params['end_time'])
        )

        experiment_json = json.loads(gp_training_config_json)
        for key in params:
            experiment_json[key] = params[key]
        experiment_json = json.dumps(experiment_json)
        e = ExperimentManager(experiment_container=experiment_json, read_from_file=False, database=db_interface,
                              hof_size=10)
        return e

    def time_doge_performance(self):
        import time

        end_timestamp = time.time()
        start_timestamp = end_timestamp - 60*60

        start_time = db_interface.get_nearest_db_timestamp(start_timestamp, 'BTC_USDT')
        end_time = db_interface.get_nearest_db_timestamp(end_timestamp, 'BTC_USDT')

        self._fill_experiment_params(start_time=start_time,
                                     end_time=end_time,
                                     population_sizes=[500])


