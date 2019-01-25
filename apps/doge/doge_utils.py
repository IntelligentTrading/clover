from settings.redis_db import database
from settings import logger
from apps.backtesting.utils import datetime_from_timestamp
from apps.backtesting.data_sources import db_interface
from apps.genetic_algorithms.gp_artemis import ExperimentManager
import json
from apps.doge.doge_train_test import GP_TRAINING_CONFIG, DogeCommittee
import time
import pandas as pd
import pickle
from apps.doge.doge_TA_actors import CommitteeStorage

def clean_redis():
    for key in database.keys('*Doge*'):
        database.delete(key)
    for key in database.keys('*Committee*'):
        database.delete(key)


def view_keys(pattern):
    return database.keys(pattern)


def get_key_values(key):
    return database.zrange(key, 0, -1)


def load_committees_in_period(ticker, exchange, start_timestamp, end_timestamp):
    key = f'{ticker}:{exchange}:CommitteeStorage'
    timestamps = [CommitteeStorage.timestamp_from_score(score.decode('utf8').split(':')[-1]) for score in
        database.zrangebyscore(key, min=CommitteeStorage.score_from_timestamp(start_timestamp),
                               max=CommitteeStorage.score_from_timestamp(end_timestamp))]
    committees = [DogeCommittee(committee_timestamp=timestamp) for timestamp in timestamps]
    return committees


class DogePerformanceTimer:

    def __init__(self, run_variants_in_parallel=False):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()
        self.run_variants_in_parallel = run_variants_in_parallel
        self.time_doge_performance()

    def _build_experiment_manager(self, use_cached_redis, **params):

        if use_cached_redis:
            transaction_currency, counter_currency = params['ticker'].split('_')

            from apps.backtesting.data_sources import CachedRedis
            cached_redis = CachedRedis(start_time=params['start_time'], end_time=params['end_time'],
                                       transaction_currency=transaction_currency, counter_currency=counter_currency,
                                       horizon=None)
            database = cached_redis
        else:
            database = db_interface

        gp_training_config_json = self.gp_training_config_json.format(
            ticker=params['ticker'],
            start_time=datetime_from_timestamp(params['start_time']),
            end_time=datetime_from_timestamp(params['end_time'])
        )

        experiment_json = json.loads(gp_training_config_json)
        for key in params:
            experiment_json[key] = params[key]
        experiment_json = json.dumps(experiment_json)
        return ExperimentManager(experiment_container=experiment_json, read_from_file=False, database=database,
                                 hof_size=10, parallel_run=self.run_variants_in_parallel)

    def time_doge_performance(self, use_cached_redis=True):

        ENTRIES_CACHE_FILENAME = 'entries.p'
        import os
        if os.path.exists(ENTRIES_CACHE_FILENAME):
            entries = pickle.load(open(ENTRIES_CACHE_FILENAME, 'rb'))
        else:
            entries = []

        training_periods_secs = [60*60,      # 1 hour
                                60*60*4,     # 4 hours
                                60*60*24,    # 24 hours
                                60*60*24*3]  # 3 days

        population_sizes = [50, 100, 200, 500]
        num_generations = [5, 10, 50, 100]

        for training_period in training_periods_secs:
            for generations in num_generations:
                for population_size in population_sizes:
                    exists = False
                    for entry in entries:
                        if entry['training_period'] == training_period \
                                and entry['generations'] == generations \
                                and entry['population_size'] == population_size:
                            exists = True
                            break
                    if exists:
                        logger.info('Entry exists, skipping...')
                        continue
                    end_timestamp = time.time()
                    start_timestamp = end_timestamp - training_period

                    start_time = db_interface.get_nearest_db_timestamp(start_timestamp, 'BTC_USDT')
                    end_time = db_interface.get_nearest_db_timestamp(end_timestamp, 'BTC_USDT')

                    e = self._build_experiment_manager(use_cached_redis=use_cached_redis,
                                                       ticker='BTC_USDT',
                                                       start_time=start_time,
                                                       end_time=end_time,
                                                       population_sizes=[population_size],
                                                       num_generations=generations,
                                                       mating_probabilities=[0.7],  # ensure only one variant is tested
                                                       mutation_probabilities=[0.8]  # ensure only one variant is tested
                                                       )
                    tick = time.time()
                    e.run_experiments()
                    tock = time.time()

                    duration = tock - tick
                    entry = {
                        'training_period': training_period,
                        'population_size': population_size,
                        'generations': generations,
                        'duration': duration
                    }
                    entries.append(entry)
                    pickle.dump(entries, open(ENTRIES_CACHE_FILENAME, 'wb'))

                    entries_txt_file = ENTRIES_CACHE_FILENAME.split('.')[0] + '.txt'
                    with open(entries_txt_file, 'w') as f:
                        f.writelines(map(str, entries))

        entries = pickle.load(open(ENTRIES_CACHE_FILENAME, 'rb'))
        df = pd.DataFrame(entries)

