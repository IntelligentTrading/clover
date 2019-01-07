import logging

from apps.TA.storages.abstract.key_value import KeyValueStorage
from apps.TA.storages.abstract.ticker import TickerStorage
from apps.backtesting.data_sources import db_interface
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.tick_provider_heartbeat import TickProviderHeartbeat
from apps.backtesting.tick_provider import TickerData
from apps.backtesting.utils import datetime_from_timestamp
from apps.genetic_algorithms.genetic_program import GeneticTickerStrategy
from apps.genetic_algorithms.gp_artemis import ExperimentManager
from apps.genetic_algorithms.leaf_functions import RedisTAProvider
from apps.TA import PERIODS_1HR
from settings import DOGE_RETRAINING_PERIOD_SECONDS
import time

METRIC_IDS = {
    'mean_profit': 0,
}
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))

GP_TRAINING_CONFIG = os.path.join(BASE, 'doge_config.json')


class DogeRedisEntry:

    HASH_DIGITS_TO_KEEP = 8

    def __init__(self, train_end_timestamp, doge_str, metric_id, metric_value, rank):
        self.train_end_timestamp = train_end_timestamp
        self.doge_str = doge_str
        self.metric_id = metric_id
        self.metric_value = metric_value
        self.rank = rank  # TODO remove rank information if not needed

    @property
    def hash(self):
        return hash(self.doge_str) % 10 ** self.HASH_DIGITS_TO_KEEP

    def save_to_storage(self):
        # save the doge itself
        new_doge_storage = DogeStorage(value=self.doge_str, key_suffix=str(self.hash))
        new_doge_storage.save()

        # save performance info
        new_doge_performance_storage = DogePerformance(key_suffix=f'{str(self.hash)}:{self.metric_id}',
                                                       ticker='BTC_USDT',
                                                       exchange='binance',
                                                       timestamp=self.train_end_timestamp,
                                                       value=self.metric_value)
        new_doge_performance_storage.save()


class DogeStorage(KeyValueStorage):

    # self.value = string_format_of_entire_decision_tree
    # self.db_key_prefix = "ticker:exchange" # but we don't care, so don't distinguish!
    # self.db_key_suffix = str(hash(self.value))[:8] #last 8 chars of the hash

    pass


class DogePerformance(TickerStorage):
    """
        defines the performance score for a doge over time, unique per ticker
    """
    # self.key_suffix = doge_id
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = performance_score
    # self.timestamp


class CommitteeStorage(TickerStorage):
    """
        defines which doges are valid for voting in the committee at the timestamp
    """
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = str(doge_id_list)
    # self.timestamp = timestamp



class DogeTrainer:
    """
    A class that encapsulates GP training.
    """

    def __init__(self, database):
        """

        :param database: database from data_sources, either Redis or Postgres
        """
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()
        self.database = database

    def retrain_doges(self, start_timestamp, end_timestamp, max_doges_to_save=10):
        """
        Reruns doge training and saves results to the database.
        :param start_timestamp: starting time to use for training (tickers are specified in GP_TRAINING_CONFIG json)
        :param end_timestamp: end time to use for training (tickers are specified in GP_TRAINING_CONFIG json)
        :param max_doges_to_save: maximum number of GPs to save in the database
        :return:
        """
        config_json = self.fill_json_template(self.gp_training_config_json, start_timestamp, end_timestamp)

        logging.info('>>>>>>> Starting GP training... ')
        logging.info(f'    >>> start_time = {datetime_from_timestamp(start_timestamp)}')
        logging.info(f'    >>> end_time = {datetime_from_timestamp(end_timestamp)}')

        # create an experiment manager
        e = ExperimentManager(experiment_container=config_json, read_from_file=False, database=self.database,
                              hof_size=10)  # we will have one central json with all the parameters

        # run experiments
        e.run_experiments(keep_record=True)

        logging.info('>>>>>>> GP training completed.')

        # retrieve best performing doges
        logging.info('>>>>>>> Ranking doges by performance...')
        doge_df = e.get_best_performing_across_variants_and_datasets(datasets=e.training_data,
                                                                     sort_by=['mean_profit'],
                                                                     top_n_per_variant=1)
        logging.info('>>>>>>> Ranking completed.')

        # write these doges to database
        logging.info('>>>>>>> Saving GPs to database...')

        string_representations = []
        redis_entries = []
        for i, row in enumerate(doge_df.itertuples()):
            if i > max_doges_to_save:
                break
            redis_entries.append(
                DogeRedisEntry(train_end_timestamp=end_timestamp, doge_str=str(row.doge),
                               metric_id=METRIC_IDS['mean_profit'], metric_value=row.mean_profit, rank=i))


        # save individual doges
        self._save_doges(redis_entries)

        # save current committee
        committee_str = ":".join(map(str, [redis_entry.hash for redis_entry in redis_entries]))
        new_committee_storage = CommitteeStorage(timestamp=end_timestamp, ticker='BTC_USDT', exchange='binance')
                                                                          # TODO remove this hardcoding once more tickers are supported
        new_committee_storage.value = committee_str
        new_committee_storage.save(publish=True)
        logging.info('>>>>>>> GPs saved to database.')

    def _save_doges(self, redis_entries):
        for redis_entry in redis_entries:
            redis_entry.save_to_storage()

    @staticmethod
    def fill_json_template(gp_training_config_json, start_timestamp, end_timestamp):
        """
        Fills the json training template with starting and ending times.
        :param gp_training_config_json: string json describing the training process (loaded from GP_TRAINING_CONFIG file)
        :param start_timestamp: training start time
        :param end_timestamp: training end time
        :return:
        """
        return gp_training_config_json.format(
            start_time=datetime_from_timestamp(start_timestamp),
            end_time=datetime_from_timestamp(end_timestamp)
        )

    @staticmethod
    def run_training(start_timestamp, end_timestamp):
        """
        Instantiates a DogeTrainer and reruns training.
        :param start_timestamp:
        :param end_timestamp:
        :return:
        """

        trainer = DogeTrainer(db_interface)

        start_time = db_interface.get_nearest_db_timestamp(start_timestamp, 'BTC', 'USDT')
        end_time = db_interface.get_nearest_db_timestamp(end_timestamp, 'BTC', 'USDT')

        trainer.retrain_doges(start_time, end_time, max_doges_to_save=10)


class DogeTrader:
    """
    A class that encapsulates a single Doge that trades.
    NOTE: instantiated in DogeCommittee, no need to manually instantiate
    """

    def __init__(self, doge_str, doge_id, function_provider, gp_training_config_json):
        """
        Instantiates a doge trader.
        :param doge_str: the decision tree in string format
        :param function_provider: an instance of the TAProvider class that provides TA values
        :param gp_training_config_json: a string representation of the training config json (loaded from GP_TRAINING_CONFIG file)
        """

        self.doge_str = doge_str
        self.hash = doge_id
        self.gp_training_config_json = gp_training_config_json
        experiment_json = DogeTrainer.fill_json_template(self.gp_training_config_json, 0, 0)
        self.doge, self.gp = ExperimentManager.resurrect_better_doge(experiment_json, self.doge_str, function_provider)
        self.strategy = GeneticTickerStrategy(tree=self.doge, gp_object=self.gp)

    def vote(self, ticker_data):
        """
        :param ticker_data: an instance of TickerData class, containing the ticker info and optionally OHLCV data and signals
        :return: an instance of StrategyDecision class
        """
        return self.strategy.process_ticker(ticker_data)

    def weight_at_timestamp(self, timestamp=None, metric_id=0):
        result = DogePerformance.query(key_suffix=f'{str(self.hash)}:{metric_id}',
                                                   ticker='BTC_USDT',
                                                   exchange='binance',
                                                   timestamp=timestamp)
        return float(result['values'][-1])


class DogeCommittee:
    """
    A class that encapsulates trading using a committee of GPs.
    The committee is built out of the latest GPs in the database.
    """

    def __init__(self, database=db_interface, max_doges=100, ttl=DOGE_RETRAINING_PERIOD_SECONDS):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()

        self.max_doges = max_doges
        self.function_provider = RedisTAProvider()
        doge_traders = self._load_latest_doge_traders()
        self.doge_traders = doge_traders if len(doge_traders) <= max_doges else doge_traders[:max_doges]
        self.periods = PERIODS_1HR  # TODO remove this hardcoding if we decide to use more horizons
        self._init_time = time.time()
        self._ttl = ttl

    @property
    def expired(self):
        return time.time() - self._init_time > self._ttl

    def _load_latest_doge_traders(self):
        """
        Loads latest doge traders from the database.
        :return: a list of DogeTrader objects
        """
        doge_traders = []

        # get doges out of DB
        # get the latest committee
        query_response = CommitteeStorage.query(ticker='BTC_USDT', exchange='binance')
        self.committee_timestamp = CommitteeStorage.timestamp_from_score(query_response['scores'][-1])
        doge_committee_ids = query_response['values'][-1].split(':')
        for doge_id in doge_committee_ids:
            doge_storage = DogeStorage(key_suffix=doge_id)
            doge_str = doge_storage.get_value().decode('utf-8')

            doge = DogeTrader(doge_str=doge_str, doge_id=doge_id,
                              function_provider=self.function_provider,
                              gp_training_config_json=self.gp_training_config_json)
            doge_traders.append(doge)

        return doge_traders

    def vote(self, transaction_currency, counter_currency, timestamp, source='binance', resample_period=5):
        """
        Produces votes of all committee members for the given ticker at timestamp.
        :param transaction_currency: transaction currency
        :param counter_currency: counter currency
        :param timestamp: timestamp
        :param source: exchange
        :param resample_period: resampling period in minutes
        :return: a list of votes (+1=buy, -1=sell, 0=ignore) and a list of
                 unnormalized weights of doges that produced the votes
        """
        ticker_data = TickerData(
            timestamp=timestamp,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            source=source,
            resample_period=resample_period,
            open_price=None, # TODO fill if needed
            high_price=None,
            low_price=None,
            close_price=None,
            close_volume=None,
            signals=[],
        )

        votes = []
        weights = []

        for i, doge in enumerate(self.doge_traders):
            decision = doge.vote(ticker_data)
            weight = doge.weight_at_timestamp(timestamp=self.committee_timestamp)
            print(f'  Doge {i} says: {str(decision)} (its weight is {weight:.2f})')
            votes.append(decision.outcome)
            weights.append(weight)

        return votes, weights



class DogeTradingManager(TickListener):
    """
    A class that initializes a heartbeat tick provider that periodically polls Redis for ticker prices and
    trades on them using a DogeCommittee.
    """

    def __init__(self, database=db_interface, heartbeat_period_secs=60):

        self.doge_committee = DogeCommittee(database)

        tick_provider_heartbeat = TickProviderHeartbeat(
            heartbeat_period_secs=heartbeat_period_secs,
            database=database,
            ticker_list=['BTC_USDT']
        )
        tick_provider_heartbeat.add_listener(self)
        tick_provider_heartbeat.run()


    def process_event(self, ticker_data):
        """
        This method will be called every heartbeat_period_secs seconds for all the tickers on which the trader
        is subscribed.
        :param ticker_data: an instance of TickerData (OHLCV for a ticker at timestamp)
        :return:
        """
        logging.info(f'So wow! Price for {ticker_data.transaction_currency}-{ticker_data.counter_currency} '
              f'arrived ({datetime_from_timestamp(ticker_data.timestamp)})')

        votes, weights = self.doge_committee.vote(ticker_data.transaction_currency,
                                                  ticker_data.counter_currency,
                                                  ticker_data.timestamp)

        weights = [w if w != 0 else w+0.0001 for w in weights]  # TODO: smarter way to check for weights 0?

        weighted_votes = [weight*votes[i] for i, weight in enumerate(weights)]

        logging.info(f'End decision for {ticker_data.transaction_currency}-{ticker_data.counter_currency} '
              f'at {datetime_from_timestamp(ticker_data.timestamp)}: {sum(weighted_votes)/sum(weights)}\n\n')

    def broadcast_ended(self):
        logging.info('Doges have spoken.')