import logging

from apps.backtesting.data_sources import db_interface
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.tick_provider_heartbeat import TickProviderHeartbeat
from apps.backtesting.tick_provider import TickerData
from apps.backtesting.utils import datetime_from_timestamp
from apps.doge.doge_TA_actors import DogeStorage, DogePerformance, CommitteeStorage, SignalSubscriber, \
    CommitteeVoteStorage
from apps.genetic_algorithms.genetic_program import GeneticTickerStrategy
from apps.genetic_algorithms.gp_artemis import ExperimentManager
from apps.genetic_algorithms.leaf_functions import RedisTAProvider
from apps.TA import PERIODS_1HR
from settings import DOGE_RETRAINING_PERIOD_SECONDS, DOGE_TRAINING_PERIOD_DURATION_SECONDS, \
    logger, SUPPORTED_DOGE_TICKERS, DOGE_LOAD_ROCKSTARS
import time
from apps.genetic_algorithms.chart_plotter import save_dot_graph
from apps.backtesting.utils import datetime_from_timestamp

METRIC_IDS = {
    'mean_profit': 0,
}
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))

GP_TRAINING_CONFIG = os.path.join(BASE, 'doge_config.json')


class DogeRecord:

    def __init__(self, train_end_timestamp, doge_str, metric_id, metric_value, rank):
        self.train_end_timestamp = train_end_timestamp
        self.doge_str = doge_str
        self.metric_id = metric_id
        self.metric_value = metric_value
        self.rank = rank  # TODO remove rank information if not needed

    @property
    def hash(self):
        return DogeStorage.hash(self.doge_str)

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

        # DEBUG: loading rockstars
        rockstars = []
        if DOGE_LOAD_ROCKSTARS:
            from settings import DOGE_MAX_ROCKSTARS
            rockstars = CommitteeStorage.load_rockstars(num_previous_committees_to_search=2, max_num_rockstars=5,
                                                        ticker='BTC_USDT', exchange='binance', timestamp=end_timestamp)
            logging.info(f'Loaded {len(rockstars)} rockstars.')


        # create an experiment manager
        e = ExperimentManager(experiment_container=config_json, read_from_file=False, database=self.database,
                              hof_size=10, rockstars=rockstars)  # we will have one central json with all the parameters

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
                DogeRecord(train_end_timestamp=end_timestamp, doge_str=str(row.doge),
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
        for i, redis_entry in enumerate(redis_entries):
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

        start_time = db_interface.get_nearest_db_timestamp(start_timestamp, 'BTC_USDT')
        end_time = db_interface.get_nearest_db_timestamp(end_timestamp, 'BTC_USDT')

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

    def save_doge_img(self, out_filename, format='svg'):
        print(self.doge_str)
        return save_dot_graph(self.doge, out_filename, format)


class DogeCommittee:
    """
    A class that encapsulates trading using a committee of GPs.
    The committee is built out of the latest GPs in the database.
    """

    def __init__(self, committee_timestamp=None, max_doges=100, ttl=DOGE_RETRAINING_PERIOD_SECONDS):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()

        self.max_doges = max_doges
        self._committee_timestamp = committee_timestamp  # uses the last committee if timestamp is None
        self.function_provider = RedisTAProvider()
        doge_traders = self._load_doge_traders()
        self.doge_traders = doge_traders if len(doge_traders) <= max_doges else doge_traders[:max_doges]
        self.periods = PERIODS_1HR  # TODO remove this hardcoding if we decide to use more horizons
        self._ttl = ttl

    def expired(self, at_timestamp):
        return at_timestamp - self._committee_timestamp > self._ttl

    def _load_doge_traders(self):
        """
        Loads doge traders that belong to this committee (based on timestamp, self._init_time).
        :return: a list of DogeTrader objects
        """
        doge_traders = []

        # get doges out of DB
        # get the latest committee
        query_response = CommitteeStorage.query(ticker='BTC_USDT', exchange='binance', timestamp=self._committee_timestamp)
        self.committee_timestamp = CommitteeStorage.timestamp_from_score(query_response['scores'][-1])
        assert self._committee_timestamp == self.committee_timestamp or self._committee_timestamp is None # for debugging TODO: decide on how to make the distinction

        if not query_response['values']:
            raise Exception('No committee members found for timestamp {self.committee_timetstamp}!')

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

    def generate_doge_images(self):
        for i, doge in enumerate(self.doge_traders):
            doge.save_doge_img(out_filename=f'apps/doge/static/{i}')


class DogeTradingManager(TickListener):
    """
    A class that initializes a heartbeat tick provider that periodically polls Redis for ticker prices and
    trades on them using a DogeCommittee.
    """

    def __init__(self, database=db_interface, heartbeat_period_secs=60):

        self.doge_committee = DogeCommittee()
        self.doge_committee.generate_doge_images()

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


class DogeSubscriber(SignalSubscriber):
    storage_class = CommitteeVoteStorage  # override with applicable storage class

    def __init__(self, *args, **kwargs):
        self._reload_committee()
        super().__init__(*args, **kwargs)
        logger.info("                                                      (😎 IT IS THE LAST ONE 😎)")
        logger.info(f'Initialized DogeSubscriber at {time.time()}')

    def _reload_committee(self):
        self.committee = DogeCommittee()

    def _check_committee_expired(self):
        return self.committee.expired(at_timestamp=self.timestamp)

    def handle(self, channel, data, *args, **kwargs):
        # check if we received data for a ticker we support
        if self.ticker not in SUPPORTED_DOGE_TICKERS:  # @tomcounsell please check if this is OK or I should register
                                                       # for tickers of interest in some other way
            logger.debug(f'Ticker {self.ticker} not in {SUPPORTED_DOGE_TICKERS}, skipping...')
            return

        # check if the committee has expired
        if self._check_committee_expired():
            logger.info('Doge committee expired, reloading...')
            self._reload_committee()

        logger.info(f'Doge subscriber invoked at {self.timestamp}, channel={str(channel)}, data={str(data)} '
                    f'(it is now {time.time()})')
        transaction_currency, counter_currency = self.ticker.split('_')

        new_doge_storage = CommitteeVoteStorage(ticker=self.ticker,
                                                exchange=self.exchange,
                                                timestamp=self.timestamp,
                                                periods=self.committee.periods)

        ticker_votes, weights = self.committee.vote(transaction_currency, counter_currency, self.timestamp)
        # weighted_vote = sum([ticker_votes[i] * weights[i] for i in range(len(ticker_votes))]) / sum(weights)

        new_doge_storage.value = (sum(ticker_votes) * 100 / len(ticker_votes))  # normalize to +-100 scale
        new_doge_storage.save(publish=True)
        logger.info('Doge vote saved')


    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)


class DummyDogeSubscriber(DogeSubscriber):

    def __init__(self, committee):
        self.committee = committee
        # no super constructor calls, bypass the Redis and pubsub stuff

    def _check_committee_expired(self):
        return False



class DogeHistorySimulator:

    def __init__(self, start_time, end_time, ticker, exchange, horizon,
                 training_period_length=DOGE_TRAINING_PERIOD_DURATION_SECONDS,
                 time_to_retrain_seconds=DOGE_RETRAINING_PERIOD_SECONDS):
        self._start_time = db_interface.get_nearest_db_timestamp(start_time, ticker, exchange)
        self._end_time = db_interface.get_nearest_db_timestamp(end_time, ticker, exchange)
        self._training_period_length = training_period_length
        self._time_to_retrain_seconds = time_to_retrain_seconds
        self._ticker = ticker
        self._exchange = exchange
        self._transaction_currency, self._counter_currency = ticker.split('_')
        self._horizon = horizon


    def fill_history(self):
        karen = DogeTrainer(database=db_interface)  # see Karen Pryor

        for training_end_time in range(self._end_time,
                                       self._start_time + self._training_period_length,
                                       -self._time_to_retrain_seconds):
            # training_end_time = 1547132400  # debug stuff
            training_start_time = training_end_time - self._training_period_length

            logging.info(f'Processing data for committee trained on data '
                         f'from {datetime_from_timestamp(training_start_time)} '
                         f'to {datetime_from_timestamp(training_end_time)}')

            # check if a committee record already exists
            try:
                committee = DogeCommittee(committee_timestamp=training_end_time)
                logging.info(f'Committee successfully loaded at {training_end_time}')
            except:
                # no committee, we need to rerun training
                logging.info(f'No committee found, running training for timestamp {training_end_time}...')
                karen.retrain_doges(start_timestamp=training_start_time, end_timestamp=training_end_time)

                # now that Karen did her job we should totally have a working committee
                committee = DogeCommittee(committee_timestamp=training_end_time)

            # we need to simulate incoming price data
            self.feed_price_to_doge(committee=committee,
                                    committee_valid_from=training_end_time,
                                    committee_valid_to=training_end_time + self._time_to_retrain_seconds)

    def feed_price_to_doge(self, committee, committee_valid_from, committee_valid_to):
        prices_df = db_interface.get_resampled_prices_in_range(
            start_time=committee_valid_from,
            end_time=committee_valid_to,
            transaction_currency=self._transaction_currency,
            counter_currency=self._counter_currency,
            horizon=self._horizon
        )

        # rudely hijack the DogeSubscriber class
        subscriber = DummyDogeSubscriber(committee)

        for row in prices_df.itertuples():
            logging.info(f'Feeding {str(row)} to doge (using committee '
                         f'valid from {datetime_from_timestamp(committee_valid_from)} '
                         f'to {datetime_from_timestamp(committee_valid_to)}...')
            data_event = {
                'type': 'message',
                'pattern': None,
                'channel': b'PriceStorage',
                'data': f'''{{
                                "key": "{self._ticker}:{self._exchange}:PriceStorage:1",
                                "name": "{row.close_price}:{row.score}",
                                "score": "{row.score}"
                            }}'''.encode(encoding='UTF8')
            }
            # call the doge with this event
            subscriber(data_event=data_event)


