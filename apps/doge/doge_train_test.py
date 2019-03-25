import logging

from apps.backtesting.data_sources import DB_INTERFACE
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.tick_provider_heartbeat import TickProviderHeartbeat
from apps.backtesting.tick_provider import TickerData
from apps.doge.doge_TA_actors import DogeStorage, DogePerformance, CommitteeStorage, SignalSubscriber, \
    CommitteeVoteStorage, BenchmarkPerformance
from apps.genetic_algorithms.genetic_program import GeneticTickerStrategy
from apps.genetic_algorithms.gp_artemis import ExperimentManager
from apps.genetic_algorithms.leaf_functions import RedisTAProvider
from apps.TA import PERIODS_1HR
from settings import DOGE_RETRAINING_PERIOD_SECONDS, logger, SUPPORTED_DOGE_TICKERS, DOGE_LOAD_ROCKSTARS
import time
from apps.genetic_algorithms.chart_plotter import save_dot_graph, get_dot_graph
from apps.backtesting.utils import datetime_from_timestamp, time_performance

METRIC_IDS = {
    'mean_profit': 0,
    'fitness': 1
}
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))

GP_TRAINING_CONFIG = os.path.join(BASE, 'doge_config.json')


class DogeRecord:

    def __init__(self, train_end_timestamp, doge_str, metric_id, metric_value, rank, fitness_function, fitness_value):
        self.train_end_timestamp = train_end_timestamp
        self.doge_str = doge_str
        self.metric_id = metric_id
        self.metric_value = metric_value
        self.rank = rank  # TODO remove rank information if not needed
        self.fitness_function = fitness_function
        self.fitness_value = fitness_value
        self.value = f'{doge_str}:{fitness_function}'

    @property
    def hash(self):
        return DogeStorage.hash(self.value)

    def save_to_storage(self):
        # save the doge itself
        new_doge_storage = DogeStorage(value=self.value, key_suffix=str(self.hash))
        new_doge_storage.save()

        if self.metric_value is None:
            logger.warning('Empty value encountered!')

        # save performance info
        new_doge_performance_storage = DogePerformance(key_suffix=f'{str(self.hash)}:{self.metric_id}',
                                                       ticker='BTC_USDT',
                                                       exchange='binance',
                                                       timestamp=self.train_end_timestamp,
                                                       value=f'{self.metric_value}:{self.fitness_value}:{self.rank}')
        new_doge_performance_storage.save()


class DogeTrainer:
    """
    A class that encapsulates GP training.
    """

    def __init__(self, database, gp_training_config_json=None):
        """

        :param database: database from data_sources, either Redis or Postgres
        :param gp_training_config_json: string JSON representing the training config, if None loaded from GP_TRAINING_CONFIG
        """
        if gp_training_config_json is None:
            with open(GP_TRAINING_CONFIG, 'r') as f:
                self.gp_training_config_json = f.read()
        else:
            self.gp_training_config_json = gp_training_config_json

        self.database = database
        # self.function_provider = function_provider or RedisTAProvider(db_interface=database)

    def retrain_doges(self, start_timestamp, end_timestamp, max_doges_to_save=10, training_ticker='BTC_USDT'):
        """
        Reruns doge training and saves results to the database.
        :param start_timestamp: starting time to use for training (tickers are specified in GP_TRAINING_CONFIG json)
        :param end_timestamp: end time to use for training (tickers are specified in GP_TRAINING_CONFIG json)
        :param max_doges_to_save: maximum number of GPs to save in the database
        :return:
        """
        config_json = self.fill_json_template(self.gp_training_config_json, start_timestamp, end_timestamp, training_ticker)

        logging.info('>>>>>>> Starting GP training... ')
        logging.info(f'    >>> start_time = {datetime_from_timestamp(start_timestamp)}')
        logging.info(f'    >>> end_time = {datetime_from_timestamp(end_timestamp)}')

        # DEBUG: loading rockstars
        rockstars = []
        if DOGE_LOAD_ROCKSTARS:
            rockstars = CommitteeStorage.load_rockstars(num_previous_committees_to_search=2, max_num_rockstars=5,
                                                        ticker=training_ticker, exchange='binance', timestamp=end_timestamp)
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
                                                                     sort_by=['mean_profit'], min_fitness=0)
        logging.info('>>>>>>> Ranking completed.')

        # write these doges to database
        logging.info('>>>>>>> Saving GPs to database...')

        redis_entries = []
        for i, row in enumerate(doge_df.itertuples()):
            if i > max_doges_to_save:
                break
            redis_entries.append(
                DogeRecord(train_end_timestamp=end_timestamp, doge_str=str(row.doge),
                           metric_id=METRIC_IDS['mean_profit'], metric_value=row.mean_profit, rank=i,
                           fitness_function=row.fitness_function, fitness_value=row.fitness_value))


        # save individual doges
        self._save_doges(redis_entries)

        # save benchmark performance
        new_benchmark_storage = BenchmarkPerformance(timestamp=end_timestamp, ticker=training_ticker, exchange='binance')
        new_benchmark_storage.value = f'{start_timestamp}:{doge_df.iloc[0].benchmark_profits}'
        new_benchmark_storage.save()

        # save current committee
        committee_str = ":".join(map(str, [redis_entry.hash for redis_entry in redis_entries]))
        new_committee_storage = CommitteeStorage(timestamp=end_timestamp, ticker=training_ticker, exchange='binance')
        new_committee_storage.value = committee_str
        new_committee_storage.save(publish=True)
        logging.info('>>>>>>> GPs saved to database.')
        return e

    def _save_doges(self, redis_entries):
        for i, redis_entry in enumerate(redis_entries):
            redis_entry.save_to_storage()

    @staticmethod
    def fill_json_template(gp_training_config_json, start_timestamp, end_timestamp, ticker):
        """
        Fills the json training template with starting and ending times.
        :param gp_training_config_json: string json describing the training process (loaded from GP_TRAINING_CONFIG file)
        :param start_timestamp: training start time
        :param end_timestamp: training end time
        :param ticker in the form of TRANSACTION_COUNTER
        :return:
        """
        return gp_training_config_json.format(
            start_time=datetime_from_timestamp(start_timestamp),
            end_time=datetime_from_timestamp(end_timestamp),
            ticker=ticker
        )

    @staticmethod
    def run_training(start_timestamp, end_timestamp, ticker, exchange="binance", horizon=12):
        """
        Instantiates a DogeTrainer and reruns training.
        :param start_timestamp:
        :param end_timestamp:
        :return:
        """
        start_time = DB_INTERFACE.get_nearest_db_timestamp(start_timestamp, ticker)
        end_time = DB_INTERFACE.get_nearest_db_timestamp(end_timestamp, ticker)

        # trainer = DogeTrainer.build_cached_redis_trainer(start_time, end_time, ticker, exchange, horizon)
        trainer = DogeTrainer(database=DB_INTERFACE)

        if start_time is None or end_time is None:
            logging.error(f'Unable to find close enough timestamp for {ticker},'
                         f'start time = {datetime_from_timestamp(start_timestamp)}, '
                         f'end time = {datetime_from_timestamp(end_timestamp)} ')
            logging.error('Training cannot continue.')
            from apps.tests.manual_scripts import show_indicator_status
            show_indicator_status(indicator_key='PriceStorage', ticker=ticker)
            return

        return trainer.retrain_doges(start_timestamp=start_time, end_timestamp=end_time, max_doges_to_save=10,
                              training_ticker=ticker)


class DogeTrader:
    """
    A class that encapsulates a single Doge that trades.
    NOTE: instantiated in DogeCommittee, no need to manually instantiate
    """

    def __init__(self, doge_str, doge_id, function_provider, gp_training_config_json, fitness_function):
        """
        Instantiates a doge trader.
        :param doge_str: the decision tree in string format
        :param function_provider: an instance of the TAProvider class that provides TA values
        :param gp_training_config_json: a string representation of the training config json (loaded from GP_TRAINING_CONFIG file)
        """

        self.doge_str = doge_str
        self.hash = doge_id
        self.gp_training_config_json = gp_training_config_json
        experiment_json = DogeTrainer.fill_json_template(self.gp_training_config_json, 0, 0, '_')
                # filling the template with empty values, because training data info isn't used when resurrecting
        self.doge, self.gp = ExperimentManager.resurrect_doge(experiment_json, self.doge_str, function_provider, fitness_function)
        self.strategy = GeneticTickerStrategy(tree=self.doge, gp_object=self.gp)

    def vote(self, ticker_data):
        """
        :param ticker_data: an instance of TickerData class, containing the ticker info and optionally OHLCV data and signals
        :return: an instance of StrategyDecision class
        """
        return self.strategy.process_ticker(ticker_data)

    def weight_at_timestamp(self, timestamp=None, metric_id=0, ticker='BTC_USDT'):
        result = self.performance_at_timestamp(timestamp, metric_id, ticker)
        if result is None:
            return None
        return result['mean_profit']

    def performance_at_timestamp(self, timestamp, metric_id=0, ticker='BTC_USDT'):
        return DogePerformance.performance_at_timestamp(doge_id=str(self.hash),
                                                        ticker=ticker,
                                                        exchange='binance',
                                                        timestamp=timestamp,
                                                        metric_id=metric_id)

    def save_doge_img(self, out_filename, format='svg'):
        return save_dot_graph(self.doge, out_filename, format)

    def show_doge_chart(self):
        chart = get_dot_graph(self.doge)
        from apps.backtesting.utils import in_notebook
        if in_notebook():
            from IPython.display import display
            display(chart)
        return chart

    @property
    def svg_source_chart(self):
        self.save_doge_img('tmp', format='svg')
        f = open('tmp.svg', 'r')
        loading = False
        lines = []
        for line in f:
            if not loading:
                if line.startswith('<svg'):  # rewind to svg tag
                    loading = True
                else:
                    continue
            lines.append(line)
        return '\n'.join(lines)



class DogeCommittee:
    """
    A class that encapsulates trading using a committee of GPs.
    The committee is built out of the latest GPs in the database.
    """

    def __init__(self, committee_timestamp=None, max_doges=100,
                 ttl=DOGE_RETRAINING_PERIOD_SECONDS, training_ticker='BTC_USDT',
                 db_interface=DB_INTERFACE, function_provider=None):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()

        self.max_doges = max_doges
        self._committee_timestamp = committee_timestamp  # gets filled with the last committee timestamp if set to None
        self.function_provider = function_provider or RedisTAProvider(db_interface=db_interface)
        self.periods = PERIODS_1HR  # TODO remove this hardcoding if we decide to use more horizons
        self._ttl = ttl
        self._training_ticker = training_ticker

        doge_traders = self._load_doge_traders()
        self.doge_traders = doge_traders if len(doge_traders) <= max_doges else doge_traders[:max_doges]
        self._benchmark_profit = self._get_benchmark_profit()


    def expired(self, at_timestamp):
        return at_timestamp - self._committee_timestamp > self._ttl

    def _load_doge_traders(self):
        """
        Loads doge traders that belong to this committee (based on timestamp, self._init_time).
        :return: a list of DogeTrader objects
        """
        doge_traders = []

        query_response = CommitteeStorage.query(ticker=self._training_ticker, exchange='binance', timestamp=self._committee_timestamp)
        loaded_timestamp = CommitteeStorage.timestamp_from_score(query_response['scores'][-1])
        if self._committee_timestamp is not None and self._committee_timestamp != loaded_timestamp:
            logger.critical('Loaded committee timestamp differs from the requested timestamp')
        elif self._committee_timestamp is None:
            self._committee_timestamp = loaded_timestamp

        if not query_response['values']:
            raise Exception(f'No committee members found for timestamp {self._committee_timetstamp}!')

        doge_committee_ids = query_response['values'][-1].split(':')
        for doge_id in doge_committee_ids:
            doge_storage = DogeStorage(key_suffix=doge_id)
            doge_str, fitness_function = doge_storage.get_value().decode('utf-8').split(':')

            doge = DogeTrader(doge_str=doge_str, doge_id=doge_id,
                              function_provider=self.function_provider,
                              gp_training_config_json=self.gp_training_config_json,
                              fitness_function=fitness_function)
            doge_traders.append(doge)

        logging.info(f'Loaded a committee of {len(doge_traders)} traders for {self._training_ticker} '
                     f'trained at {datetime_from_timestamp(self._committee_timestamp)}')
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

        if self._training_ticker != f'{transaction_currency}_{counter_currency}':
            logger.warning(f'Doge voting on ticker {transaction_currency}_{counter_currency}, '
                           f'and trained on {self._training_ticker}')

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
            weight = doge.weight_at_timestamp(timestamp=self._committee_timestamp)
            logger.debug(f'  Doge {i} says: {str(decision)} (its weight is {weight:.2f})')
            votes.append(decision.outcome)
            weights.append(weight)

        return votes, weights

    def generate_doge_images(self):
        for i, doge in enumerate(self.doge_traders):
            doge.save_doge_img(out_filename=f'apps/doge/static/{i}')

    def _get_benchmark_profit(self):
        try:
            result = BenchmarkPerformance.query(ticker=self._training_ticker, exchange='binance', timestamp=self._committee_timestamp)
            result = result.split(':')
            if self._committee_timestamp - int(result[0]) != DOGE_RETRAINING_PERIOD_SECONDS:
                return None
            return float(result[1])
        except:
            return None

    @staticmethod
    def latest_training_timestamp(ticker, exchange='binance'):
        try:
            query_response = CommitteeStorage.query(ticker=ticker, exchange=exchange, timestamp=None)
            loaded_timestamp = CommitteeStorage.timestamp_from_score(query_response['scores'][-1])
            return loaded_timestamp
        except:
            return None

    def show_all_traders(self):
        for doge in self.doge_traders:
            doge.show_doge_chart()

    @property
    def timestamp(self):
        return self._committee_timestamp

    @property
    def time_str(self):
        return datetime_from_timestamp(self.timestamp)

    @property
    def benchmark_profit(self):
        return self._benchmark_profit




class DogeTradingManager(TickListener):
    """
    A class that initializes a heartbeat tick provider that periodically polls Redis for ticker prices and
    trades on them using a DogeCommittee.
    """

    def __init__(self, database=DB_INTERFACE, heartbeat_period_secs=60):

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
        super().__init__(*args, **kwargs)
        logger.info("                                                      (😎 IT IS THE LAST ONE 😎)")
        self._rewrite_history = kwargs.get('rewrite_history', True)
        self._init_committees()


    def _init_committees(self):
        # load committees for all supported tickers
        self.committees = {}
        for ticker in SUPPORTED_DOGE_TICKERS:
            self._reload_committee(ticker)
        logger.info(f'Initialized DogeSubscriber at {time.time()}')

    def _reload_committee(self, ticker):
        logging.info(f'Reloading committee for {ticker}...')
        self.committees[ticker] = DogeCommittee(training_ticker=ticker)

    def _check_committee_expired(self, ticker):
        return self.committees[ticker].expired(at_timestamp=self.timestamp)

    def handle(self, channel, data, *args, **kwargs):
        # check if we received data for a ticker we support
        if self.ticker not in SUPPORTED_DOGE_TICKERS:  # @tomcounsell please check if this is OK or I should register
                                                       # for tickers of interest in some other way
            logger.debug(f'Ticker {self.ticker} not in {SUPPORTED_DOGE_TICKERS}, skipping...')
            return

        # check if the committee has expired
        if self._check_committee_expired(self.ticker):
            logger.info(f'Doge committee for ticker {self.ticker} expired, reloading...')
            self._reload_committee(ticker=self.ticker)

        logger.info(f'Doge subscriber invoked at {datetime_from_timestamp(self.timestamp)}, '
                    f'channel={str(channel)}, data={str(data)} '
                    f'(it is now {datetime_from_timestamp(time.time())})')
        transaction_currency, counter_currency = self.ticker.split('_')

        new_doge_storage = CommitteeVoteStorage(ticker=self.ticker,
                                                exchange=self.exchange,
                                                timestamp=self.timestamp,
                                                periods=self.committees[self.ticker].periods)

        if not self._rewrite_history:
            if new_doge_storage.has_saved_value():
                logging.warning(f'Found existing committee vote for {self.ticker} '
                                f'at {datetime_from_timestamp(self.timestamp)}, skipping computation.')
                logging.warning('To rewrite history, set rewrite_history=True when invoking DogeSubscriber.')
                return


        try:
            ticker_votes, weights = self.committees[self.ticker].vote(transaction_currency, counter_currency, self.timestamp)
            # weighted_vote = sum([ticker_votes[i] * weights[i] for i in range(len(ticker_votes))]) / sum(weights)

            new_doge_storage.value = (sum(ticker_votes) / len(ticker_votes))  # normalize to +-1 scale
            new_doge_storage.save(publish=True)
            logger.debug('Doge vote saved')
        except Exception as e:
            logging.info(f'Unable to vote for {self.ticker} '
                          f'at {datetime_from_timestamp(self.timestamp)}')


    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)


