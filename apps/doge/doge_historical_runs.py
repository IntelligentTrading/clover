import logging

from apps.backtesting.data_sources import DB_INTERFACE
from apps.backtesting.utils import datetime_from_timestamp, parallel_run, time_performance
from apps.doge.doge_train_test import DogeTrainer, DogeCommittee, DogeSubscriber
from settings.doge import DOGE_RETRAINING_PERIOD_SECONDS, DOGE_TRAINING_PERIOD_DURATION_SECONDS
from functools import partial
from apps.genetic_algorithms.leaf_functions import TAProvider

class DummyDogeSubscriber(DogeSubscriber):

    def __init__(self, committees):
        self.committees = committees
        self._rewrite_history = False
        # no super constructor calls, bypass the Redis and pubsub stuff

    def _check_committee_expired(self, ticker):
        return False

    def _should_process_event(self, channel, data, *args, **kwargs):
        return True


class DogeHistorySimulator:

    def __init__(self, start_time, end_time, ticker, exchange, horizon,
                 training_period_length=DOGE_TRAINING_PERIOD_DURATION_SECONDS,
                 time_to_retrain_seconds=DOGE_RETRAINING_PERIOD_SECONDS,
                 parallel=True):
        self._start_time = DB_INTERFACE.get_nearest_db_timestamp(start_time, ticker, exchange)
        self._end_time = DB_INTERFACE.get_nearest_db_timestamp(end_time, ticker, exchange)
        self._training_period_length = training_period_length
        self._time_to_retrain_seconds = int(time_to_retrain_seconds)
        self._ticker = ticker
        self._exchange = exchange
        self._ticker = ticker
        self._horizon = horizon
        self._parallel = parallel

    @time_performance
    def fill_history(self, rewrite_history=False):

        training_intervals = []

        for training_end_time in range(self._end_time,
                                      self._start_time + self._training_period_length,
                                       -self._time_to_retrain_seconds):
            # training_end_time = 1547132400  # debug stuff
            training_start_time = training_end_time - self._training_period_length
            training_intervals.append((training_start_time, training_end_time))

        partial_func = partial(DogeHistorySimulator._single_period_run,
                               time_to_retrain_seconds=self._time_to_retrain_seconds,
                               ticker=self._ticker,
                               horizon=self._horizon,
                               exchange=self._exchange,
                               rewrite_history=rewrite_history)

        logging.info(f'Preparing to run history filling for {len(training_intervals)} periods...')

        if self._parallel:
            parallel_run(partial_func, param_list=training_intervals)
        else:
            for time_interval in training_intervals:
                partial_func(time_interval)

        logging.info('Filling historical data completed successfully.')


    @staticmethod
    @time_performance
    def _single_period_run(time_interval, time_to_retrain_seconds, ticker, horizon, exchange, rewrite_history=False):

        training_start_time, training_end_time = time_interval
        logging.info(f'Processing data for committee trained on data '
                     f'from {datetime_from_timestamp(training_start_time)} '
                     f'to {datetime_from_timestamp(training_end_time)}')

        # first check if all the votes are already there (performance optimization)
        from apps.tests.manual_scripts import RedisTests
        if not rewrite_history and \
                not any(RedisTests.find_gaps(
                    '{ticker}*{exchange}*CommitteeVoteStorage',
                    training_end_time,
                    training_end_time + time_to_retrain_seconds)
                                .values()):   # we have all the votes already
            logging.info(' ---> all the votes already exist, skipping...')
            return

        transaction_currency, counter_currency = ticker.split('_')
        ta_provider = TAProvider(db_interface=DB_INTERFACE)
        DB_INTERFACE.build_data_object(start_time=training_start_time,
                                       end_time=training_end_time,
                                       ticker=f'{transaction_currency}_{counter_currency}',
                                       horizon=horizon,
                                       exchange=exchange)
        committee = None

        if not rewrite_history:
            # check if a committee record already exists
            try:
                committee = DogeCommittee(committee_timestamp=training_end_time, db_interface=DB_INTERFACE,
                                          function_provider=ta_provider)
                logging.info(f'Committee successfully loaded at {training_end_time}')
            except:
                committee = None

        if committee is None:
            # no committee, we need to rerun training
            logging.info(f'No committee found, running training for timestamp {training_end_time}...')
            karen = DogeTrainer(database=DB_INTERFACE)  # see Karen Pryor; TODO: ensure cached TA values are used

            karen.retrain_doges(start_timestamp=training_start_time, end_timestamp=training_end_time,
                                training_ticker=ticker, parallelize=False)

            # now that Karen did her job we should totally have a working committee
            committee = DogeCommittee(committee_timestamp=training_end_time, db_interface=DB_INTERFACE,
                                      function_provider=ta_provider)
        # we need to simulate incoming price data
        DogeHistorySimulator.feed_price_to_doge(committee=committee,
                                                committee_valid_from=training_end_time,
                                                committee_valid_to=training_end_time + time_to_retrain_seconds,
                                                ticker=ticker,
                                                horizon=horizon,
                                                exchange=exchange)


    @staticmethod
    @time_performance
    def feed_price_to_doge(committee, committee_valid_from, committee_valid_to, ticker, horizon, exchange):
        logging.info('Feeding prices to doge...')
        transaction_currency, counter_currency = ticker.split('_')
        prices_df = DB_INTERFACE.get_resampled_prices_in_range(
            start_time=committee_valid_from,
            end_time=committee_valid_to,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            horizon=horizon
        )

        committees = {ticker: committee}

        # rudely hijack the DogeSubscriber class
        subscriber = DummyDogeSubscriber(committees)

        for i, row in enumerate(prices_df.itertuples()):
            if i % 100 == 0:
                logging.debug(f'Feeding {str(row)} ({i+1}/{len(prices_df)}) to doge (using committee '
                             f'valid from {datetime_from_timestamp(committee_valid_from)} '
                             f'to {datetime_from_timestamp(committee_valid_to)}...')


            if row.close_price is None:
                continue
            data_event = {
                'type': 'message',
                'pattern': None,
                'channel': b'PriceStorage',
                'data': f'''{{
                                "key": "{ticker}:{exchange}:PriceStorage:1",
                                "name": "{row.close_price}:{row.score}",
                                "score": "{row.score}"
                            }}'''.encode(encoding='UTF8')
            }
            # call the doge with this event
            try:
                subscriber(data_event=data_event)
            except Exception as e:
                logging.error(str(e))
        logging.info('Doges ingested all prices.')
