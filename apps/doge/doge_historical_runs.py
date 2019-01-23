import logging

from apps.backtesting.data_sources import db_interface
from apps.backtesting.utils import datetime_from_timestamp, parallel_run, time_performance
from apps.doge.doge_train_test import DogeTrainer, DogeCommittee, DogeSubscriber
from settings import DOGE_TRAINING_PERIOD_DURATION_SECONDS, DOGE_RETRAINING_PERIOD_SECONDS
from functools import partial

class DummyDogeSubscriber(DogeSubscriber):

    def __init__(self, committees):
        self.committees = committees
        # no super constructor calls, bypass the Redis and pubsub stuff

    def _check_committee_expired(self, ticker):
        return False


class DogeHistorySimulator:

    def __init__(self, start_time, end_time, ticker, exchange, horizon,
                 training_period_length=DOGE_TRAINING_PERIOD_DURATION_SECONDS,
                 time_to_retrain_seconds=DOGE_RETRAINING_PERIOD_SECONDS,
                 parallel=True):
        self._start_time = db_interface.get_nearest_db_timestamp(start_time, ticker, exchange)
        self._end_time = db_interface.get_nearest_db_timestamp(end_time, ticker, exchange)
        self._training_period_length = training_period_length
        self._time_to_retrain_seconds = time_to_retrain_seconds
        self._ticker = ticker
        self._exchange = exchange
        self._transaction_currency, self._counter_currency = ticker.split('_')
        self._horizon = horizon
        self._parallel = parallel

    @time_performance
    def fill_history(self):
        karen = DogeTrainer(database=db_interface)  # see Karen Pryor
        training_intervals = []

        for training_end_time in range(self._end_time,
                                       self._start_time + self._training_period_length,
                                       -self._time_to_retrain_seconds):
            # training_end_time = 1547132400  # debug stuff
            training_start_time = training_end_time - self._training_period_length
            training_intervals.append((training_start_time, training_end_time))

        partial_func = partial(DogeHistorySimulator._single_period_run,
                               doge_trainer=karen,
                               time_to_retrain_seconds=self._time_to_retrain_seconds,
                               transaction_currency=self._transaction_currency,
                               counter_currency=self._counter_currency,
                               horizon=self._horizon,
                               exchange=self._exchange)

        if self._parallel:
            parallel_run(partial_func, param_list=training_intervals)
        else:
            for time_interval in training_intervals:
                partial_func(time_interval)

        logging.info('Filling historical data completed successfully.')

    @staticmethod
    def _single_period_run(time_interval, doge_trainer, time_to_retrain_seconds, transaction_currency,
                           counter_currency, horizon, exchange):
        training_start_time, training_end_time = time_interval
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
            doge_trainer.retrain_doges(start_timestamp=training_start_time, end_timestamp=training_end_time,
                                       training_ticker=f'{transaction_currency}_{counter_currency}')

            # now that Karen did her job we should totally have a working committee
            committee = DogeCommittee(committee_timestamp=training_end_time)
        # we need to simulate incoming price data
        DogeHistorySimulator.feed_price_to_doge(committee=committee,
                                                committee_valid_from=training_end_time,
                                                committee_valid_to=training_end_time + time_to_retrain_seconds,
                                                transaction_currency=transaction_currency,
                                                counter_currency=counter_currency,
                                                horizon=horizon,
                                                exchange=exchange)

    @staticmethod
    def feed_price_to_doge(committee, committee_valid_from, committee_valid_to, transaction_currency,
                           counter_currency, horizon, exchange):
        prices_df = db_interface.get_resampled_prices_in_range(
            start_time=committee_valid_from,
            end_time=committee_valid_to,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            horizon=horizon
        )

        ticker = f'{transaction_currency}_{counter_currency}'
        committees = {ticker: committee}

        # rudely hijack the DogeSubscriber class
        subscriber = DummyDogeSubscriber(committees)

        for row in prices_df.itertuples():
            logging.info(f'Feeding {str(row)} to doge (using committee '
                         f'valid from {datetime_from_timestamp(committee_valid_from)} '
                         f'to {datetime_from_timestamp(committee_valid_to)}...')
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
            subscriber(data_event=data_event)

