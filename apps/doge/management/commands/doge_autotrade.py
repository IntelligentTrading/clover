from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeTrainer, DogeCommittee
from settings import SUPPORTED_DOGE_TICKERS, DOGE_TRAINING_PERIOD_DURATION_SECONDS, \
    DOGE_RETRAINING_PERIOD_SECONDS, DOGE_REBALANCING_PERIOD_SECONDS
from apps.portfolio.management.commands.rebalancer import balance_portfolios
from apps.doge.doge_train_test import NoGoodDogesException
import logging
import time
from apps.backtesting.utils import datetime_from_timestamp, parallel_run_thread_pool, time_performance
from functools import partial

class Command(BaseCommand):

    @staticmethod
    @time_performance
    def run_training_in_parallel(arguments, num_processes):
        parallel_run_thread_pool(DogeTrainer.run_training_zipped_args, arguments, pool_size=num_processes)


    def handle(self, *args, **options):

        while(True):
            # check if retrain is needed, run retraining
            try:
                arguments = []
                for ticker in SUPPORTED_DOGE_TICKERS:
                    latest_training_timestamp = DogeCommittee.latest_training_timestamp(ticker)

                    if latest_training_timestamp is None or True: # (time.time() - latest_training_timestamp > DOGE_RETRAINING_PERIOD_SECONDS):
                        end_timestamp = int(time.time())  # UTC timestamp
                        start_timestamp = end_timestamp - DOGE_TRAINING_PERIOD_DURATION_SECONDS

                        logging.info(f'AUTOTRADING: >>> Starting to retrain committee for {ticker} '
                                     f'at {datetime_from_timestamp(time.time())}...')
                        end_time_str = datetime_from_timestamp(latest_training_timestamp) if latest_training_timestamp else "N/A"
                        logging.info(f'AUTOTRADING: >>> (the latest committee was '
                                     f'trained with end time {end_time_str}')
                        arguments.append((start_timestamp, end_timestamp, ticker))

                # DogeTrainer.run_training(start_timestamp, end_timestamp, ticker)
                if len(arguments) > 0:
                    self.run_training_in_parallel(arguments, num_processes=2)   # should run for BTC_USDT, ETH_USDT, ETH_BTC
                    logging.info(
                        f'AUTOTRADING: >>> Retraining for {ticker} completed at {datetime_from_timestamp(time.time())}...')

            except Exception as e:
                logging.critical(f'Error during retraining committees: {str(e)}')


            # run rebalancer
            try:
                logging.info(f'AUTOTRADING: >>> Starting rebalancer at {datetime_from_timestamp(time.time())}...')
                balance_portfolios()
                logging.info(f'AUTOTRADING: >>> Rebalancing completed at {datetime_from_timestamp(time.time())}...')
            except Exception as e:
                logging.critical(f'Error during rebalancing: {str(e)}')
                logging.info(f'AUTOTRADING: >>> Rebalancing failed.')

            # sleep for rebalance duration
            time.sleep(DOGE_REBALANCING_PERIOD_SECONDS)

