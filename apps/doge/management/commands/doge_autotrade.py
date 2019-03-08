from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeTrainer, DogeCommittee
from settings import SUPPORTED_DOGE_TICKERS, DOGE_TRAINING_PERIOD_DURATION_SECONDS, \
    DOGE_RETRAINING_PERIOD_SECONDS, DOGE_REBALANCING_PERIOD_SECONDS
from apps.portfolio.management.commands.rebalancer import balance_portfolios
import logging
import time
from apps.backtesting.utils import datetime_from_timestamp


class Command(BaseCommand):

    def handle(self, *args, **options):

        while(True):
            # run rebalancer
            try:
                logging.info(f'AUTOTRADING: >>> Starting rebalancer at {datetime_from_timestamp(time.time())}...')
                balance_portfolios()
                logging.info(f'AUTOTRADING: >>> Rebalancing completed at {datetime_from_timestamp(time.time())}...')
            except Exception as e:
                logging.critical(f'Error during rebalancing: {str(e)}')

            # sleep for rebalance duration
            time.sleep(DOGE_REBALANCING_PERIOD_SECONDS)

            # check if retrain is needed, run retraining
            try:
                for ticker in SUPPORTED_DOGE_TICKERS:
                    if time.time() - DogeCommittee.latest_training_timestamp(ticker) > DOGE_RETRAINING_PERIOD_SECONDS:
                        end_timestamp = int(time.time())  # UTC timestamp
                        start_timestamp = end_timestamp - DOGE_TRAINING_PERIOD_DURATION_SECONDS

                        logging.info(f'AUTOTRADING: >>> Starting to retrain committee for {ticker} at {datetime_from_timestamp(time.time())}...')
                        DogeTrainer.run_training(start_timestamp, end_timestamp, ticker)
                        logging.info(
                            f'AUTOTRADING: >>> Retraining for {ticker} completed at {datetime_from_timestamp(time.time())}...')
            except Exception as e:
                logging.critical(f'Error during retraining committees: {str(e)}')