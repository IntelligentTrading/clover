from django.core.management.base import BaseCommand
from apps.doge.doge_historical_runs import DogeHistorySimulator
from settings.doge import DOGE_RETRAINING_PERIOD_SECONDS, DOGE_TRAINING_PERIOD_DURATION_SECONDS, SUPPORTED_DOGE_TICKERS
from apps.TA import PERIODS_1HR
from apps.backtesting.utils import datetime_to_timestamp

class Command(BaseCommand):

    def handle(self, *args, **options):
        start_time = datetime_to_timestamp('2019/03/24 00:00:00')
        end_time = datetime_to_timestamp('2019/04/07 00:00:00')
        tickers = SUPPORTED_DOGE_TICKERS

        for ticker in tickers:
            simulator = DogeHistorySimulator(start_time=start_time, end_time=end_time, ticker=ticker,
                                             exchange='binance', horizon=PERIODS_1HR, training_period_length=DOGE_TRAINING_PERIOD_DURATION_SECONDS,
                                             time_to_retrain_seconds=DOGE_RETRAINING_PERIOD_SECONDS)
            simulator.fill_history()