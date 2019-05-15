from django.core.management.base import BaseCommand
from apps.doge.doge_historical_runs import DogeHistorySimulator
import time
from apps.TA import PERIODS_1HR


class Command(BaseCommand):

    def handle(self, *args, **options):
        end_time = 1548250130 # time.time()
        start_time = end_time - 60*60*24*7
        tickers = ['BTC_USDT', 'ETH_USDT']

        for ticker in tickers:
            simulator = DogeHistorySimulator(start_time=start_time, end_time=end_time, ticker=ticker,
                                             exchange='binance', horizon=PERIODS_1HR, training_period_length=60*60*24,
                                             time_to_retrain_seconds=60*60*12)
            simulator.fill_history()