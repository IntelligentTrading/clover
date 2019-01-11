from django.core.management.base import BaseCommand
from apps.doge.doge_historical_runs import DogeHistorySimulator
import time
from apps.TA import PERIODS_1HR


class Command(BaseCommand):

    def handle(self, *args, **options):
        end_time = time.time()
        start_time = end_time - 60*60*2
        simulator = DogeHistorySimulator(start_time=start_time, end_time=end_time, ticker='BTC_USDT',
                                         exchange='binance', horizon=PERIODS_1HR, training_period_length=60 * 60,
                                         time_to_retrain_seconds=60 * 30)
        simulator.fill_history()