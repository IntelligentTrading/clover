from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeHistorySimulator
import time
from apps.TA import PERIODS_1HR


class Command(BaseCommand):

    def handle(self, *args, **options):
        end_time = time.time()
        start_time = end_time - 60*60*2
        simulator = DogeHistorySimulator(end_time=end_time,
                                         start_time=start_time,
                                         training_period_length=60*60,
                                         time_to_retrain_seconds=60*30,
                                         ticker='BTC_USDT',
                                         exchange='binance',
                                         horizon=PERIODS_1HR
                                         )
        simulator.fill_history()