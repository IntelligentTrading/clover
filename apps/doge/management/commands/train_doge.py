from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeTrainer
from settings.doge import SUPPORTED_DOGE_TICKERS, DOGE_TRAINING_PERIOD_DURATION_SECONDS
import time


class Command(BaseCommand):

    def handle(self, *args, **options):
        end_timestamp = int(time.time())  # UTC timestamp
        start_timestamp = end_timestamp - DOGE_TRAINING_PERIOD_DURATION_SECONDS
        for ticker in SUPPORTED_DOGE_TICKERS:
            DogeTrainer.run_training(start_timestamp, end_timestamp, ticker)