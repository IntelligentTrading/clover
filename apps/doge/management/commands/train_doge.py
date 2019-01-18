from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeTrainer
from settings import SUPPORTED_DOGE_TICKERS
import time

ONE_WEEK = 60*60*24*7
ONE_DAY = 60*60*24
ONE_HOUR = 60*60


class Command(BaseCommand):

    def handle(self, *args, **options):
        end_timestamp = int(time.time())  # UTC timestamp
        start_timestamp = end_timestamp - ONE_HOUR
        for ticker in SUPPORTED_DOGE_TICKERS:
            DogeTrainer.run_training(start_timestamp, end_timestamp, ticker)