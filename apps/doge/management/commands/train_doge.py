from django.core.management.base import BaseCommand
from apps.doge.doge_train_test import DogeTrainer
import time

ONE_WEEK = 60*60*24*7
ONE_DAY = 60*60*24
ONE_HOUR = 60*60

class Command(BaseCommand):

    def handle(self, *args, **options):
        end_timestamp = int(time.time())  # UTC timestamp
        start_timestamp = end_timestamp - ONE_HOUR
        ticker = 'ETH_USDT'
        DogeTrainer.run_training(start_timestamp, end_timestamp, ticker)