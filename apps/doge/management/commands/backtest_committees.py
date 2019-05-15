from django.core.management.base import BaseCommand
from apps.doge.doge_utils import committees_report


class Command(BaseCommand):

    def handle(self, *args, **options):
        import time
        end_timestamp = 1552611600
        start_timestamp = end_timestamp - 60*60*24*7
        committees_report(ticker='BTC_USDT', exchange='binance',
                          start_timestamp=start_timestamp, end_timestamp=end_timestamp)