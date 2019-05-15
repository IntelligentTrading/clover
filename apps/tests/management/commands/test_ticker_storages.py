from django.core.management.base import BaseCommand
from apps.tests.manual_scripts import RedisTests


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str)

    def handle(self, *args, **options):
        ticker = options['ticker'] or 'BTC_USDT'
        RedisTests.test_ticker_storages(ticker)