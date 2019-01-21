from django.core.management.base import BaseCommand
from apps.tests.manual_scripts import show_indicator_status

class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str)
        parser.add_argument('--indicator_key', type=str)

    def handle(self, *args, **options):
        ticker = options['ticker'] or 'BTC_USDT'
        storage = options['indicator_key'] or 'PriceStorage'

        show_indicator_status(storage, ticker)