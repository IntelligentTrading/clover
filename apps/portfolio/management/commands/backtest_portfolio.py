from django.core.management.base import BaseCommand
from apps.backtesting.backtester_portfolio import DummyDataProvider


class Command(BaseCommand):

    def handle(self, *args, **options):
        provider = DummyDataProvider()
        provider.run()