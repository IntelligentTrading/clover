from django.core.management.base import BaseCommand
from apps.doge.models.doge_train_test import DogeTrader


class Command(BaseCommand):

    def handle(self, *args, **options):
        trader = DogeTrader()  # TODO replace with a static method