from django.core.management.base import BaseCommand
from apps.doge.models.doge_train_test import DogeTradingManager


class Command(BaseCommand):

    def handle(self, *args, **options):
        trader = DogeTradingManager()  # TODO replace with a static method