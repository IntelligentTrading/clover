from django.core.management.base import BaseCommand
from apps.doge.models.doge import DogeTrader
class Command(BaseCommand):

    def handle(self, *args, **options):
        trader = DogeTrader()  # TODO replace with a static method