from django.core.management.base import BaseCommand
from apps.doge.models.doge import DogeTrainer
class Command(BaseCommand):

    def handle(self, *args, **options):
        DogeTrainer.run_training(start_timestamp=None, end_timestamp=None) # TODO: fill with actual times