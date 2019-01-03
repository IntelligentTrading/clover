from django.core.management.base import BaseCommand
from apps.doge.models.doge_train_test import DogeTradingManager
from apps.portfolio.services.doge_votes import get_allocations_from_doge


class Command(BaseCommand):

    def handle(self, *args, **options):
        get_allocations_from_doge()