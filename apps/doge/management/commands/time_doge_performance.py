from django.core.management.base import BaseCommand
from apps.doge.doge_utils import DogePerformanceTimer


class Command(BaseCommand):

    def handle(self, *args, **options):
        DogePerformanceTimer(run_variants_in_parallel=False)