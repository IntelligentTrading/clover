from django.core.management.base import BaseCommand
from apps.doge.doge_health_status import CloverHealth


class Command(BaseCommand):

    def handle(self, *args, **options):
        dogtor = CloverHealth()
        dogtor.allocation_votes(dogtor.latest_allocation.id)