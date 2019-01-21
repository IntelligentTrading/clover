from django.core.management.base import BaseCommand
from apps.tests.manual_scripts import RedisTests
import time


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--key_pattern', type=str)

    def handle(self, *args, **options):

        key_pattern = options['key_pattern'] or '*BTC_USDT*Willr*'
        RedisTests.find_gaps(key_pattern, time.time() - 60 * 60 * 24 * 30, time.time())
