import logging

from django.core.management.base import BaseCommand
import time


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--key_pattern', type=str)

    def handle(self, *args, **options):

        key_pattern = options['key_pattern'] or '*BTC_USDT*Dracarys*'
        find_gaps(key_pattern, time.time() - 60 * 60 * 24 * 30, time.time())

def find_gaps(key_pattern, start_timestamp, end_timestamp):
    from settings.redis_db import database
    from apps.TA.storages.data.price import PriceStorage
    from apps.backtesting.utils import datetime_from_timestamp

    start_score = PriceStorage.score_from_timestamp(start_timestamp)
    end_score = PriceStorage.score_from_timestamp(end_timestamp)

    keys = database.keys(key_pattern)
    for key in keys:
        logging.info(f'Processing data for {key}...')
        values = database.zrangebyscore(key, min=start_score, max=end_score)
        gaps = []
        for i, item in enumerate(values):
            if i == len(values) - 1:  # the last element
                break
            current_score = int(item.decode('UTF8').split(':')[-1])
            next_score = int(values[i + 1].decode('UTF8').split(':')[-1])
            if next_score == current_score:
                logging.warning(f'     Encountered duplicate scores: {item} and {values[i + 1]}')

                continue
            if next_score != current_score + 1:
                gaps.append((current_score, next_score))

        logging.info('Found gaps: ')

        for gap in gaps:

            start = datetime_from_timestamp(PriceStorage.timestamp_from_score(gap[0]))
            end = datetime_from_timestamp(PriceStorage.timestamp_from_score(gap[1]))
            logging.info(f'    start: {start}, end: {end}  (scores {gap[0]}-{gap[1]})')
