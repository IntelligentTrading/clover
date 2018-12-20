import json
import logging
from datetime import datetime, timedelta


from django.core.management.base import BaseCommand

from apps.TA.management.commands.TA_worker import get_subscriber_classes
from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from settings.redis_db import database

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run Redis Indicator wipe and restore'

    def add_arguments(self, parser):
        parser.add_argument('arg', nargs='?', default='some_arg', type=str)

    def handle(self, *args, **options):
        logger.info("Starting wipe and restore...")

        arg = options['arg']

        wipe_all_indicators()

        start_score = TimeseriesStorage.score_from_timestamp(datetime.now().timestamp())
        end_score = TimeseriesStorage.score_from_timestamp((datetime.now()-timedelta(days=2)).timestamp())
        restore_indicators(start_score, end_score)



def wipe_all_indicators():
    for key in database.keys("*Storage*"):
        storage_class = key.decode("utf-8").split(":")[2]

        if "Price" in storage_class or "Volume" in storage_class:
            continue
        else:
            logger.debug("deleting key: " + key.decode("utf-8"))
            # database.delete(key)

def restore_indicators(start_score, end_score):

    subscribers = []
    for subsrciber_class in get_subscriber_classes():
        subscribers.append(subsrciber_class())

    for key in database.keys("*PriceStorage*close_price*"):
        [ticker, exchange, storage_class, index] = key.decode("utf-8").split(":")

        for score in range(int(start_score), int(end_score)+1):
            for subscriber in subscribers:
                subscriber(data_event=forge_data_event(
                    ticker, exchange, storage_class, index, "123ABC", score
                ))


def forge_data_event(ticker, exchange, storage_class, index, value, score):

    data = {
        "key": f"{ticker}:{exchange}:{storage_class}:{index}",
        "name": f"{value}:{score}",
        "score": f"{score}"
    }

    return {
        'type': 'message',
        'pattern': None,
        'channel': bytes(str(storage_class), "utf-8"),
        'data': bytes(json.dumps(data), "utf-8")
    }
