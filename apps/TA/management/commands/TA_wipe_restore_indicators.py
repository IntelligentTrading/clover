import json
import logging
from datetime import datetime, timedelta


from django.core.management.base import BaseCommand

from apps.TA.management.commands.TA_worker import get_subscriber_classes
from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from apps.common.utilities.multithreading import run_all_multithreaded
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

def fill_data_gaps(force_fill=False):
    method_params = []
    from apps.TA.management.commands.TA_worker import get_subscriber_classes


    for ticker in ["BTC_USDT", ]:  # ["*_USDT", "*_BTC"]:
        for exchange in ["binance", ]:  # ["binance", "poloniex", "bittrex"]:
            for storage_class_name in [subscriber_class.storage_class.__name__
                                       for subscriber_class in get_subscriber_classes()]:
                for key in database.keys(f"{ticker}*{exchange}*{storage_class_name}*"):
                    [ticker, exchange, storage_class_name, key_suffixes] = key.decode("utf-8").split(":", 3)

                    ugly_tuple = (ticker, exchange, storage_class_name, key_suffixes, bool(force_fill))
                    method_params.append(ugly_tuple)

    logger.info(f"{len(method_params)} tickers ready to fill gaps")

    results = run_all_multithreaded(condensed_fill_redis_TA_gaps, method_params)
    missing_scores_count = sum([len(result) for result in results])
    logger.warning(f"{missing_scores_count} scores could not be recovered and are still missing.")


def condensed_fill_redis_TA_gaps(ugly_tuple):
    (ticker, exchange, storage_class_name, key_suffixes, force_fill) = ugly_tuple
    from apps.TA.storages.utils import missing_TA_data
    return missing_TA_data.find_TA_storage_data_gaps(
        ticker, exchange, storage_class_name, key_suffixes, force_fill=force_fill
    )
