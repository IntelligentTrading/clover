import logging

from django.core.management.base import BaseCommand

from apps.common.utilities.multithreading import run_all_multithreaded

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run Redis Indicator fill gaps'

    def add_arguments(self, parser):
        parser.add_argument('arg', nargs='?', default='some_arg', type=str)

    def handle(self, *args, **options):
        logger.info("Starting fill gaps...")

        arg = options['arg']

        while True:
            fill_data_gaps()


def fill_data_gaps(force_fill=False):
    method_params = []
    from apps.TA.management.commands.TA_worker import get_subscriber_classes

    for ticker in ["BTC_USDT", ]:  # ["*_USDT", "*_BTC"]:
        for exchange in ["binance", ]:  # ["binance", "poloniex", "bittrex"]:
            for storage_class_name in [
                subscriber_class.storage_class.__name__ for subscriber_class in get_subscriber_classes()
            ]:
                ugly_tuple = (ticker, exchange, storage_class_name, bool(force_fill))
                method_params.append(ugly_tuple)

    logger.info(f"{len(method_params)} storages ready to fill gaps")

    results = run_all_multithreaded(condensed_fill_redis_TA_gaps, method_params)
    missing_scores_count = sum([len(result) for result in results])
    logger.warning(f"{missing_scores_count} scores could not be recovered and are still missing.")


def condensed_fill_redis_TA_gaps(ugly_tuple):
    (ticker, exchange, storage_class_name, force_fill) = ugly_tuple
    from apps.TA.storages.utils import missing_TA_data
    return missing_TA_data.find_TA_storage_data_gaps(
        ticker, exchange, storage_class_name, force_fill=force_fill
    )
