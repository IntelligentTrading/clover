import logging
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand

from apps.TA.storages.utils import missing_TA_data
from apps.common.utilities.multithreading import start_new_thread, run_all_multithreaded

from settings.redis_db import database

logger = logging.getLogger(__name__)


START_DATE = datetime(2019, 5, 1)
END_DATE = datetime.now()


class Command(BaseCommand):
    help = 'Run Redis Data gaps filler'

    def add_arguments(self, parser):
        parser.add_argument('arg', nargs='?', default='fill_gaps', type=str)

    def handle(self, *args, **options):
        logger.info("Starting data gaps restoration...")

        arg = options['arg']

        # See if the worker missed generating PV values
        refill_pv_storages()
        fill_data_gaps()


def refill_pv_storages():
    from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
    from apps.TA.storages.utils.pv_resampling import generate_pv_storages
    from apps.TA.storages.utils.memory_cleaner import clear_pv_history_values

    start_score = int(TimeseriesStorage.score_from_timestamp(START_DATE.timestamp()))
    end_score = int(TimeseriesStorage.score_from_timestamp(END_DATE.timestamp()))  # 206836 is Dec 20

    tei_processed = {}  # tei: ticker-exchange-index

    for key in database.keys("*PriceVolumeHistoryStorage*"):
        logger.info("running pv refill for " + str(key))
        [ticker, exchange, object_class, index] = key.decode("utf-8").split(":")
        tei_key = f"{ticker}_{exchange}"

        for score in range(start_score, end_score):
            generate_pv_storages(ticker, exchange, index, score)

            if not tei_key in tei_processed:
                tei_processed[tei_key] = []  # initialize with 0 indexes

            tei_processed[tei_key].append(index)  # add indexes
            if len(tei_processed[tei_key]) >= 5:  # vol + price hloc
                clear_pv_history_values(ticker, exchange, score)


def fill_data_gaps():
    method_params = []

    for ticker in ["BTC_USDT", ]:  # ["*_USDT", "*_BTC"]:
        for exchange in ["binance", ]:  # ["binance", "poloniex", "bittrex"]:
            for index in ['close_volume', 'open_price', 'high_price', 'low_price', 'close_price']:

                for key in database.keys(f"{ticker}*{exchange}*PriceStorage*{index}*"):
                    [ticker, exchange, storage_class, index] = key.decode("utf-8").split(":")

                    ugly_tuple = (ticker, exchange, index)
                    method_params.append(ugly_tuple)

    logger.info(f"{len(method_params)} tickers ready to fill gaps")

    results = run_all_multithreaded(condensed_fill_redis_gaps, method_params)
    missing_scores_count = sum([len(result) for result in results])

    logger.warning(f"{missing_scores_count} scores could not be recovered and are still missing.")


def condensed_fill_redis_gaps(ugly_tuple):
    (ticker, exchange, index) = ugly_tuple
    return missing_TA_data.fill_TA_storage_data_gaps(ticker, exchange, index)
