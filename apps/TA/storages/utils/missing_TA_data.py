from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from apps.TA.storages.utils.list_search import missing_elements
import logging
from datetime import datetime


logger = logging.getLogger(__name__)

def find_TA_storage_data_gaps(ticker, exchange, storage_class_name, force_fill=False):
    remaining_missing_scores = []

    storage_class = next(
        (
            storage_class for storage_class in supported_storage_classes
            if storage_class.__name__ == storage_class_name
        ),
        None
    )

    if not storage_class:
        return remaining_missing_scores

    for periods in storage_class.get_periods_list():

        now_score = int(TimeseriesStorage.score_from_timestamp(datetime.now().timestamp()))
        one_month = 1*30*24*12
        query_results = storage_class.query(ticker=ticker, exchange=exchange, periods_key=periods, periods_range=one_month)
        query_results['scores'].append(now_score)
        query_results['scores'].append(now_score-one_month)

        missing_scores = [int(float(score)) for score in missing_elements(query_results['scores'])]

        logger.info(f"{len(missing_scores)} total scores are missing. start filling...")

        for score in missing_scores:
            timestamp = TimeseriesStorage.timestamp_from_score(score)

            storage_object = storage_class(ticker=ticker, exchange=exchange, timestamp=timestamp, periods=periods)
            value = storage_object.get_value()
            if not value and force_fill:
                # do some force fill
                pass

            elif not value:
                remaining_missing_scores.append(score)
            else:
                logger.debug(f"filled one for {storage_class} at score {score}")

    return remaining_missing_scores


from apps.TA.indicators.overlap import sma, ema, wma, dema, tema, trima, bbands, ht_trendline, kama, midprice
from apps.TA.indicators.momentum import adx, adxr, apo, aroon, aroonosc, bop, cci, cmo, dx, macd, mom, ppo, \
    roc, rocr, rsi, stoch, stochf, stochrsi, trix, ultosc, willr
from apps.TA.indicators.events import bbands_squeeze_180min

supported_storage_classes = [
    sma.SmaStorage, ema.EmaStorage, wma.WmaStorage, bbands.BbandsStorage,

    adx.AdxStorage, stoch.StochStorage, macd.MacdStorage,

    bbands_squeeze_180min.BbandsSqueeze180MinStorage,

    willr.WillrStorage,  # still the last one
]

"""

from datetime import datetime
from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from apps.TA.indicators.overlap import sma
storage_class = sma.SmaStorage

ticker = "BTC_USDT"
exchange = "binance"
periods = 20*12

query_results = storage_class.query(ticker=ticker, exchange=exchange, periods_key=periods, periods_range=6*30*24*12)
len(query_results['scores'])

now_score = int(TimeseriesStorage.score_from_timestamp(datetime.now().timestamp()))

"""
