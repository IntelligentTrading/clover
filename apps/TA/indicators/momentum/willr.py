from settings import LOAD_TALIB
if LOAD_TALIB:
    import math, talib

from apps.TA import HORIZONS
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class WillrStorage(IndicatorStorage):

    class_periods_list = [14,]
    requisite_pv_indexes = ["high_price", "low_price", "close_price"]


    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """

        :param requisite_pv_index_arrays:
        :param periods:
        :return:
        """
        periods = periods or self.periods

        willr_value = talib.WILLR(
            requisite_pv_index_arrays["high_price"],
            requisite_pv_index_arrays["low_price"],
            requisite_pv_index_arrays["close_price"],
            timeperiod=periods
        )[-1]

        logger.debug(f"Willr computed: {willr_value}")

        if math.isnan(willr_value):
            return ""

        return  str(willr_value)


class WillrSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = WillrStorage
