from settings import LOAD_TALIB

if LOAD_TALIB:
    import math, talib

from apps.TA.storages.abstract.indicator import IndicatorStorage
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class AdxStorage(IndicatorStorage):

    class_periods_list = [14,]
    requisite_pv_indexes = ["high_price", "low_price", "close_price"]

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrrays: dict, periods: int = 0) -> str:
        """

        :param requisite_pv_index_arrrays:
        :param periods:
        :return:
        """
        periods = periods or self.periods

        adx_value = talib.ADX(
            requisite_pv_index_arrrays["high_price"],
            requisite_pv_index_arrrays["low_price"],
            requisite_pv_index_arrrays["close_price"],
            timeperiod=periods
        )[-1]

        logger.debug(f"ADX computed: {adx_value}")

        if math.isnan(adx_value):
            return ""

        return str(adx_value)


class AdxSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = AdxStorage
