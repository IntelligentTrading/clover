from settings import LOAD_TALIB
if LOAD_TALIB:
    import math, talib

from apps.TA.storages.abstract.indicator import IndicatorStorage
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger

EMA_LIST = [20, 21, 30, 50, 55, 200, ]


class EmaStorage(IndicatorStorage):
    # sorted_set_key = "BTC_USDT:poloniex:EmaStorage:30"

    class_periods_list = EMA_LIST
    requisite_pv_indexes = ["close_price"]

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """
        with cls.requisite_pv_indexes set

        :param index_value_arrays: a dict with keys matching requisite+pv_indexes and values from self.get_denoted_price_array()
        :param periods: number of periods to compute value for
        :return:
        """

        ema_value = talib.EMA(
            requisite_pv_index_arrays["close_price"],
            timeperiod=periods or self.periods
        )[-1]

        # logger.debug(f"EMA computed: {ema_value}")

        if math.isnan(ema_value): return ""

        return str(ema_value)

    def produce_signal(self):
        pass


class EmaSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = EmaStorage
