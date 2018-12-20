from settings import LOAD_TALIB
if LOAD_TALIB:
    import math, talib

from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH, OTHER
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class StochStorage(IndicatorStorage):

    class_periods_list = [5,]
    requisite_pv_indexes = ["high_price", "low_price", "close_price"]

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """

        :param requisite_pv_index_arrays:
        :param periods:
        :return:
        """

        periods = periods or self.periods
        slowk, slowd = talib.STOCH(
            requisite_pv_index_arrays["high_price"],
            requisite_pv_index_arrays["low_price"],
            requisite_pv_index_arrays["close_price"],
            fastk_period=periods, slowk_period=periods*3/5,
            slowk_matype=0, slowd_period=periods*3/5, slowd_matype=0
        )[-1]

        stoch_values = f'{slowk}:{slowd}'

        logger.debug(f"Stoch computed: {stoch_values}")

        if math.isnan(slowk) or math.isnan(slowd):
            return ""

        return stoch_values


    def produce_signal(self):
        """
        defining the criteria for sending signals

        :return: None
        """
        if "this indicator" == "interesting":
            self.send_signal(trend=BULLISH)


class StochSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = StochStorage
