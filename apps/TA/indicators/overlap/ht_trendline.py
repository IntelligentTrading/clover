from settings import LOAD_TALIB
if LOAD_TALIB:
    import math,talib

from apps.TA import HORIZONS, PERIODS_24HR
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class HtTrendlineStorage(IndicatorStorage):

    class_periods_list = [200,]
    requisite_pv_indexes = ["close_price"]

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """
        with cls.requisite_pv_indexes set

        :param requisite_pv_index_arrays: a dict with keys matching requisite+pv_indexes and values from self.get_denoted_price_array()
        :param periods: number of periods to compute value for
        :return:
        """
        periods = periods or self.periods

        ht_trendline_value = talib.HT_TRENDLINE(
            requisite_pv_index_arrays["close_price"]
        )[-1]

        logger.debug(f"HT Trendline computed: {ht_trendline_value}")

        if math.isnan(ht_trendline_value):
            return ""

        return str(ht_trendline_value)



class HtTrendlineSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = HtTrendlineStorage
