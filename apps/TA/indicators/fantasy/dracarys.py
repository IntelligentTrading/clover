import math
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class DracarysStorage(IndicatorStorage):

    class_periods_list = [1,]
    add_horizons = [2,]
    requisite_pv_indexes = ["close_price"]
    always_publish = True # do not change! It's the last one. All the doge is watching.


    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """

        :param requisite_pv_index_arrays:
        :param periods:
        :return:
        """
        periods = periods or self.periods

        if min([len(array) for array in requisite_pv_index_arrays] + [periods, ]) < periods:
            logger.debug("not enough data to compute")
            return ""

        if len(requisite_pv_index_arrays["close_price"]) < 2:
            return None

        dracarys_value = (int(
            (float(requisite_pv_index_arrays["close_price"][-1]) / requisite_pv_index_arrays["close_price"][-2])
            * 100
        ) - 100)

        logger.debug(f"Dracarys computed: {dracarys_value}")

        if math.isnan(dracarys_value):
            return None

        return str(dracarys_value)


class DracarysSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = DracarysStorage


def test_Dracarys():

    ticker="BTC_USDT"
    exchange="binance"
    class_name = "PriceStorage"
    score=251297

    from apps.TA.storages.data.price import PriceStorage as PS
    PS.query(ticker="BTC_USDT", exchange="binance", periods_range=2, score=251297)
