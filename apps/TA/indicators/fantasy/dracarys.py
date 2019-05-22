import math
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class DracarysStorage(IndicatorStorage):

    class_periods_list = [2,]
    requisite_pv_indexes = ["close_price"]
    always_publish = True # do not change! it's the last one! everyone is watching


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

        dracarys_value = (int(
            (float(requisite_pv_index_arrays["close_price"][-1]) / requisite_pv_index_arrays["close_price"][-2])
            * 100
        ) - 100)

        logger.debug(f"Dracarys computed: {dracarys_value}")

        if math.isnan(dracarys_value):
            return ""

        return str(dracarys_value)


class DracarysSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = DracarysStorage
