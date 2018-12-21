from settings import LOAD_TALIB
if LOAD_TALIB:
    import math, talib

from apps.TA import HORIZONS
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class RsiStorage(IndicatorStorage):

    class_periods_list = [14,]
    requisite_pv_indexes = ["close_price"]

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """

        :param requisite_pv_index_arrays:
        :param periods:
        :return:
        """

        periods = periods or self.periods

        rsi_value = talib.RSI(
            requisite_pv_index_arrays["close_price"],
            timeperiod=periods+1
        )[-1]

        logger.debug(f"RSI computed: {rsi_value}")

        if math.isnan(rsi_value):
            return ""

        return str(rsi_value)


    def get_rsi_strength(self) -> int:
        rsi = int(self.value)
        if rsi is None or rsi <= 0.0 or rsi >= 100.0:
            return None

        assert (rsi>0.0) & (rsi<100.0), '>>> ERROR: RSI has extreme value of 0 or 100, highly unlikely'

        logger.debug(f"RSI={rsi}")

        rsi_strength = 0
        if rsi >= 80:
            rsi_strength = -3  # Extremely overbought
        elif rsi >= 75:
            rsi_strength = -2  # very overbought
        elif rsi >= 70:
            rsi_strength = -1  # overbought
        elif rsi <= 20:
            rsi_strength = 3  # Extremely oversold
        elif rsi <= 25:
            rsi_strength = 2   # very oversold
        elif rsi <= 30:
            rsi_strength = 1  # oversold
        return rsi_strength


    def produce_signal(self):
        import numpy as np

        rsi_strength = self.get_rsi_strength()
        if rsi_strength != 0:
            self.send_signal(
                trend=(BULLISH if rsi_strength > 0 else BEARISH),
                strength_value = int(np.abs(rsi_strength)), # should be 1,2,or3
                strength_max = int(3),
            )


class RsiSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [PriceStorage]
    storage_class = RsiStorage
