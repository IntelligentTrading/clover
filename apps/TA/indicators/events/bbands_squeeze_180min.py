from apps.TA.indicators.overlap.bbands import BbandsStorage
from settings import LOAD_TALIB

if LOAD_TALIB:
    import math, talib

from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH, OTHER
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from settings import logger


class BbandsSqueeze180MinStorage(IndicatorStorage):
    # class_periods_list = [1, ] Not applicable, only one width and squeeze for each Bband
    requisite_pv_indexes = []
    always_publish = False

    def compute_value_with_requisite_indexes(self, requisite_pv_index_arrays: dict, periods: int = 0) -> str:
        """
        with cls.requisite_pv_indexes set

        :param index_value_arrays: a dict with keys matching requisite+pv_indexes and values from self.get_denoted_price_array()
        :param periods: number of periods to compute value for
        :return:
        """
        periods = periods or self.periods

        self.width = BbandsStorage(
            ticker=self.ticker, exchange=self.exchange, timestamp=self.unix_timestamp, periods_key=periods
        ).get_width()

        logger.debug(f"got width: {self.width}")

        if not self.width:
            return None

        query_result = BbandsStorage.query(
            ticker=self.ticker, exchange=self.exchange, timestamp=self.unix_timestamp, periods_key=periods,
            periods=periods*180
        )

        if query_result['values_count'] < 180:
            self.squeeze = False

        else:
            upper, middle, lower = 0, 1, 2  # based on order defined in BbandsStorage.compute_value...
            self.squeeze = any([
                True for v in query_result['values'] if bool(
                    (float(v.split(":")[upper]) - float(v.split(":")[lower])) / float(v.split(":")[middle])
                    <= self.width
                )
            ])

        self.value = f"{self.width}:{str(self.squeeze)}"  # eg. "0.04523353:True"
        logger.debug(f"BbandsSqueeze180MinStorage computed: {self.value}")

        return self.value


    def produce_signal(self):

        self.value = self.get_value()
        logger.debug("found TA value for BbandsSqueeze180MinStorage")
        logger.debug(self.value)

        self.width, self.squeeze = self.value.split(":")
        self.width, self.squeeze = float(self.width), bool(self.squeeze == 'True')


        if self.squeeze:

            self.bbands_value = BbandsStorage(
                ticker=self.ticker, exchange=self.exchange, timestamp=self.unix_timestamp
            ).get_value()
            [self.upperband_val, self.middleband_val, self.lowerband_val] = [float(val) for val in
                                                                             self.value.split(":")]
            self.price = float(PriceStorage.query(ticker=self.ticker, exchange=self.exchange, index="close_price",
                                                  timestamp=self.unix_timestamp)['values'][-1])

            if self.price > self.upperband_val:  # price breaks out above the band
                self.trend = BULLISH

            elif self.price < self.lowerband_val:  # price breaks out below the band
                self.trend = BEARISH

            else:  # price doesn't break out - but the squeeze thing is cool
                self.trend = OTHER

            self.send_signal(type="BbandsSqueeze180Min", trend=self.trend, width=self.width)
            logger.debug("new BbandsSqueeze180Min signal sent")


class BbandsSqueeze180MinSubscriber(IndicatorSubscriber):
    classes_subscribing_to = [BbandsStorage]
    storage_class = BbandsSqueeze180MinStorage
