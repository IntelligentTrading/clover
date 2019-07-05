from apps.TA.storages.abstract.indicator import IndicatorException, IndicatorStorage
from apps.TA.storages.abstract.ticker_subscriber import TickerSubscriber, get_nearest_5min_timestamp
from settings import logger
from settings.doge import ENABLE_TA_FOR_SUPPORTED_DOGE_TICKERS_ONLY, SUPPORTED_DOGE_TICKERS, SUPPORTED_TICKERS

class IndicatorSubscriber(TickerSubscriber):
    class_describer = "indicator_subscriber"
    classes_subscribing_to = [
        # ...
    ]
    storage_class = IndicatorStorage  # override with applicable storage class

    def extract_params(self, channel, data, *args, **kwargs):

        # parse data like...
        # {
        #     "key": "VEN_USDT:binance:PriceStorage:close_price",
        #     "name": "176760000:1532373300",
        #     "score": "1532373300"
        # }

        self.data = data

        [self.ticker, self.exchange, object_class, self.key_suffix] = data["key"].split(":")

        if not object_class == channel and object_class in [
            sub_class.__name__ for sub_class in self.classes_subscribing_to
        ]:
            logger.warning(f'Unexpected that these are not the same:'
                           f'object_class: {object_class}, '
                           f'channel: {channel}, '
                           f'subscribing classes: {self.classes_subscribing_to}')

        [self.value, self.score] = data["name"].rsplit(":", 1)

        self.score = float(self.score)
        if not self.value or self.value == 'None':
            self.value = None


        if not self.score == int(float(data["score"])):
            logger.warning(f'Unexpected that score in name {self.score} '
                           f'is different than score {data["score"]}')

        if not self.score == int(self.score):
            raise IndicatorException("indicator timestamp should be 5min timestamp, score should be whole number (int)")

        self.timestamp = IndicatorStorage.timestamp_from_score(self.score)

        # import time
        # from apps.backtesting.utils import datetime_from_timestamp
        # if time.time() - self.timestamp > 60*20:
        #     logger.debug(f'Timestamp is {datetime_from_timestamp(self.timestamp)} which is more than 20 minutes old. Data: {data}')

        return


    def pre_handle(self, channel, data, *args, **kwargs):
        self.extract_params(channel, data, *args, **kwargs)
        super().pre_handle(channel, data, *args, **kwargs)


    def handle(self, channel, data, *args, **kwargs):
        """
        A new instance was saved belonging to one of the classes subscribed to.
        For standard indicators, self.ticker, self.exchange, and self.timestamp are available.
        These values were collected by self.pre_handle() and self.extract_params()

        :param channel:
        :param data:
        :param args:
        :param kwargs:
        :return:
        """

        # only run TA on the tickers that our doge support.
        should_compute = self.ticker in SUPPORTED_DOGE_TICKERS or not \
            ENABLE_TA_FOR_SUPPORTED_DOGE_TICKERS_ONLY and self.ticker in SUPPORTED_TICKERS
        if not should_compute:
            return  # stop handler

        if len(self.storage_class.requisite_pv_indexes):  # check that we have all the requisits needed for handling
            if self.key_suffix not in self.storage_class.requisite_pv_indexes:
                logger.debug(f'index {self.key_suffix} is not in {self.storage_class.requisite_pv_indexes} ...ignoring...')
                return  # stop handler
            else:
                # logger.debug(f'using indexes {self.storage_class.requisite_pv_indexes} for {self.storage_class.__name__} indicator')
                pass  # all good, carry on...

        import time
        from apps.backtesting.utils import datetime_from_timestamp

        self.storage_class.compute_and_save_all_values_for_timestamp(self.ticker, self.exchange, self.timestamp)
        # logger.info(f'Computed {self.storage_class.__name__} '
        #              f'for {self.ticker} with message timestamp {datetime_from_timestamp(self.timestamp)} '
        #              f'(it is now {datetime_from_timestamp(time.time())})')
        # logger.debug(f'Ignoring {self.storage_class.__name__} for {self.ticker}')
