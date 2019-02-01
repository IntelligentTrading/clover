from abc import ABC, abstractmethod

from collections import namedtuple
TickerData = namedtuple('TickerData', 'timestamp transaction_currency counter_currency source resample_period '
                                      'open_price close_price high_price low_price close_volume signals')

class TickProvider(ABC):

    def __init__(self):
        self._listeners = []

    def add_listener(self, listener):
        self._listeners.append(listener)

    def notify_listeners(self, ticker_data):
        for listener in self._listeners:
            listener.process_event(ticker_data)

    def broadcast_ended(self):
        for listener in self._listeners:
            listener.broadcast_ended()

    @abstractmethod
    def run(self):
        pass


class PriceDataframeTickProvider(TickProvider):

    def __init__(self, price_df, transaction_currency, counter_currency, source, resample_period):
        super(PriceDataframeTickProvider, self).__init__()
        self.price_df = price_df
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.source = source
        self.resample_period = resample_period

    def run(self):
        for i, row in enumerate(self.price_df.itertuples()):
            ticker_data = TickerData(
                timestamp=row.Index,
                transaction_currency=self.transaction_currency,
                counter_currency=self.counter_currency,
                source=self.source,
                resample_period=self.resample_period,
                open_price=None, #row.open_price,
                high_price=None, #row.high_price,
                low_price=None, #row.low_price,
                close_price=row.close_price,
                close_volume=None,
                signals=[],
            )

            self.notify_listeners(ticker_data)
        self.broadcast_ended()

