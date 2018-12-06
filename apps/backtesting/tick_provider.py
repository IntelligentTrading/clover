from abc import ABC, abstractmethod


class TickProvider(ABC):

    def __init__(self):
        self._listeners = []

    def add_listener(self, listener):
        self._listeners.append(listener)

    def notify_listeners(self, price_data, signal_data):
        for listener in self._listeners:
            listener.process_event(price_data, signal_data)

    def broadcast_ended(self):
        for listener in self._listeners:
            listener.broadcast_ended()

    @abstractmethod
    def run(self):
        pass


class PriceDataframeTickProvider(TickProvider):

    def __init__(self, price_df):
        super(PriceDataframeTickProvider, self).__init__()
        self.price_df = price_df

    def run(self):
        for i, row in enumerate(self.price_df.itertuples()):
            # price = row.close_price
            # timestamp = row.Index

            self.notify_listeners(row, [])
        self.broadcast_ended()

