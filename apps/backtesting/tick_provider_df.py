from apps.backtesting.tick_provider import TickProvider, TickerData


class TickProviderDataframe(TickProvider):

    def __init__(self, transaction_currency, counter_currency, source, dataframe, close_price_column_name):
        super(TickProviderDataframe, self).__init__()
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._source = source
        self._dataframe = dataframe
        self._close_price_column_name = close_price_column_name


    def run(self):
        for timestamp, row in self._dataframe.iterrows():
            close_price = row[self._close_price_column_name]
            ticker_data = TickerData(
                timestamp=timestamp,
                transaction_currency=self._transaction_currency,
                counter_currency=self._counter_currency,
                source=self._source,
                resample_period=None,
                open_price=close_price,  # row.open_price,
                high_price=close_price,  # row.high_price,
                low_price=close_price,  # row.low_price,
                close_price=close_price,
                close_volume=0,
                signals=[],
            )
            self.notify_listeners(ticker_data)
        self.broadcast_ended()