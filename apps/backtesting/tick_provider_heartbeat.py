import time
import pandas as pd
from apps.backtesting.data_sources import postgres_db
from apps.backtesting.tick_provider import TickProvider, TickerData


class TickProviderHeartbeat(TickProvider):

    def __init__(self, heartbeat_period_secs, transaction_currency, counter_currency, source=0, resample_period=60, database=postgres_db):

        super().__init__()
        self.heartbeat_period_secs = heartbeat_period_secs
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.source = source
        self.resample_period = resample_period
        self.database = database


    def run(self):
        while(True):
            # get the price for our transaction_currency-counter_currency pair
            # mock data for now
            ticker_data = TickerData(
                timestamp=1923891283,
                transaction_currency='BTC',
                counter_currency='USDT',
                source=0,
                resample_period=5,
                open_price=123124344, #row.open_price,
                high_price=213455555, #row.high_price,
                low_price=9999999, #row.low_price,
                close_price=213123233,
                close_volume=4,
                signals=[],
            )
            self.notify_listeners(ticker_data)
            time.sleep(self.heartbeat_period_secs)

        self.broadcast_ended()