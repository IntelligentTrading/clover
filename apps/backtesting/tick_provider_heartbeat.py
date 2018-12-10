import time
import pandas as pd
from apps.backtesting.data_sources import postgres_db
from apps.backtesting.tick_provider import TickProvider


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
            price_data = pd.DataFrame(
                {'close_price': 6956, 'high_price': 6973, 'low_price': 6892, 'close_volume': None},
                index=[1522540800]).iloc[0]  # TODO: fill this with real data

            signals_now = []
            for row in price_data.iteritems():
                self.notify_listeners(row, signals_now)
            time.sleep(self.heartbeat_period_secs)

        self.broadcast_ended()