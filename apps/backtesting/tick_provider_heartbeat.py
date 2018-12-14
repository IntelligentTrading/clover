import time
import datetime
import pandas as pd
from apps.backtesting.data_sources import postgres_db, redis_db
from apps.backtesting.tick_provider import TickProvider, TickerData
import logging
from apps.backtesting.utils import datetime_from_timestamp


class TickProviderHeartbeat(TickProvider):

    def __init__(self, heartbeat_period_secs, ticker_list=['BTC_USDT', 'ETH_BTC'], database=redis_db):

        super().__init__()
        self.heartbeat_period_secs = heartbeat_period_secs
        self.ticker_list = ticker_list
        self.database = database


    def run(self):
        while(True):
            # get the price for our transaction_currency-counter_currency pair
            # mock data for now

            logging.info('♡♡♡♡                                                         ♡♡♡♡')
            logging.info(f'♡_.~"~._.~"~._(heart tick: {datetime_from_timestamp(datetime.datetime.now().timestamp())})_.~"~._.~"~._♡')
            logging.info('♡♡♡♡                                                         ♡♡♡♡')

            for ticker in self.ticker_list:
                transaction_currency, counter_currency = ticker.split('_')
                close_price, timestamp = self.database.get_latest_price(transaction_currency, counter_currency)
                ticker_data = TickerData(
                    timestamp=timestamp,
                    transaction_currency=transaction_currency,
                    counter_currency=counter_currency,
                    source=0,
                    resample_period=5,
                    open_price=close_price, #row.open_price,
                    high_price=close_price, #row.high_price,
                    low_price=close_price, #row.low_price,
                    close_price=close_price,
                    close_volume=0,
                    signals=[],
                )
                self.notify_listeners(ticker_data)
            time.sleep(self.heartbeat_period_secs)

        self.broadcast_ended()