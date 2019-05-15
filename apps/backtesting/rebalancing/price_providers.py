import logging
import pandas as pd
from apps.backtesting.data_sources import NoPriceDataException
from apps.backtesting.legacy_postgres import POSTGRES


class PriceProvider:

    def __init__(self):
        self.providers = [cls() for cls in PriceProvider.__subclasses__()]
        self.default_provider = CryptoPriceProvider()

    def get_price(self, asset, timestamp, **kwargs):
        for provider in self.providers:
            if provider.can_handle(asset):
                return provider.get_price(asset, timestamp, **kwargs)
        return self.default_provider.get_price(asset, timestamp, **kwargs)

    def can_handle(self, asset_name):
        pass


class CryptoPriceProvider(PriceProvider):

    def __init__(self):
        pass

    def get_price(self, coin, timestamp, db_interface, counter_currency='BTC'):

        if coin == counter_currency:
            return 1

        if db_interface == POSTGRES:
            kwargs = {'source': 2, 'normalize': True}
        else:
            kwargs = {'exchange': 'binance'}

        price = None
        if coin == counter_currency:
            return 1
        elif coin == 'USDT' and counter_currency == 'BTC':
            # get BTC_USDT price and invert
            btc_usdt_price, _ = db_interface.get_price_nearest_to_timestamp(transaction_currency='BTC',
                                                                            timestamp=timestamp,
                                                                            counter_currency='USDT',
                                                                            return_timestamp=True,
                                                                            **kwargs)
            return 1.0 / btc_usdt_price

        try:
            price, retrieved_timestamp = db_interface.get_price_nearest_to_timestamp(transaction_currency=coin,
                                                                                     timestamp=timestamp,
                                                                                     counter_currency=counter_currency,
                                                                                     return_timestamp=True,
                                                                                     **kwargs)

        except NoPriceDataException:
            if counter_currency == 'USDT':
                btc_price, retrieved_timestamp = db_interface.get_price_nearest_to_timestamp(transaction_currency=coin,
                                                                                             timestamp=timestamp,
                                                                                             counter_currency='BTC',
                                                                                             return_timestamp=True,
                                                                                             **kwargs)
                price = db_interface.convert_value_to_USDT(value=1, timestamp=retrieved_timestamp, transaction_currency=coin,
                                                      **kwargs)

        if price is None:
            raise NoPriceDataException(f'error retrieving price for {coin}_{counter_currency} at {timestamp}')
        else:
            return price

    def can_handle(self, asset_name):
        return False


class GoldPriceProvider(PriceProvider):

    def __init__(self, csv_relative_path='GLD.csv'):
        # load the CSV file data
        import os
        from settings import BASE_DIR
        csv_path = os.path.join(BASE_DIR, csv_relative_path)
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        self.price_df = df
        self.btc_price_provider = CryptoPriceProvider()

    def get_price(self, asset, timestamp, **kwargs):
        assert asset == 'GLD_ETF'
        dt = pd.to_datetime(timestamp, unit='s')
        result = self.price_df.iloc[self.price_df.index.get_loc(dt, method='nearest')]
        delta_time = abs(result.name.timestamp() - dt.timestamp())
        if delta_time > 60*60*24*3:
            logging.warning(f'Retrieved price of gold more than +-3 days from the requested timestamp! '
                            f'(requested: {dt}, retrieved: {result.name})')

        counter_currency = kwargs.get('counter_currency', 'BTC')
        if counter_currency == 'USDT':
            return result.Close
        else:
            # someone wants to know how much this costs in Bitcoin
            btc_price = self.btc_price_provider.get_price('BTC', timestamp, counter_currency='USDT')
            return result.Close / btc_price

    def can_handle(self, asset_name):
        return asset_name == 'GLD_ETF'