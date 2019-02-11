import numpy as np
import inspect
from collections import namedtuple
from abc import ABC, abstractmethod
from apps.TA import PERIODS_4HR, PERIODS_1HR

class FunctionProvider(ABC):

    def if_then_else(self, input, output1, output2):
        try:
            return output1 if input else output2
        except:
            return output1

    @classmethod
    def buy(self):
        pass

    @classmethod
    def sell(self):
        pass

    @classmethod
    def ignore(self):
        pass

    def identity(self, x):
        return x


DataKey = namedtuple('DataKey', 'source resample_period transaction_currency counter_currency start_time end_time')


class TAProvider(FunctionProvider):

    def __init__(self, db_interface, exchange='binance', resample_period=PERIODS_1HR):
        self._db_interface = db_interface
        self._exchange = exchange
        self._resample_period = resample_period  # TODO horizon, resample period and all that


    def get_indicator(self, indicator_name, input):
        timestamp = self._get_timestamp(input)
        transaction_currency, counter_currency = input[1:3]
        return self._db_interface.get_indicator(indicator_name, transaction_currency, counter_currency,
                      timestamp)


    def get_indicator_at_previous_timestamp(self, indicator_name, input):
        timestamp = self._get_timestamp(input)
        transaction_currency, counter_currency = input[1:3]
        return self._db_interface.get_indicator_at_previous_timestamp(indicator_name, transaction_currency,
                                                                      counter_currency,
                                                                      timestamp)

    def _get_timestamp(self, input):
        return input[0]

    def rsi(self, input):
        # query Redis for RSI value at _get_timestamp(input)
        return self.get_indicator('rsi', input)

    def rsi_lt_20(self, input):
        return self.rsi(input) < 20

    def rsi_lt_25(self, input):
        return self.rsi(input) < 25

    def rsi_lt_30(self, input):
        return self.rsi(input) < 30

    def rsi_gt_70(self, input):
        return self.rsi(input) > 70

    def rsi_gt_75(self, input):
        return self.rsi(input) > 75

    def rsi_gt_80(self, input):
        return self.rsi(input) > 80

    def macd_bullish(self, input):
        return self._crosses_from_below('macd', 'macd_signal', input)

    def macd_bearish(self, input):
        return self._crosses_from_above('macd', 'macd_signal', input)

    def ema_bullish_cross(self, input):
        return self._crosses_from_above('ema55', 'ema21', input)

    def ema_bearish_cross(self, input):
        return self._crosses_from_below('ema55', 'ema21', input)

    def bbands_bearish_cross(self, input):
        return self._crosses_from_below('close_price', 'bb_up', input)

    def bbands_bullish_cross(self, input):
        return self._crosses_from_above('close_price', 'bb_low', input)

    def bbands_squeeze_bullish(self, input):
        return self.get_indicator('bb_squeeze', input)\
               and self._crosses_from_below('close_price', 'bb_up', input)

    def bbands_squeeze_bearish(self, input):
        return self.get_indicator('bb_squeeze', input)\
               and self._crosses_from_above('close_price', 'bb_low', input)

    def bbands_price_gt_up(self, input):
        return self.get_indicator('close_price', input) > self.get_indicator('bb_up', input)

    def bbands_price_lt_low(self, input):
        return self.get_indicator('close_price', input) < self.get_indicator('bb_low', input)

    def slowd_gt_80(self, input):
        return self.get_indicator('slowd', input) > 80

    def slowd_lt_20(self, input):
        return self.get_indicator('slowd', input) < 20

    def candlestick_momentum_buy(self, input):
        return self.get_indicator('close_price', input) > \
               0.5 * self.get_indicator_at_previous_timestamp('close_price', input)

    def candlestick_momentum_sell(self, input):
        return self.get_indicator('close_price', input) <\
            0.5 * self.get_indicator_at_previous_timestamp('close_price', input)

    def _crosses_from_below(self, indicator, other, input):
        current_indicator = self.get_indicator(indicator, input)
        current_other = self.get_indicator(other, input)
        previous_indicator = self.get_indicator_at_previous_timestamp(indicator, input)
        previous_other = self.get_indicator_at_previous_timestamp(other, input)

        return current_indicator > current_other and previous_indicator <= previous_other

    def _crosses_from_above(self, indicator, other, input):
        current_indicator = self.get_indicator(indicator, input)
        current_other = self.get_indicator(other, input)
        previous_indicator = self.get_indicator_at_previous_timestamp(indicator, input)
        previous_other = self.get_indicator_at_previous_timestamp(other, input)

        return current_indicator < current_other and previous_indicator >= previous_other

    def adx(self, input):
        return self.get_indicator('adx', input)

    def sma20(self, input):
        return self.get_indicator('sma20', input)

    def ema20(self, input):
        return self.get_indicator('ema20', input)

    def sma50(self, input):
        return self.get_indicator('sma50', input)

    def ema50(self, input):
        return self.get_indicator('ema50', input)

    def sma200(self, input):
        return self.get_indicator('sma200', input)

    def ema200(self, input):
        return self.get_indicator('ema200', input)

    def price(self, input):
        return self.get_indicator('close_price', input)

    def volume(self, input):
        return self.get_indicator('volume', input)

    def sma50_volume(self, input):
        return self.get_indicator('sma50_volume', input)

    def volume_cross_up(self, input):
        return self._crosses_from_below('close_volume', 'sma50_volume', input)

    def volume_cross_down(self, input):
        return self._crosses_from_above('close_volume', 'sma50_volume', input)

    def macd_stoch_sell(self, input):
        return self.get_indicator('macd_hist', input) < 0 and \
               self.get_indicator('slowd', input) > 71

    def macd_stoch_buy(self, input):
        return self.get_indicator('macd_hist', input) > 0 and \
               self.get_indicator('slowd', input) < 29


class CachedDataTAProvider(TAProvider):

    def __init__(self, data):
        from apps.backtesting.data_sources import DB_INTERFACE
        super().__init__(DB_INTERFACE, 'binance', PERIODS_1HR)
        self.data = data

    def __str__(self):
        return("TAprovider")

    """
    def get_indicator(self, indicator_name, input):
        return self.data.__dict__[indicator_name][self._get_timestamp_index(input)]

    def get_indicator_at_previous_timestamp(self, indicator_name, input):
        index = self._get_timestamp_index(input)
        return self.data.__dict__[indicator_name][index-1]

    def _get_timestamp_index(self, input):
        assert self.data.price_data.index.get_loc(input[0]) == np.where(self.data.price_data.index == input[0])[0][0]
        return self.data.price_data.index.get_loc(input[0])
    """


    @staticmethod
    def build(start_time, end_time, ticker, horizon, exchange, db_interface):
        from apps.backtesting.data_sources import Data
        self._db_interface
        return CachedDataTAProvider(Data(start_time=start_time,
                                         end_time=end_time,
                                         ticker=ticker,
                                         horizon=horizon,
                                         start_cash=0,
                                         start_crypto=0,
                                         source=exchange, database=db_interface))


class RedisTAProvider(TAProvider):

    def __init__(self, db_interface):
        self.db_interface = db_interface


    def get_indicator(self, indicator_name, input, horizon=PERIODS_1HR): # TODO ensure the horizon can be changed
        timestamp = self._get_timestamp(input)
        transaction_currency, counter_currency = input[1:3]

        indicator_value = self.db_interface.get_indicator(
            timestamp=timestamp,
            indicator_name=indicator_name,  # TODO @tomcounsell ensure we have data for all indicator_names
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            horizon=horizon,
        )

        return indicator_value


    def get_indicator_at_previous_timestamp(self, indicator_name, input, horizon=PERIODS_1HR):
        timestamp = self._get_timestamp(input)
        transaction_currency, counter_currency = input[1:3]
        indicator_value = self.db_interface.get_indicator_at_previous_timestamp(
            timestamp=timestamp,
            indicator_name=indicator_name,  # TODO @tomcounsell ensure we have data for all indicator_names
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            horizon=horizon,
        )

        return indicator_value


class TAProviderCollection(FunctionProvider):

    def __init__(self, data_collection):
        self.providers = {DataKey(data.exchange, data.resample_period, data.transaction_currency, data.counter_currency,
                                  data.start_time, data.end_time): CachedDataTAProvider(data) for data in data_collection}
        # create methods
        members = inspect.getmembers(CachedDataTAProvider, predicate=inspect.isfunction)
        base_members = inspect.getmembers(FunctionProvider)
        for member in members:
            if member in base_members:
                continue

            function_name = member[0]
            if function_name.startswith('__'):
                continue

            setattr(TAProviderCollection, function_name, self._create_function(function_name))

    def _create_function(self, function_name):
        exec(f'''
def {function_name}(self, input):
    timestamp, transaction_currency, counter_currency = input
    provider = self.get_provider(timestamp, transaction_currency, counter_currency)
    return provider.{function_name}(input)
''')
        return locals()[f'{function_name}']

    def get_provider(self, timestamp, transaction_currency, counter_currency):
        for key in self.providers.keys():
            if key.transaction_currency == transaction_currency \
                    and key.counter_currency == counter_currency \
                    and key.end_time >= timestamp >= key.start_time:
                return self.providers[key]
        raise Exception(f'No data loaded for {transaction_currency} - {counter_currency} at {timestamp}!')

    def __str__(self):
        return ','.join([f'{key.transaction_currency}-{key.counter_currency}-{key.start_time}-{key.end_time}'
                         for key in self.providers.keys()])

