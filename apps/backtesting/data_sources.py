import numpy as np
import pandas as pd
import logging

from dateutil import parser
from apps.TA.storages.data.price import PriceStorage
from apps.TA.storages.data.pv_history import PriceVolumeHistoryStorage
from abc import ABC
from apps.TA import PERIODS_1HR
from apps.backtesting.utils import time_performance
from apps.TA.indicators.momentum import rsi, stochrsi, adx, macd, mom, stoch, willr
from apps.TA.indicators.overlap import sma, ema, wma, bbands, ht_trendline
from apps.TA.indicators.events import bbands_squeeze_180min
from collections import OrderedDict


STORAGE_CLASS = {
    'rsi': rsi.RsiStorage,
    'stoch_rsi': stochrsi.StochrsiStorage,
    'adx': adx.AdxStorage,
    'macd': macd.MacdStorage,
    'macd_value': macd.MacdStorage,
    'macd_signal': macd.MacdStorage,
    'macd_hist': macd.MacdStorage,
    'mom': mom.MomStorage,
    'sma': sma.SmaStorage,
    'ema': ema.EmaStorage,
    'wma': wma.WmaStorage,
    'bbands': bbands.BbandsStorage,
    'bb_up': bbands.BbandsStorage,
    'bb_mid': bbands.BbandsStorage,
    'bb_low': bbands.BbandsStorage,
    'bb_squeeze': bbands_squeeze_180min.BbandsSqueeze180MinStorage,
    'bb_width': bbands.BbandsStorage,
    'ht_trendline': ht_trendline.HtTrendlineStorage,
    'slowd': stoch.StochStorage,
    'willr': willr.WillrStorage,
    'close_price': PriceStorage,
}

MAX_CACHED_INDICATORS = 20
MAXED_CACHED_DATA_OBJECTS = 20
START_CASH = 1000
START_CRYPTO = 0


class IndicatorCache:
    """
    Caches up to MAX_CACHED_INDICATORS indicators.
    Once the maximum cache size is reached, the least recently used indicators are removed
    as new ones come in.
    """
    def __init__(self, max_size=MAX_CACHED_INDICATORS):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get_indicator(self, indicator_name, ticker, timestamp, exchange='binance', horizon=PERIODS_1HR):
        """
        Gets an indicator from the cache.
        :param indicator_name: indicator name (e.g. rsi, sma50, etc.)
        :param ticker: ticker in the form of TRANSACTION_COUNTER
        :param timestamp: timestamp
        :param exchange: exchange
        :param horizon: horizon
        :return: indicator value if cached, None if not
        """
        key = (indicator_name, ticker, timestamp, exchange, horizon)
        return self.cache.get(key, None)

    def save_indicator(self, indicator_name, ticker,
                       timestamp, value, exchange='binance', horizon=PERIODS_1HR):
        """
        Saves an indicator value to the cache.
        :param indicator_name: indicator name (e.g. rsi)
        :param ticker: ticker in the form of TRANSACTION_COUNTER
        :param timestamp: timestamp
        :param value: indicator value to save
        :param exchange: exchange
        :param horizon: horizon
        :return:
        """
        key = (indicator_name, ticker, timestamp, exchange, horizon)
        if len(self.cache.keys()) > self.max_size:
            del self.cache[self.cache.keys()[0]]
        self.cache[key] = value


class DataCache:
    """
    A cache for data objects.
    """

    def __init__(self, max_objects=MAXED_CACHED_DATA_OBJECTS):
        self.cached_data_objects = []
        self.index = 0
        self.max_objects = max_objects

    def add_data_object(self, data):
        if len(self.cached_data_objects) == self.max_objects:
            self.cached_data_objects[self.index] = data
            self.index = (self.index + 1) % self.max_objects
        else:
            self.cached_data_objects.append(data)
            self.index += 1

    @property
    def objects(self):
        return self.cached_data_objects


class Database(ABC):
    """
    Abstract class encapsulating basic database operations.
    """

    def convert_value_to_USDT(self, value, timestamp, transaction_currency, source):
        if value == 0:
            return 0
        if transaction_currency == "USDT":  # already in USDT
            return value
        try:
            value_USDT = value * self.get_price(transaction_currency, timestamp, source, "USDT")  # if trading against USDT
            return value_USDT
        except:
            # fall back to converting to BTC fitst
            value_BTC_in_USDT = self.get_price("BTC", timestamp, source, "USDT")
            if transaction_currency == "BTC":
                return value * value_BTC_in_USDT

            value_transaction_currency_in_BTC = self.get_price(transaction_currency, timestamp, source, "BTC")
            return value_BTC_in_USDT * value_transaction_currency_in_BTC * value

    def fetch_delayed_price(self, timestamp, transaction_currency, counter_currency, source, time_delay, original_price=None):
        if time_delay != 0 or original_price is None:
            return self.get_price(transaction_currency, timestamp + time_delay, source, counter_currency)
        else:
            return original_price


class RedisDB(Database):
    """
    Redis DB interface.
    """

    DEFAULT_INDICATOR_PERIODS = {
        'rsi': 14,
        'bb_up': 5,
        'bb_mid': 5,
        'bb_low': 5,
        'bb_squeeze': 1,
        'macd_value': 26,
        'macd_signal': 26,
        'macd_hist': 26,
        'adx': 1,
        'slowd': 5,
        'close_price': 1,
    }

    def __init__(self):
        self.data_cache = DataCache()
        self.indicator_cache = IndicatorCache()

    def _default_indicator_period(self, indicator_name):
        if indicator_name.startswith('sma') or indicator_name.startswith('ema'):
            return int(indicator_name[3:])
        else:
            return self.DEFAULT_INDICATOR_PERIODS.get(indicator_name, 1)

    def _remove_duplicate_indexes(self, df, data_description=''):
        df_len = len(df)
        cleaned = df[~df.index.duplicated(keep='first')]
        if (len(cleaned) != df_len):
            logging.critical(f'Encountered duplicate values in {data_description} data, {df_len} '
                             f'values loaded, {len(cleaned)} remain after removal.')

        return cleaned

    def get_resampled_prices_in_range(self, start_time, end_time, transaction_currency,
                                      counter_currency, horizon, exchange="binance"):
        """

        :param start_time: start of time range (timestamp in seconds)
        :param end_time: end of time range (timestamp in seconds)
        :param transaction_currency: transaction currency
        :param counter_currency: counter currency
        :param index:
        :param exchange: the used exchange
        :return: a dataframe with open, high, low and closing prices
        """

        # TODO add support for horizon

        if end_time is None or start_time is None:
            logging.error('Start or end time is set to None!')

        periods_range = (end_time - start_time) // (5*60)

        close_prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=exchange,
            index="close_price",
            timestamp=end_time,
            periods_range=periods_range,
            timestamp_tolerance=0
        )
        
        if close_prices['values_count'] == 0:
            logging.error('No close prices returned!')

        timestamps = [PriceStorage.timestamp_from_score(float(score)) for score in close_prices['scores']]
        close_price_values = list(map(float, close_prices['values']))
        scores = list(map(float, close_prices['scores']))

        data = {'timestamp': timestamps, 'close_price': close_price_values, 'score': scores}

        df = pd.DataFrame(data, columns=['timestamp', 'close_price', 'score'])
        df = df.set_index('timestamp')
        df = self._remove_duplicate_indexes(df, 'price data')
        return df

    def get_nearest_resampled_price(self, timestamp, transaction_currency, counter_currency, exchange):
        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=exchange,
            index="close_price",
            timestamp=timestamp.timestamp(),
            timestamp_tolerance = 0
        )['values']
        return prices[-1] if len(prices) else None

    def get_price(self, transaction_currency, timestamp, source="binance", counter_currency="BTC", normalize=False):
        return self.get_indicator('close_price', transaction_currency, counter_currency, timestamp, source)[0]


    def get_latest_price(self, transaction_currency, counter_currency="BTC", normalize=False, exchange="binance"):

        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=exchange,
            index="close_price",
        )
        return float(prices['values'][-1]), prices['latest_timestamp'] if len(prices) else (None, None)


    def get_price_nearest_to_timestamp(self, transaction_currency, timestamp, exchange, counter_currency,
                                       max_delta_seconds_past=60*60,
                                       max_delta_seconds_future=60*5):

        results = PriceVolumeHistoryStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=exchange,
            index="close_price",
            timestamp=timestamp.timestamp(),
            timestamp_tolerance=max_delta_seconds_future
        )

        if results['latest_timestamp'] > (timestamp.timestamp() - max_delta_seconds_past):
            return results['values'][-1] if len(results['values']) else None


    def get_timestamp_n_ticks_earlier(self, timestamp, n, transaction_currency, counter_currency, source, resample_period):
        return timestamp - n*60*5


    def get_nearest_db_timestamp(self, timestamp, ticker, exchange="binance"):

        timestamp_tolerance = 60*5

        results = PriceStorage.query(
            ticker=ticker,
            exchange=exchange,
            index="close_price",
            timestamp=timestamp,
            timestamp_tolerance=timestamp_tolerance
        )

        if not results['scores']:
            return None

        return PriceStorage.timestamp_from_score(results['scores'][-1])


    def _extract_indicator_value(self, indicator_name, result):
        result = result.split(':')
        if indicator_name == 'bb_up':
            return float(result[0])
        elif indicator_name == 'bb_mid':
            return float(result[1])
        elif indicator_name == 'bb_low':
            return float(result[2])
        elif indicator_name == 'bb_squeeze':
            return bool(result[1])
        elif indicator_name == 'macd_value':
            return float(result[0])
        elif indicator_name == 'macd_signal':
            return float(result[1])
        elif indicator_name == 'macd_hist':
            return float(result[2])
        elif indicator_name == 'slowd':
            return float(result[1])
        return float(result[0])


    def get_indicator(self, indicator_name, transaction_currency, counter_currency,
                      timestamp, exchange='binance', horizon=PERIODS_1HR, periods_key=None, periods_range=None):

        ticker = f'{transaction_currency}_{counter_currency}'

        # if we're looking for just one indicator, check if we have it cached
        if periods_range is None:
            cached = self.query_data_cache(ticker, exchange, horizon, timestamp, indicator_name)
            if cached is not None:
                return cached
            else:
                logging.debug(f'No cached value found for {indicator_name} at {timestamp}')


        # build the periods key
        if periods_key is None:
            indicator_period = self._default_indicator_period(indicator_name)
            periods_key = horizon * indicator_period


        # we don't have the indicator cached, we'll go to Redis
        params = dict(
            ticker=ticker,
            exchange=exchange,
            timestamp=timestamp,
            periods_key=periods_key,
            timestamp_tolerance=0,
         )

        if periods_range is not None:
            params['periods_range'] = periods_range

        try:
            logging.debug(f'Going to Redis for indicator {indicator_name} for {ticker} at {timestamp}...')

            # remove the periods for sma or ema, as they aren't stored in STORAGE_CLASS
            if indicator_name.startswith('sma') or indicator_name.startswith('ema'):
                indicator_name = indicator_name[:3]

            # if querying for price, we shouldn't use periods_key
            if STORAGE_CLASS[indicator_name] == PriceStorage:
                del params['periods_key']

            # get the results
            results = STORAGE_CLASS[indicator_name].query(**params)

            # do we want to get multiple values?
            if periods_range is None:
                #  we need to return just one value
                if len(results['values']):
                    value = self._extract_indicator_value(indicator_name, results['values'][-1])
                    self.indicator_cache.save_indicator(indicator_name, ticker,
                                                        timestamp, value, exchange, horizon=horizon)
                    return value
            else:
                # periods_range is not None, we are returning an array of values
                # we'll return an array of scores too
                if len(results['values']):
                    return [self._extract_indicator_value(indicator_name, result) for result in results['values']], \
                           [PriceStorage.timestamp_from_score(score) for score in results['scores']]
                else:
                    return [], []

        except IndexError:
            logging.error(f"Unknown indicator {indicator_name}")
        except Exception as e:
            raise e


    def get_indicator_at_previous_timestamp(self, indicator_name, transaction_currency, counter_currency,
                      timestamp, exchange='binance', horizon=PERIODS_1HR, periods_key=None, periods_range=None):
        indicator_value = self.get_indicator(indicator_name, transaction_currency, counter_currency,
                           (timestamp - 5*60),  # TODO check if this hardcoding is OK
                           exchange, horizon, periods_key, periods_range)

        if indicator_value is None:
            logging.debug(f"Indicator {indicator_name} not found at timestamp {timestamp} (previous)")
        return indicator_value


    def build_data_object(self, start_time, end_time, ticker, horizon=PERIODS_1HR,
                          start_cash=START_CASH, start_crypto=START_CRYPTO, exchange='binance'):
        """
        Use this to build Data objects for training GP. This way the built objects will be cached and the retrieved
        values will be used whenever possible instead of going to Redis.
        :param start_time: start time (use a string representation or a timestamp)
        :param end_time: end time (use a string representation or a timestamp)
        :param ticker: ticker in the form of TRANSACTION_COUNTER
        :param horizon: horizon
        :param start_cash: starting cash, used for backtesting
        :param start_crypto: starting crypto, used for backtesting
        :param exchange: exchange
        :return: a Data object
        """
        logging.info(f' >>>>>>>> Building a data object for {ticker}...')
        data = Data(start_time, end_time, ticker, horizon, start_cash, start_crypto, exchange)
        data.btc_usdt_price_df = self.get_resampled_prices_in_range(data.start_time, data.end_time, 'BTC', 'USDT',
                                                                    horizon, exchange)
        self.data_cache.add_data_object(data)
        logging.info('Data object built.')
        return data


    def query_data_cache(self, ticker, exchange, horizon, timestamp, indicator_name):
        # first try to find the indicator within cached data objects
        for data in self.data_cache.objects:
            if data.applicable(ticker, exchange, horizon, timestamp):
                logging.info('Returning value from cache...')
                return data.get_indicator(indicator_name, timestamp)
            if ticker == 'BTC_USDT' and data.start_time <= timestamp <= data.end_time:
                return data.btc_usdt_price_df.loc[timestamp].close_price

        # if not, maybe it's in the indicator cache?
        return self.indicator_cache.get_indicator(indicator_name, ticker, timestamp, exchange, horizon)


DB_INTERFACE = RedisDB()


class Data:

    def __init__(self, start_time, end_time, ticker, horizon, start_cash, start_crypto, exchange, database=DB_INTERFACE):
        self.start_time = self._parse_time(start_time)
        self.end_time = self._parse_time(end_time)
        self.transaction_currency, self.counter_currency = ticker.split('_')
        self.horizon = horizon
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.exchange = exchange
        self.database = database
        self.resample_period = self.horizon  # legacy compatibility
        self.ticker = ticker
        # self.horizon = PERIODS_1HR


        self.price_data = self.database.get_resampled_prices_in_range\
            (self.start_time, self.end_time, self.transaction_currency, self.counter_currency, horizon)

        self.price_data = self.price_data[~self.price_data.index.duplicated(keep='first')]


        # do some sanity checks with data
        if not self.price_data.empty and self.price_data.iloc[0].name > self.start_time + 60*60*8:
            raise Exception(f"The retrieved price data for {self.transaction_currency}-{self.counter_currency} starts "
                            f"{(self.price_data.iloc[0].name - self.start_time)/60:.2f} minutes after "
                            f"the set start time!")

        if not self.price_data.empty and self.end_time - self.price_data.iloc[-1].name > 60*60*8:
            raise Exception(f"The retrieved price data for {self.transaction_currency}-{self.counter_currency} ends "
                            f"{(self.end_time - self.price_data.iloc[-1].name)/60:.2f} minutes before "
                            f"the set end time! (end time = {self.end_time}, data end time = {self.price_data.iloc[-1].name}")

        self._buy_and_hold_benchmark = None
        self._compute_ta_indicators()


    def _parse_time(self, time_input):
        if isinstance(time_input, str):
            time_object = parser.parse(time_input)
            return time_object.timestamp()
        return time_input


    def _fill_indicator_values(self, indicator_name):
        logging.info(f'Retrieving values for {indicator_name}')
        timestamp = self.price_data.index.values[-1]

        periods_range = len(self.price_data.index.values)  # we need this many values
        indicator_values, timestamps = self.database.get_indicator(
            indicator_name, self.transaction_currency, self.counter_currency,
            timestamp,
            periods_range=periods_range
        )

        if not indicator_values or not timestamps:
            logging.warning(f'   -> unable to fill values for {indicator_name}, some computations will not work!')
            return [np.nan] * len(self.price_data)  # TODO: verify that this is OK

        df = pd.DataFrame({'timestamp': timestamps, 'indicator': indicator_values})
        df = df[~df.index.duplicated(keep='first')]
        result = pd.merge(self.price_data, df, how='left', on=['timestamp'])
        return np.array(result['indicator'])


    def get_indicator(self, indicator_name, timestamp):
        try:
            value = self.indicators[indicator_name][self.price_data.index.get_loc(timestamp)]
            #logging.info('Successfully retrieved indicator value')
            return value
        except Exception as e:
            #logging.error(f'Unable to retrieve indicator: {e}')
            return None


    @time_performance
    def _compute_ta_indicators(self):
        """
        prices = np.array(self.price_data.close_price, dtype=float)
        high_prices = np.array(self.price_data.high_price, dtype=float)
        low_prices = np.array(self.price_data.low_price, dtype=float)


        volumes = np.array(self.price_data.close_volume, dtype=float)
        if np.isnan(volumes).all():
            logging.warning(f'Unable to load valid volume data for for {self.transaction_currency}-{self.counter_currency}.')
            self.sma50_volume = volumes[TICKS_FOR_PRECOMPUTE:]
        else:
            self.sma50_volume = talib.SMA(volumes, timeperiod=50)[TICKS_FOR_PRECOMPUTE:]

        self.close_volume = volumes[TICKS_FOR_PRECOMPUTE:]
        """

        supported_indicators = ['rsi', 'sma20', 'sma50', 'sma200', 'ema20','ema50', 'ema200',
                                'bb_up', 'bb_mid', 'bb_low', 'bb_squeeze', 'macd', 'macd_signal',
                                'macd_hist', 'adx', 'slowd',]

        self.indicators = {}
        for indicator_name in supported_indicators:
            self.indicators[indicator_name] = self._fill_indicator_values(indicator_name)

        self.close_price = self.price_data.as_matrix(columns=["close_price"])
        self.indicators['close_price'] = self.close_price
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.close_price) == len(self.timestamps)


    def applicable(self, ticker, exchange, horizon, timestamp):
        return ticker == self.ticker and exchange == self.exchange \
               and horizon == self.horizon and self.start_time <= timestamp <= self.end_time


    def __str__(self):
        return self.to_string(self.transaction_currency, self.counter_currency, self.start_time, self.end_time)

    @staticmethod
    def to_string(transaction_currency, counter_currency, start_time, end_time):
        return f"{transaction_currency}-{counter_currency}-{int(start_time)}-{int(end_time)}"

    @time_performance
    def _build_buy_and_hold_benchmark(self):
        from apps.backtesting.backtester_ticks import TickDrivenBacktester
        from apps.backtesting.tick_provider import PriceDataframeTickProvider
        self._buy_and_hold_benchmark = TickDrivenBacktester.build_benchmark(
            transaction_currency=self.transaction_currency,
            counter_currency=self.counter_currency,
            start_cash=self.start_cash,
            start_crypto=self.start_crypto,
            start_time=self.start_time,
            end_time=self.end_time,
            source=self.exchange,
            tick_provider=PriceDataframeTickProvider(
                price_df=self.price_data,
                transaction_currency=self.transaction_currency,
                counter_currency=self.counter_currency,
                source=self.exchange,
                resample_period=self.resample_period,
            ),
            database=self.database
        )


    @property
    def buy_and_hold_benchmark(self):
        if self._buy_and_hold_benchmark is None:
            self._build_buy_and_hold_benchmark()   # lazily evaluate only when first invoked
        return self._buy_and_hold_benchmark

    def get_all_indicator_values(self, indicator_name):
        return self.indicators[indicator_name]

    def _filter_fields(self, fields, individual_str):
        filtered_dict = {}
        for field in fields:
            if not field.lower() in individual_str and field != "Close price" and field != "MACD signal":
                continue
            if field == "MACD signal" and "macd" not in individual_str:
                continue
            filtered_dict[field] = fields[field]
        return filtered_dict


    def plot(self, orders=None, individual_str=None):
        timestamps = self.price_data.index
        data_primary_axis = {
            "Close price" : self.price_data.close_price,
            "SMA50": self.get_all_indicator_values('sma50'),
            "EMA50": self.get_all_indicator_values('ema50'),
            "SMA200": self.get_all_indicator_values('sma200'),
            "EMA200": self.get_all_indicator_values('ema200'),

        }

        data_secondary_axis = {
            "ADX": self.get_all_indicator_values('adx'),
            "MACD": self.get_all_indicator_values('macd'),
            "MACD signal": self.get_all_indicator_values('macd_signal'),
            "RSI": self.get_all_indicator_values('rsi')
        }

        if individual_str is not None:
            data_primary_axis = self._filter_fields(data_primary_axis, individual_str)
            data_secondary_axis = self._filter_fields(data_secondary_axis, individual_str)

        from apps.backtesting.charting import time_series_chart

        time_series_chart(timestamps, series_dict_primary=data_primary_axis, series_dict_secondary=data_secondary_axis,
                          title=f"{self.transaction_currency} - {self.counter_currency}", orders=orders)


