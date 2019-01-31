from enum import Enum

import numpy as np
from dateutil import parser

from apps.TA.storages.data.price import PriceStorage
from apps.TA.storages.data.pv_history import PriceVolumeHistoryStorage
#from apps.backtesting.backtester_ticks import TickDrivenBacktester
#from apps.backtesting.charting import time_series_chart
#from apps.backtesting.signals import Signal
from abc import ABC
from apps.TA import PERIODS_1HR
import pandas as pd
import psycopg2
import psycopg2.extras
import logging

#from apps.backtesting.tick_provider import PriceDataframeTickProvider
from apps.backtesting.utils import time_performance


class NoPriceDataException(Exception):
    pass

class CounterCurrency(Enum):
    BTC = 0
    ETH = 1
    USDT = 2
    XMR = 3


class Horizon(Enum):
    any = None
    short = 0
    medium = 1
    long = 2

class Strength(Enum):
    any = None
    short = 1
    medium = 2
    long = 3

#(BTC, ETH, USDT, XMR) = list(range(4))

from apps.TA.indicators.momentum import rsi, stochrsi, adx, macd, mom, stoch, willr
from apps.TA.indicators.overlap import sma, ema, wma, bbands, ht_trendline
from apps.TA.indicators.events import bbands_squeeze_180min


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
    'willr': willr.WillrStorage
}



class Database(ABC):

    def log_price_error(self, msg, counter_currency):
        """
        If our counter currency is not BTC and we can't find prices, we're probably dealing with an altcoin
        that trades only against BTC. In that case we output a debug-level msg. If price against BTC isn't
        found, we output an error-level msg.
        :param msg: message to output
        :param counter_currency: counter currency for which we're finding the price
        :return:
        """
        if counter_currency == "BTC":
            logging.error(msg)
        else:
            logging.debug(msg)


    def convert_value_to_USDT(self, value, timestamp, transaction_currency, source):
        if value == 0:
            return 0
        if transaction_currency == "USDT": # already in USDT
            return value
        try:
            value_USDT = value * self.get_price(transaction_currency, timestamp, source, "USDT") # if trading against USDT
            # print("Found USDT price data for {}".format(transaction_currency))
            return value_USDT
        except:
            # print("Couldn't find USDT price data for {}".format(transaction_currency))
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




class PostgresDatabaseConnection(Database):

    def __init__(self):
        from settings import postgres_connection_string
        self.conn = psycopg2.connect(postgres_connection_string)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def get_cursor(self):
        return self.cursor

    def get_connection(self):
        return self.conn

    def execute(self, statement, params):
        self.cursor.execute(statement, params)
        return self.cursor


    def get_resampled_prices_in_range(self, start_time, end_time, transaction_currency, counter_currency, resample_period, source=0,
                                      normalize=True):
        resampled_price_range_query = """SELECT timestamp, close_price, high_price, low_price, close_volume
                                         FROM indicator_priceresampl 
                                         WHERE transaction_currency = %s 
                                         AND counter_currency = %s 
                                         AND source = %s 
                                         AND timestamp >= %s AND timestamp <= %s
                                         AND resample_period = %s
                                         AND close_price is not null
                                         ORDER BY timestamp ASC"""
        counter_currency_id = CounterCurrency[counter_currency].value
        connection = self.get_connection()
        price_data = pd.read_sql(resampled_price_range_query, con=connection, params=(transaction_currency,
                                                                                   counter_currency_id,
                                                                                   source,
                                                                                   start_time,
                                                                                   end_time,
                                                                                   resample_period),
                                 index_col="timestamp")
        if normalize:
            price_data.loc[:, 'close_price'] /= 1E8
            price_data.loc[:, 'high_price'] /= 1E8
            price_data.loc[:, 'low_price'] /= 1E8
        return price_data


    def get_nearest_resampled_price(self, timestamp, transaction_currency, counter_currency, resample_period, source, normalize):
        query = """SELECT timestamp, close_price, high_price, low_price, close_volume
                                         FROM indicator_priceresampl 
                                         WHERE transaction_currency = %s 
                                         AND counter_currency = %s 
                                         AND source = %s 
                                         AND timestamp >= %s
                                         AND resample_period = %s
                                         AND close_price is not null
                                         ORDER BY timestamp ASC LIMIT 1"""
        counter_currency_id = CounterCurrency[counter_currency].value
        connection = self.get_connection()
        price_data = pd.read_sql(query, con=connection, params=(transaction_currency,
                                                                counter_currency_id,
                                                                source,
                                                                timestamp,
                                                                resample_period), index_col="timestamp")
        if normalize:
            price_data.loc[:, 'close_price'] /= 1E8
            price_data.loc[:, 'high_price'] /= 1E8
            price_data.loc[:, 'low_price'] /= 1E8
        return price_data


    def get_filtered_signals(self, signal_type=None, transaction_currency=None, start_time=None, end_time=None, horizon=Horizon.any,
                             counter_currency=None, strength=Strength.any, source=None, resample_period=None, return_df=False,
                             normalize=True):
        query = """ SELECT signal_signal.signal, trend, horizon, strength_value, strength_max, price, price_change, 
                    timestamp, rsi_value, transaction_currency, counter_currency, source, resample_period FROM signal_signal 
                     """
        additions = []
        params = []
        if signal_type is not None:
            additions.append("signal_signal.signal=%s")
            params.append(signal_type)
        if transaction_currency is not None:
            additions.append("transaction_currency = %s")
            params.append(transaction_currency)
        if start_time is not None:
            additions.append("timestamp >= %s")
            params.append(start_time)
        if end_time is not None:
            additions.append("timestamp <= %s")
            params.append(end_time)
        if horizon.value is not None:
            additions.append("horizon = %s")
            params.append(horizon.value)
        if counter_currency is not None:
            additions.append("counter_currency = %s")
            params.append(CounterCurrency[counter_currency].value)
        if strength.value is not None:
            additions.append("strength_value = %s")
            params.append(strength.value)
        if source is not None:
            additions.append("source = %s")
            params.append(source)
        if resample_period is not None:
            additions.append("resample_period = %s")
            params.append(resample_period)


        if len(additions) > 0:
            query += "WHERE {}".format(" AND ".join(additions))
            params = tuple(params)
        query += ' ORDER BY timestamp'
        if return_df:
            connection = db_interface.get_connection()
            signals_df = pd.read_sql(query, con=connection, params=params, index_col="timestamp")
            signals_df['counter_currency'] = [CounterCurrency(counter_currency).name
                                              for counter_currency in signals_df.counter_currency]
            if normalize:
                signals_df['price'] = signals_df['price']/1E8
            return signals_df

        # otherwise, we return a list of Signal objectss
        cursor = self.execute(query, params)

        signals = []

        for (signal_type, trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value,
             transaction_currency, counter_currency, source, resample_period) in cursor:
            if len(trend) > 5:   # hacky solution for one instance of bad data
                continue
            signals.append(Signal(signal_type, trend, horizon, strength_value, strength_max,
                                  price/1E8 if normalize else price,  price_change, timestamp, rsi_value, transaction_currency,
                                  CounterCurrency(counter_currency).name, source, resample_period))
        return signals



    def get_price(self, currency, timestamp, source, counter_currency="BTC", normalize=True):
        price_query = """SELECT price FROM indicator_price 
                                    WHERE transaction_currency = %s
                                    AND timestamp = %s
                                    AND source = %s
                                    AND counter_currency = %s;"""

        if currency == counter_currency:
            return 1
        counter_currency_id = CounterCurrency[counter_currency].value
        cursor = self.execute(price_query, params=(currency, timestamp, source, counter_currency_id))

        price = cursor.fetchall()
        if cursor.rowcount == 0:
            price = self.get_price_nearest_to_timestamp(currency, timestamp, source, counter_currency)
        else:
            assert cursor.rowcount == 1
            price = price[0][0]

        if normalize:
            return price / 1E8
        else:
            return price


    price_in_range_query_asc = """SELECT price, timestamp 
                                   FROM indicator_price 
                                   WHERE transaction_currency = %s AND counter_currency = %s 
                                        AND source = %s AND timestamp >= %s 
                                        AND timestamp <= %s ORDER BY timestamp ASC"""
    price_in_range_query_desc = """SELECT price, timestamp 
                                   FROM indicator_price 
                                   WHERE transaction_currency = %s AND counter_currency = %s 
                                        AND source = %s AND timestamp >= %s 
                                        AND timestamp <= %s ORDER BY timestamp DESC"""


    def get_price_nearest_to_timestamp(self, currency, timestamp, source, counter_currency, max_delta_seconds_past=60*60,
                                       max_delta_seconds_future=60*5):

        counter_currency_id = CounterCurrency[counter_currency].value
        cursor = self.execute(self.price_in_range_query_desc, params=(currency, counter_currency_id, source,
                                                     timestamp - max_delta_seconds_past, timestamp))
        history = cursor.fetchall()
        cursor = self.execute(self.price_in_range_query_asc, params=(currency, counter_currency_id, source,
                                                              timestamp, timestamp + max_delta_seconds_future))
        future = cursor.fetchall()

        if len(history) == 0:

            self.log_price_error(f"No historical price data for {currency}-{counter_currency} in "
                            f"{max_delta_seconds_past/60} minutes before timestamp {timestamp}...",
                            counter_currency)

            if len(future) == 0:
                self.log_price_error("No future data found.", counter_currency)
                raise NoPriceDataException()
            else:
                logging.warning("Returning future price...")

                return future[0][0]
        else:
            logging.debug("Returning historical price data for timestamp {} (difference of {} minutes)"
                  .format(timestamp,(timestamp - history[0][1])/60))
            return history[0][0]

    def get_prices_in_range(self, start_time, end_time, transaction_currency, counter_currency, source):
        counter_currency_id = CounterCurrency[counter_currency].value
        connection = self.get_connection()
        price_data = pd.read_sql(self.price_in_range_query_asc, con=connection, params=(transaction_currency,
                                                                                   counter_currency_id,
                                                                                   source,
                                                                                   start_time,
                                                                                   end_time), index_col="timestamp")
        return price_data


    def get_volumes_in_range(self, start_time, end_time, transaction_currency, counter_currency, source):
        volume_in_range_query_asc = """SELECT volume, timestamp 
                                       FROM indicator_volume 
                                       WHERE transaction_currency = %s AND counter_currency = %s 
                                            AND source = %s AND timestamp >= %s 
                                            AND timestamp <= %s ORDER BY timestamp ASC"""
        counter_currency_id = CounterCurrency[counter_currency].value
        connection = self.get_connection()
        volume_data = pd.read_sql(volume_in_range_query_asc, con=connection, params=(transaction_currency,
                                                                                    counter_currency_id,
                                                                                    source,
                                                                                    start_time,
                                                                                    end_time),
                                 index_col="timestamp")
        return volume_data


    def get_currencies_trading_against_counter(self, counter_currency, source):
        trading_against_counter_query = """SELECT DISTINCT(transaction_currency) 
                                           FROM indicator_price WHERE counter_currency = %s AND source = %s"""

        counter_currency_id = CounterCurrency[counter_currency].value
        cursor = self.execute(trading_against_counter_query, params=(counter_currency_id,source,))
        data = cursor.fetchall()
        currencies = []
        for currency in data:
            currencies.append(currency[0])
        return currencies


    def get_currencies_for_signal(self, counter_currency, signal, source):
        trading_against_counter_and_signal_query = """SELECT DISTINCT(transaction_currency) FROM signal_signal 
                                                      WHERE counter_currency = %s 
                                                      AND signal_signal.signal = %s AND source = %s"""
        counter_currency_id = CounterCurrency[counter_currency].value
        cursor = self.execute(trading_against_counter_and_signal_query, params=(counter_currency_id, signal, source,))
        data = cursor.fetchall()
        currencies = []
        for currency in data:
            currencies.append(currency[0])
        return currencies


    def get_timestamp_n_ticks_earlier(self, timestamp, n, transaction_currency, counter_currency, source, resample_period):
        query = """SELECT DISTINCT timestamp 
                   FROM indicator_priceresampl 
                   WHERE timestamp < %s AND transaction_currency = %s AND counter_currency = %s 
                   AND source=%s AND resample_period=%s ORDER BY timestamp DESC LIMIT %s
        """
        counter_currency_id = CounterCurrency[counter_currency].value
        cursor = self.execute(query, params=(timestamp, transaction_currency, counter_currency_id, source, resample_period, n,))
        data = cursor.fetchall()
        assert len(data) == n
        return data[-1][0]


    def count_signals_ocurring_at_the_same_time(self, type1, type2, start_time, end_time):
        query = """select count(*) from 
                        (select * from signal_signal 
                            where signal = %s 
                            and timestamp >= %s and timestamp <= %s) t1 
                        inner join 
                        (select * from signal_signal 
                            where signal = %s 
                            and timestamp >= %s and timestamp <= %s) t2 
                        on  t1.timestamp = t2.timestamp 
                            and t1.transaction_currency = t2.transaction_currency 
                            and t1.counter_currency = t2.counter_currency 
                            and t1.source = t2.source 
                            and t1.resample_period = t2.resample_period"""
        cursor = self.execute(query, params=(type1, start_time, end_time, type2, start_time, end_time,))
        data = cursor.fetchall()
        return data[-1][0]



class RedisDB(Database):
    default_indicator_periods = {
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

    def _default_indicator_period(self, indicator_name):
        if indicator_name.startswith('sma') or indicator_name.startswith('ema'):
            return int(indicator_name[3:])
        else:
            return self.default_indicator_periods.get(indicator_name, 1)



    def get_resampled_prices_in_range(self, start_time, end_time,
                                      transaction_currency, counter_currency, horizon,
                                      source="binance", normalize=False):
        """

        :param start_time: start of time range (timestamp in seconds)
        :param end_time: end of time range (timestamp in seconds)
        :param transaction_currency: transaction currency
        :param counter_currency: counter currency
        :param index:
        :param source: only "binance"
        :param normalize: if True, the returned price from database should be divided by 1E8
        :return: a dataframe with open, high, low and closing price and close volume
        """

        # TODO add support for horizon

        if end_time is None or start_time is None:
            logging.error('Start or end time is set to None!')

        periods_range = (end_time - start_time) // (5*60)

        close_prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=source,
            index="close_price",
            timestamp=end_time,
            periods_range=periods_range,
            timestamp_tolerance=0
        ) # ['values']

        high_prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=source,
            index="high_price",
            timestamp=end_time,
            periods_range=periods_range,
            timestamp_tolerance=0
        ) # ['values']

        # assert len(close_prices['scores']) == len(high_prices['scores']) TODO @tomcounsell make this work, currently missing data!

        if close_prices['values_count'] == 0:
            logging.error('No close prices returned!')

        timestamps = [PriceStorage.timestamp_from_score(float(score)) for score in close_prices['scores']]
        close_price_values = list(map(float, close_prices['values']))
        scores = list(map(float, close_prices['scores']))

        high_prices = high_prices['values']
        data = {'timestamp': timestamps, 'close_price': close_price_values,
                'high_price': close_price_values, 'low_price': close_price_values,
                'close_volume': close_price_values, 'score': scores}
                # TODO fix stuff with missing high prices and other values

        df = pd.DataFrame(data, columns=['timestamp', 'close_price', 'high_price', 'low_price', 'close_volume', 'score'])
        df = df.set_index('timestamp')

        return df


    def get_nearest_resampled_price(self, timestamp, transaction_currency, counter_currency, resample_period, source, normalize=False):
        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp.timestamp(),
            timestamp_tolerance = 0
        )['values']
        return prices[-1] if len(prices) else None


    def get_price(self, transaction_currency, timestamp, source="binance", counter_currency="BTC", normalize=False):

        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=source,
            index="close_price",
            timestamp=timestamp,
            timestamp_tolerance=0
        )['values']
        return float(prices[-1]) if len(prices) else None


    def get_latest_price(self, transaction_currency, counter_currency="BTC", normalize=False):

        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
        )
        return float(prices['values'][-1]), prices['latest_timestamp'] if len(prices) else (None, None)


    def get_price_nearest_to_timestamp(self, transaction_currency, timestamp, source, counter_currency,
                                       max_delta_seconds_past=60*60,
                                       max_delta_seconds_future=60*5):

        results = PriceVolumeHistoryStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp.timestamp(),
            timestamp_tolerance = max_delta_seconds_future
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


    def get_indicator(self, indicator_name, transaction_currency, counter_currency,
                      timestamp, exchange='binance', horizon=PERIODS_1HR, periods_key=None, periods_range=None):

        if periods_range is None:
            ticker = f'{transaction_currency}_{counter_currency}'
            cached = self.query_data_cache(ticker, exchange, horizon, timestamp, indicator_name)
            if cached is not None:
                return cached
            else:
                logging.info(f'No cached value found for {indicator_name}')

        if periods_key is None:
            indicator_period = self._default_indicator_period(indicator_name)
            periods_key = horizon * indicator_period

        # query Redis to get indicator value at timestamp (+- periods range)
        if indicator_name == 'close_price':
            return self.get_price(transaction_currency=transaction_currency,
                                  counter_currency=counter_currency,
                                  timestamp=timestamp)

        params = dict(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange=exchange,
            timestamp=timestamp,
            periods_key=periods_key,
            timestamp_tolerance=0,
         )

        try:
            if indicator_name.startswith('sma') or indicator_name.startswith('ema'):
                indicator_name = indicator_name[:3]
            results = STORAGE_CLASS[indicator_name].query(**params)

            # do we want to get multiple values?
            if periods_range is None:
                if len(results['values']):
                    return self._extract_indicator_value(indicator_name, results['values'][-1])
            else:
                # periods_range is not None, we are returning an array of values
                if len(results['values']):
                    return [self._extract_indicator_value(indicator_name, result) for result in results['values']], \
                           [PriceStorage.timestamp_from_score(score) for score in results['scores']]
                else:
                    return [], []

        except IndexError:
            return "unknown indicator_name"
        except Exception as e:
            raise e

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

    def get_indicator_at_previous_timestamp(self, indicator_name, transaction_currency, counter_currency,
                      timestamp, exchange='binance', horizon=PERIODS_1HR, periods_key=None, periods_range=None):
        indicator_value = self.get_indicator(indicator_name, transaction_currency, counter_currency,
                           (timestamp - 5*60), # TODO check if this hardcoding is OK
                           exchange, horizon, periods_key, periods_range)

        if indicator_value is None:
            logging.debug(f"Indicator {indicator_name} not found at timestamp {timestamp} (previous)")
        return indicator_value


    def build_data_object(self, start_time, end_time, ticker, horizon, start_cash, start_crypto, exchange):
        data = Data(start_time, end_time, ticker, horizon, start_cash, start_crypto, exchange)
        self.cached_data_objects.append(data)
        logging.info('Built data object!')
        return data

    def query_data_cache(self, ticker, exchange, horizon, timestamp, indicator_name):
        for data in self.cached_data_objects:
            if data.applicable(ticker, exchange, horizon, timestamp):
                logging.info('Returning value from cache...')
                return data.get_indicator(indicator_name, timestamp)

    def __init__(self):
        self.cached_data_objects = []




class CachedRedis(RedisDB):

    def _remove_duplicate_indexes(self, df, data_description=''):
        df_len = len(df)
        cleaned = df[~df.index.duplicated(keep='first')]
        if (len(cleaned) != df_len):
            logging.critical(f'Encountered duplicate values in {data_description} data, {df_len} '
                             f'values loaded, {len(cleaned)} remain after removal.')

        return cleaned


    def __init__(self, start_time, end_time, transaction_currency, counter_currency, horizon, source="binance",
                 normalize=False):
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency

        self.price_df = self.get_resampled_prices_in_range(start_time, end_time, transaction_currency,
                                                           counter_currency, horizon, source)
        self.price_df = self._remove_duplicate_indexes(self.price_df,
                                                       f'price ticker {transaction_currency}_{counter_currency}')

        self.btc_usdt_price_df = self.get_resampled_prices_in_range(start_time, end_time, 'BTC', 'USDT', horizon, source)
        self.btc_usdt_price_df = self._remove_duplicate_indexes(self.btc_usdt_price_df, 'BTC_USDT price data')
        self.hits = 0

    def get_price(self, transaction_currency, timestamp, source="binance", counter_currency="BTC", normalize=False):
        if transaction_currency == self.transaction_currency and counter_currency == self.counter_currency:
            logging.debug(f'Total hits: {self.hits}')
            self.hits += 1
            return self.price_df.loc[timestamp].close_price
        elif transaction_currency == 'BTC' and counter_currency == 'USDT':
            logging.debug(f'Total hits: {self.hits}')
            self.hits += 1
            return self.btc_usdt_price_df.loc[timestamp].close_price
        else:
            logging.warning('No cached price data! Querying Redis...')
            return super().get_price(transaction_currency, timestamp, source="binance", counter_currency="BTC",
                                     normalize=False)


#postgres_db = PostgresDatabaseConnection()
db_interface = RedisDB()


class Data:

    def applicable(self, ticker, exchange, horizon, timestamp):
        return ticker == self.ticker and exchange == self.exchange \
               and horizon == self.horizon and self.start_time <= timestamp <= self.end_time

    def _parse_time(self, time_input):
        if isinstance(time_input, str):
            time_object = parser.parse(time_input)
            return time_object.timestamp()
        return time_input

    def __init__(self, start_time, end_time, ticker, horizon, start_cash, start_crypto, exchange, database=db_interface):
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
            return None

        df = pd.DataFrame({'timestamp': timestamps, 'indicator': indicator_values})
        df = df[~df.index.duplicated(keep='first')]
        result = pd.merge(self.price_data, df, how='left', on=['timestamp'])
        return np.array(result['indicator'])

    def get_indicator(self, indicator_name, timestamp):
        try:
            return self.indicators[indicator_name][self.price_data.index.get_loc(timestamp)]
        except:
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
            tick_provider=PriceDataframeTickProvider(self.price_data,
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



    def plot(self, orders=None, individual_str=None):
        timestamps = self.price_data.index
        data_primary_axis = {
            "Close price" : self.price_data.close_price,
            "SMA50": self.sma50,
            "EMA50": self.ema50,
            "SMA200": self.sma200,
            "EMA200": self.ema200,

        }

        data_secondary_axis = {
            "ADX": self.adx,
            "MACD": self.macd,
            "MACD signal": self.macd_signal,
            "RSI": self.rsi
        }

        if individual_str is not None:
            data_primary_axis = self._filter_fields(data_primary_axis, individual_str)
            data_secondary_axis = self._filter_fields(data_secondary_axis, individual_str)



        time_series_chart(timestamps, series_dict_primary=data_primary_axis, series_dict_secondary=data_secondary_axis,
                          title=f"{self.transaction_currency} - {self.counter_currency}", orders=orders)