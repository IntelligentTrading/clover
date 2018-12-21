from datetime import timedelta
from enum import Enum

from apps.TA.storages.data.price import PriceStorage
from apps.TA.storages.data.pv_history import PriceVolumeHistoryStorage
from apps.backtesting.config import postgres_connection_string
from apps.backtesting.signals import Signal
from abc import ABC
import pandas as pd
import psycopg2
import psycopg2.extras
import logging

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
            connection = postgres_db.get_connection()
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
    from settings.redis_db import database

    def __init__(self):
        pass

    def get_resampled_prices_in_range(self, start_time, end_time,
                                      transaction_currency, counter_currency, candle_duration_minutes=5,
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

        periods_range = (end_time - start_time) // (candle_duration_minutes*60)

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
        close_prices = list(map(float, close_prices['values']))
        high_prices = high_prices['values']
        data = {'timestamp': timestamps, 'close_price': close_prices,
                'high_price': close_prices, 'low_price': close_prices, 'close_volume': close_prices}
                # TODO fix stuff with missing high prices and other values

        df = pd.DataFrame(data, columns=['timestamp', 'close_price', 'high_price', 'low_price', 'close_volume'])
        df = df.set_index('timestamp')

        return df


    def get_nearest_resampled_price(self, timestamp, transaction_currency, counter_currency, resample_period, source, normalize):
        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp.timestamp(),
            timestamp_tolerance = 0
        )['values']
        return prices[-1] if len(prices) else None


    def get_price(self, transaction_currency, timestamp, source, counter_currency="BTC", normalize=True):

        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp,
            timestamp_tolerance = 0
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

        prices = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp - (resample_period*n),
            timestamp_tolerance=0
        )['values']
        return prices[-1] if len(prices) else None


    def get_nearest_db_timestamp(self, timestamp, transaction_currency, counter_currency, source=None, resample_period=None):

        timestamp_tolerance = 60*5

        results = PriceStorage.query(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            index="close_price",
            timestamp=timestamp,
            timestamp_tolerance=timestamp_tolerance
        )
        if results['latest_timestamp'] > (timestamp - timestamp_tolerance):
            return results['latest_timestamp']


    def get_indicator(self, indicator_name, transaction_currency, counter_currency, timestamp, resample_period, source='binance'):
        # query Redis to get indicator value at timestamp

        params = dict(
            ticker=f'{transaction_currency}_{counter_currency}',
            exchange="binance",
            timestamp=timestamp,
            periods_key = resample_period
        )

        from apps.TA.indicators.momentum import rsi, stochrsi, adx, macd, mom, stoch
        from apps.TA.indicators.overlap import sma, ema, wma, bbands, ht_trendline

        storage_class = {
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
            'ht_trendline': ht_trendline.HtTrendlineStorage,
            'slowd': stoch.StochStorage,

        }

        try:
            results = storage_class[indicator_name].query(**params)
            if len(results['values']):
                result = results['values'][-1].split(':')
                if indicator_name == 'bb_up':
                    return result[0]
                elif indicator_name == 'bb_mid':
                    return result[1]
                elif indicator_name == 'bb_low':
                    return result[2]
                elif indicator_name == 'macd_value':
                    return result[0]
                elif indicator_name == 'macd_signal':
                    return result[1]
                elif indicator_name == 'macd_hist':
                    return result[2]
                elif indicator_name == 'slowd':
                    return result[1]

            return results['values'][-1] if len(results['values']) else None

        except IndexError:
            return "unknown indicator_name"
        except Exception as e:
            raise e

    def get_indicator_at_previous_timestamp(self, indicator_name, transaction_currency, counter_currency, timestamp, resample_period, source):
        self.get_indicator(indicator_name, transaction_currency, counter_currency,
                           (timestamp - timedelta(seconds=resample_period*60)),
                           resample_period, source)



class RedisDummyDB(Database):

    # TODO: replace all SQL queries with Redis queries

    # TODO: adapt or remove
    def __init__(self):
        self.conn = psycopg2.connect(postgres_connection_string)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # TODO: remove these three methods, they're needed for Postgres
    def get_cursor(self):
        return self.cursor

    def get_connection(self):
        return self.conn

    def execute(self, statement, params):
        self.cursor.execute(statement, params)
        return self.cursor


    def get_resampled_prices_in_range(self, start_time, end_time, transaction_currency,
                                      counter_currency, resample_period, source=0, normalize=True):
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


    # these are non-resampled prices
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

    def get_price_nearest_to_timestamp(self, currency, timestamp, source, counter_currency,
                                       max_delta_seconds_past=60*60,
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

    def get_indicator(self, indicator_name, transaction_currency, counter_currency, resample_period, source):
        # query Redis to get indicator value at timestamp
        pass

    def get_indicator_at_previous_timestamp(self, indicator_name, transaction_currency, counter_currency, resample_period, source):
        # query Redis to get indicator_name at timestamp-1
        pass


postgres_db = PostgresDatabaseConnection()
redis_db = RedisDB()
