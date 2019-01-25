import numpy as np
import pandas as pd
import talib
import logging
from dateutil import parser
from apps.backtesting.tick_provider import PriceDataframeTickProvider
from apps.backtesting.backtester_ticks import TickDrivenBacktester
from apps.backtesting.data_sources import db_interface
from apps.backtesting.charting import time_series_chart
from apps.TA import HORIZONS, PERIODS_4HR, PERIODS_1HR
from apps.backtesting.utils import time_performance

DEBUG = True


# temporarily suspend strategies logging warnings: buy&hold strategy triggers warnings
# as our buy has to be triggered AFTER the minimum strategy initialization period
# determined by the longest_function_history_size parameter of the used grammar
strategy_logger = logging.getLogger("strategies")
strategy_logger.setLevel(logging.ERROR)

TICKS_FOR_PRECOMPUTE = 200



class Data:

    def _parse_time(self, time_input):
        if isinstance(time_input, str):
            time_object = parser.parse(time_input)
            return time_object.timestamp()
        return time_input

    def __init__(self, start_time, end_time, ticker, horizon, start_cash, start_crypto, source, database=db_interface):
        self.start_time = self._parse_time(start_time)
        self.end_time = self._parse_time(end_time)
        self.transaction_currency, self.counter_currency = ticker.split('_')
        self.horizon = horizon
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.source = source
        self.database = database
        self.resample_period = self.horizon  # legacy compatibility
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

    @time_performance
    def _fill_indicator_values_legacy(self, indicator_name, indicator_period=1):
        logging.info(f'Retrieving values for {indicator_name}, indicator_period={indicator_period}')
        result = []
        for i, timestamp in enumerate(self.price_data.index.values):
            indicator = self.database.get_indicator(
                indicator_name,
                self.transaction_currency,
                self.counter_currency,
                timestamp,
                resample_period=self.horizon * indicator_period,
                source='binance')
            # logging.info(f'Retrieved value {i}: {indicator}')
            result.append(float(indicator) if indicator is not None else None)

            if indicator is None and DEBUG:
                logging.warning('Empty indicator value, exiting...')
                break
        return result

    def _fill_indicator_values(self, indicator_name, indicator_period=1):
        logging.info(f'Retrieving values for {indicator_name}, indicator_period={indicator_period}')
        timestamp = self.price_data.index.values[-1]

        periods_range = len(self.price_data.index.values)  # we need this many values
        indicator_values, timestamps = self.database.get_indicator(
            indicator_name,
            self.transaction_currency,
            self.counter_currency,
            timestamp,
            resample_period=self.horizon * indicator_period,
            source='binance',
            periods_range=periods_range
        )

        if not indicator_values or not timestamps:
            return None

        df = pd.DataFrame({'timestamp': timestamps, 'indicator': indicator_values})
        df = df[~df.index.duplicated(keep='first')]
        result = pd.merge(self.price_data, df, how='left', on=['timestamp'])
        return np.array(result['indicator'])

    @time_performance
    def _compute_ta_indicators(self):
        prices = np.array(self.price_data.close_price, dtype=float)
        high_prices = np.array(self.price_data.high_price, dtype=float)
        low_prices = np.array(self.price_data.low_price, dtype=float)

        """
        volumes = np.array(self.price_data.close_volume, dtype=float)
        if np.isnan(volumes).all():
            logging.warning(f'Unable to load valid volume data for for {self.transaction_currency}-{self.counter_currency}.')
            self.sma50_volume = volumes[TICKS_FOR_PRECOMPUTE:]
        else:
            self.sma50_volume = talib.SMA(volumes, timeperiod=50)[TICKS_FOR_PRECOMPUTE:]

        self.close_volume = volumes[TICKS_FOR_PRECOMPUTE:]
        """


        self.rsi = self._fill_indicator_values('rsi', 14)
        self.sma20 = self._fill_indicator_values('sma', 20)
        self.ema20 = self._fill_indicator_values('ema', 20)
        self.sma50 = self._fill_indicator_values('sma', 50)
        self.ema50 = self._fill_indicator_values('ema', 50)
        self.sma200 = self._fill_indicator_values('sma', 200)
        self.ema200 = self._fill_indicator_values('ema', 200)
        # self.ema21 = self._fill_indicator_values('ema', 21)
        # self.ema55 = self._fill_indicator_values('ema', 55)
        self.bb_up = self._fill_indicator_values('bb_up', 5)
        self.bb_mid = self._fill_indicator_values('bb_mid', 5)
        self.bb_low = self._fill_indicator_values('bb_low', 5)
        self.bb_width = [(up - low)/mid for up, low, mid in zip(self.bb_up, self.bb_low, self.bb_mid)]
        self.bb_squeeze = self._fill_indicator_values('bb_squeeze', 1)

        self.macd = self._fill_indicator_values('macd_value', 26)
        self.macd_signal = self._fill_indicator_values('macd_signal', 26)
        self.macd_hist = self._fill_indicator_values('macd_hist', 26)

        self.adx = self._fill_indicator_values('adx')
        self.slowd = self._fill_indicator_values('slowd', 5)

        self.close_price = self.price_data.as_matrix(columns=["close_price"])
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.close_price) == len(self.timestamps)


    def to_dataframe(self):
        df = self.price_data.copy(deep=True)
        df['RSI'] = pd.Series(self.rsi, index=df.index)
        df['SMA20'] = pd.Series(self.sma20, index=df.index)
        df['SMA50'] = pd.Series(self.sma50, index=df.index)
        df['SMA200'] = pd.Series(self.sma200, index=df.index)
        df['EMA20'] = pd.Series(self.ema20, index=df.index)
        df['EMA50'] = pd.Series(self.ema50, index=df.index)
        df['EMA200'] = pd.Series(self.ema200, index=df.index)
        df['ADX'] = pd.Series(self.adx, index=df.index)


    def __str__(self):
        return self.to_string(self.transaction_currency, self.counter_currency, self.start_time, self.end_time)

    @staticmethod
    def to_string(transaction_currency, counter_currency, start_time, end_time):
        return f"{transaction_currency}-{counter_currency}-{int(start_time)}-{int(end_time)}"

    @time_performance
    def _build_buy_and_hold_benchmark(self):
        self._buy_and_hold_benchmark = TickDrivenBacktester.build_benchmark(
            transaction_currency=self.transaction_currency,
            counter_currency=self.counter_currency,
            start_cash=self.start_cash,
            start_crypto=self.start_crypto,
            start_time=self.start_time,
            end_time=self.end_time,
            source=self.source,
            tick_provider=PriceDataframeTickProvider(self.price_data,
                                                     transaction_currency=self.transaction_currency,
                                                     counter_currency=self.counter_currency,
                                                     source=self.source,
                                                     resample_period=self.resample_period,
                                                     ),
            database=self.database
        )


    @property
    def buy_and_hold_benchmark(self):
        if self._buy_and_hold_benchmark is None:
            self._build_buy_and_hold_benchmark()   # lazily evaluate only when first invoked
        return self._buy_and_hold_benchmark

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

