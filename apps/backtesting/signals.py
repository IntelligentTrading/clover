from enum import Enum

from apps.backtesting.utils import datetime_from_timestamp
from collections import namedtuple

class Signal:
    def __init__(self, signal_type, trend, horizon, strength_value, strength_max,
                 price, price_change, timestamp, rsi_value, transaction_currency, counter_currency, source, resample_period):
        self.signal_type = signal_type
        self.trend = trend
        self.horizon = horizon
        self.strength_value = strength_value
        self.strength_max = strength_max
        self.price = price
        self.price_change = price_change
        self.timestamp = timestamp
        self.rsi_value = rsi_value
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.source = source
        self.resample_period = resample_period
        if strength_value == None:
            strength_value = 3
        self.signal_signature = get_signal_type(SignalType(signal=signal_type, trend=int(float(trend)), strength=int(strength_value)))

    def __str__(self):
        return ("{} {}-{} strength={} trend={} horizon={} timestamp={} rsi_value={}".format(self.signal_signature, self.transaction_currency, self.counter_currency, self.strength_value,
                                                             self.trend, self.horizon, datetime_from_timestamp(self.timestamp), self.rsi_value))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @staticmethod
    def from_pandas_row(row):
        return Signal(
            row['signal'],
            row['trend'],
            row['horizon'],
            row['strength_value'],
            row['strength_max'],
            row['price'],
            row['price_change'],
            row['timestamp'],
            row['rsi_value'],
            row['transaction_currency'],
            row['counter_currency'],
            row['source'],
            row['resample_period']
        )

    @staticmethod
    def pandas_to_objects_list(signals_df):
        signals = []
        for i, row in signals_df.iterrows():
            signals.append(Signal.from_pandas_row(row))
        return signals


def get_signal_type(signal_record):
    id = [x for x in ALL_SIGNALS if ALL_SIGNALS[x] == signal_record]
    assert len(id) == 1
    return id[0]

def pretty_print_signal(signal_type):
    s = ALL_SIGNALS[signal_type]
    decision = 'buy' if s.trend == 1 else 'sell'
    if s.signal == 'RSI':
        return f'RSI{s.strength} {decision}'
    elif s.signal == 'ANN_Simple':
        return f'ANN {decision}'
    elif s.signal == 'kumo_breakout':
        return f'Ichimoku {decision}'
    elif s.signal == 'VBI':
        return f'VBI {decision}'
    elif s.signal == 'RSI_Cumulative':
        return f'RSIComb{s.strength} {decision}'
    else:
        # unknown or don't care
        return signal_type


SignalType = namedtuple('SignalType', 'signal, trend, strength')
ALL_SIGNALS = {
    'rsi_buy_1' : SignalType('RSI', 1, 1),
    'rsi_buy_2' : SignalType('RSI', 1, 2),
    'rsi_buy_3' : SignalType('RSI', 1, 3),
    'rsi_sell_1': SignalType(signal='RSI', trend=-1, strength=1),
    'rsi_sell_2': SignalType(signal='RSI', trend=-1, strength=2),
    'rsi_sell_3': SignalType(signal='RSI', trend=-1, strength=3),

    'rsi_cumulat_buy_2' : SignalType('RSI_Cumulative', 1, 2),
    'rsi_cumulat_buy_3' : SignalType('RSI_Cumulative', 1, 3),
    'rsi_cumulat_sell_2': SignalType('RSI_Cumulative', -1, 2),
    'rsi_cumulat_sell_3': SignalType('RSI_Cumulative', -1, 3),

    'ichi_kumo_up' : SignalType('kumo_breakout', 1, 3),
    'ichi_kumo_down' : SignalType('kumo_breakout', -1, 3),

    'sma_bull_1' : SignalType('SMA', 1, 1),  # price crosses sma50 up
    'sma_bear_1' : SignalType('SMA', -1, 1),
    'sma_bull_2' : SignalType('SMA', 1, 2),   # price crosses sma200 up
    'sma_bear_2' : SignalType('SMA', -1, 2),
    'sma_bull_3' : SignalType('SMA', 1, 3),    # sma50 crosses sma200 up
    'sma_bear_3' : SignalType('SMA', -1, 3),

    'ann_simple_bull': SignalType('ANN_Simple', 1, 3),  # price crosses sma200 up
    'ann_simple_bear': SignalType('ANN_Simple', -1, 3),

    'genetic_up': SignalType('Genetic', 1, 3),
    'genetic_down': SignalType('Genetic', -1, 3),

    'ema_bull_1' : SignalType('EMA', 1, 1),  # price crosses sma50 up
    'ema_bear_1' : SignalType('EMA', -1, 1),
    'ema_bull_2' : SignalType('EMA', 1, 2),   # price crosses sma200 up
    'ema_bear_2' : SignalType('EMA', -1, 2),
    'ema_bull_3' : SignalType('EMA', 1, 3),    # sma50 crosses sma200 up
    'ema_bear_3' : SignalType('EMA', -1, 3),

    'vbi_buy': SignalType('VBI', 1, 3),

    'generic_up': SignalType('Generic', 1, 3),
    'generic_down': SignalType('Generic', -1, 3),

    'ann_anomaly': SignalType('ANN_AnomalyPrc', 0, 3)

}