import random
from apps.backtesting.orders import *
from apps.backtesting.data_sources import *
from apps.backtesting.signals import *
from abc import ABC, abstractmethod
import logging
from functools import total_ordering
from apps.backtesting.backtester_signals import SignalDrivenBacktester

log = logging.getLogger("strategies")


class StrategyDecision:
    """
    A class that encapsulates decision produced by a strategy.
    """
    BUY = "BUY"
    SELL = "SELL"
    IGNORE = None

    def __init__(self, timestamp, transaction_currency=None, counter_currency=None, source=None, outcome=None, signal=None):
        assert outcome in (StrategyDecision.BUY, StrategyDecision.SELL, StrategyDecision.IGNORE)
        self.outcome = outcome
        self.timestamp = timestamp
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.source = source
        self.signal = signal

    def buy(self):
        return self.outcome == StrategyDecision.BUY

    def sell(self):
        return self.outcome == StrategyDecision.SELL

    def ignore(self):
        return self.outcome == StrategyDecision.IGNORE

@total_ordering
class Strategy(ABC):
    """
    Base class for trading strategies.
    """

    def indicates_buy(self, signal):
        return int(float(signal.trend)) == 1

    def indicates_sell(self, signal):
        return int(float(signal.trend)) == -1

    @abstractmethod
    def belongs_to_this_strategy(self, signal):
        pass

    @abstractmethod
    def get_short_summary(self):
        pass

    def get_decision(self, timestamp, price, signals):
        decision = None
        for signal in signals:
            if not self.belongs_to_this_strategy(signal):
                continue

            if self.indicates_buy(signal):
                if decision is not None and decision.outcome == StrategyDecision.SELL:  # sanity checks
                    logging.error(
                        'Multiple signals at the same time with opposing decisions! Highly unlikely, investigate!')
                decision = StrategyDecision(timestamp, signal.transaction_currency, signal.counter_currency,
                                            signal.source, StrategyDecision.BUY, signal)
            elif self.indicates_sell(signal):
                if decision is not None and decision.outcome == StrategyDecision.BUY:  # sanity checks
                    logging.error(
                        'Multiple signals at the same time with opposing decisions! Highly unlikely, investigate!')
                decision = StrategyDecision(timestamp, signal.transaction_currency, signal.counter_currency,
                                            signal.source, StrategyDecision.SELL, signal)

        if decision is None:
            decision = StrategyDecision(timestamp, outcome=StrategyDecision.IGNORE)
        return decision

    @staticmethod
    def get_buy_signals(strategy, signals):
        return [signal for signal in signals
                if strategy.belongs_to_this_strategy(signal) and strategy.indicates_buy(signal)]

    @staticmethod
    def get_sell_signals(strategy, signals):
        return [signal for signal in signals
                if strategy.belongs_to_this_strategy(signal) and strategy.indicates_sell(signal)]

    def clear_state(self):
        pass

    def __eq__(self, other):
        return self.get_short_summary() == other.get_short_summary()

    def __lt__(self, other):
        return self.get_short_summary() < other.get_short_summary()


class SignalStrategy(Strategy):
    """
    Base class for trading strategies that build orders based on a given list of signals.
    """

    def evaluate(self, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time,
                 source, resample_period, evaluate_profit_on_last_order=False, verbose=True, time_delay=0,
                 order_generator=None, database=postgres_db):
        """
        Builds a signal-based backtester using this strategy.
        See full documentation in SignalDrivenBacktester.
        :return: A SignalDrivenBacktester object.
        """
        return SignalDrivenBacktester(
            strategy=self,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            start_cash=start_cash,
            start_crypto=start_crypto,
            start_time=start_time,
            end_time=end_time,
            source=source,
            resample_period=resample_period,
            evaluate_profit_on_last_order=evaluate_profit_on_last_order,
            verbose=verbose,
            time_delay=time_delay,
            order_generator=order_generator,
            database=database
        )


@total_ordering
class SignalSignatureStrategy(SignalStrategy):
    """
    A strategy based on ITF signal signatures (see a list of signatures in the field ALL_SIGNALS).
    """
    def __init__(self, signal_set):
        """
        Builds the signal signature strategy.
        :param signal_set: A list of ITF signal signatures to be used in the strategy.
        """
        self.signal_set = signal_set

    def belongs_to_this_strategy(self, signal):
        return signal.signal_signature in self.signal_set

    def __str__(self):
        output = []
        output.append("Strategy: a simple signal set-based strategy")
        output.append("  description: trading according to signal set {}".format(str(self.signal_set)))
        return "\n".join(output)

    def get_short_summary(self):
        return ' + '.join([pretty_print_signal(s) for s in self.signal_set])

    def __lt__(self, other):
        signal_types_self = {ALL_SIGNALS[s].signal for s in self.signal_set}
        signal_types_other = {ALL_SIGNALS[s].signal for s in other.signal_set}

        if len(signal_types_self) < len(signal_types_other):
            return True
        elif len(signal_types_self) > len(signal_types_other):
            return False
        else:
            return list(sorted(signal_types_self))[0] < list(sorted(signal_types_other))[0]

    def __eq__(self, other):
        return self.signal_set == other.signal_set


class SimpleRSIStrategy(SignalStrategy):
    """
    A strategy based on RSI thresholds.
    """
    def __init__(self, overbought_threshold=80, oversold_threshold=25, signal_type="RSI"):
        """
        Builds the RSI strategy.
        :param overbought_threshold: The RSI overbought threshold.
        :param oversold_threshold: The RSI oversold threshold.
        :param signal_type: signal type: Set to "RSI" or "RSI_Cumulative".
        """
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.signal_type = signal_type

    def indicates_sell(self, signal):
        return signal.rsi_value >= self.overbought_threshold

    def indicates_buy(self, signal):
        return signal.rsi_value <= self.oversold_threshold

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.signal_type

    def __str__(self):
        output = []
        output.append("Strategy: a simple {}-based strategy".format(self.signal_type))
        output.append("  description: selling when rsi_value >= overbought_threshold, buying when rsi_value <= oversold threshold ")
        output.append("Strategy settings:")
        output.append("  overbought_threshold = {}".format(self.overbought_threshold))
        output.append("  oversold_threshold = {}".format(self.oversold_threshold))
        return "\n".join(output)

    def get_short_summary(self):
        return "{} based strategy, overbought = {}, oversold = {}".format(self.signal_type,
                                                                          self.overbought_threshold,
                                                                          self.oversold_threshold)


class SimpleTrendBasedStrategy(SignalStrategy):
    """
    A strategy that decides on buys and sells based on signal trend information.
    """
    def __init__(self, signal_type):
        """
        Builds the trend-based strategy.
        :param signal_type: The underlying signal type to use with the strategy (e.g. "SMA", "RSI", "ANN").
        """
        self.signal_type = signal_type

    def __str__(self):
        output = []
        output.append("Strategy: trend-based strategy ({})".format(self.signal_type))
        output.append(
            "  description: selling when trend = -1, buying when trend = 1 ")
        return "\n".join(output)

    def get_short_summary(self):
        return "Trend-based strategy, signal type: {}".format(self.signal_type)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.signal_type


class ANNAnomalyStrategy(SignalStrategy):

    def __init__(self, confirmation_signal='RSI', max_delta_periods=1):
        self._last_signal = None
        self.max_delta_periods = max_delta_periods
        self.confirmation_signal = confirmation_signal

    def indicates_buy(self, signal):
        decision = self._confirm_signal(signal) and int(self._last_signal.trend) == 1
        return decision

    def indicates_sell(self, signal):
        decision = self._confirm_signal(signal) and int(self._last_signal.trend) == -1
        return decision

    def _confirm_signal(self, signal):
        if signal.signal_type == self.confirmation_signal:
            self._last_signal = signal
        elif signal.signal_type == 'ANN_AnomalyPrc' \
                and self._last_signal is not None \
                and (signal.timestamp - self._last_signal.timestamp) <= signal.resample_period * 60 * self.max_delta_periods \
                and signal.resample_period == self._last_signal.resample_period:
            logging.info(
                f'Confirmed: {signal} {self._last_signal}, delta_time = {signal.timestamp - self._last_signal.timestamp}')
            assert signal.transaction_currency == self._last_signal.transaction_currency
            return True
        return False

    def clear_state(self):
        self._last_signal = None

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.confirmation_signal or signal.signal_type == 'ANN_AnomalyPrc'

    def get_short_summary(self):
        return f'ANN anomaly + {self.confirmation_signal} confirmation (past candles: {self.max_delta_periods})'



# TODO
class BuyOnFirstSignalAndHoldStrategy(SignalStrategy):
    """
    A wrapper class for SignalStrategies. Buys on first signal, then holds.
    """
    def __init__(self, strategy):
        """
        Builds the strategy.
        :param strategy: The wrapped SignalStrategy.
        """
        self.strategy = strategy

    def get_orders(self, start_cash, start_crypto, time_delay=0):
        orders, order_signals = self.strategy.get_orders(start_cash=start_cash,
                                                         start_crypto=start_crypto, time_delay=time_delay)
        filtered_orders = []
        filtered_signals = []
        for i, order in enumerate(orders):
            if order.order_type == OrderType.BUY:
                filtered_orders.append(order)
                filtered_signals.append(order_signals[i])
                break
        return filtered_orders, filtered_signals

    def get_short_summary(self):
        return "Buy first & hold: {}".format(self.strategy.get_short_summary())

    def get_signal_report(self):
        return self.strategy.get_signal_report()


class BuyAndHoldTimebasedStrategy(SignalStrategy):
    """
    Standard buy & hold strategy (signal-driven).
    """
    def __init__(self, start_time, end_time, transaction_currency, counter_currency, source):
        """
        Builds the buy and hold strategy.
        :param start_time: When to buy transaction_currency.
        :param end_time: When to sell transaction_currency.
        :param transaction_currency: Transaction currency.
        :param counter_currency: Counter currency.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._source = source
        self._bought = False
        self._sold = False

    def get_decision(self, timestamp, price, signals):
        if timestamp >= self._start_time and timestamp <= self._end_time and not self._bought:
            if abs(timestamp - self._start_time) > 120:
                log.warning("Buy and hold BUY: ticker more than 2 mins after start time ({:.2f} mins)!"
                                .format(abs(timestamp - self._start_time)/60))
            self._bought = True
            return StrategyDecision(outcome=StrategyDecision.BUY, timestamp=timestamp,
                                    transaction_currency=self._transaction_currency,
                                    counter_currency=self._counter_currency,
                                    source=self._source)
        elif timestamp >= self._end_time and not self._sold:
            if abs(timestamp - self._end_time) > 120:
                log.warning("Buy and hold SELL: ticker more than 2 mins after end time ({:.2f} mins)!"
                                .format(abs(timestamp - self._end_time) / 60))
            self._sold = True
            return StrategyDecision(outcome=StrategyDecision.SELL, timestamp=timestamp,
                                    transaction_currency=self._transaction_currency,
                                    counter_currency=self._counter_currency,
                                    source=self._source)
        else:
            return StrategyDecision(outcome=StrategyDecision.IGNORE, timestamp=timestamp,
                                    transaction_currency=self._transaction_currency,
                                    counter_currency=self._counter_currency,
                                    source=self._source
                                    )

    def get_short_summary(self):
        return str(self)

    def belongs_to_this_strategy(self, signal):
        return False

    def __str__(self):
        return f"Buy&hold, start_time={self._start_time}, end_time={self._end_time}, " \
               f"transaction_currency={self._transaction_currency}, " \
               f"counter_currency={self._counter_currency}, source={self._source}"


class TickerStrategy(Strategy):
    """
    A wrapper class that converts signal-based strategies to ticker-based.
    Ticker-based strategies are used by TickDrivenBacktester.
    """

    @abstractmethod
    def process_ticker(self, price_data, signals):
        """

        :param price_data: Pandas row with OHLC data and timestamp.
        :param signals: ITF signals co-ocurring with price tick.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """
        pass


class TickerWrapperStrategy(TickerStrategy):

    def __init__(self, strategy):
        """
        Builds a ticker-driven strategy out of a regular SignalStrategy.
        :param strategy: The wrapped SignalStrategy.
        """
        self._strategy = strategy

    def process_ticker(self, price_data, signals):
        for i, signal in enumerate(signals):
            if not self._strategy.belongs_to_this_strategy(signal):
                continue
            if self._strategy.indicates_sell(signal):
                return StrategyDecision.SELL, signal
            elif self._strategy.indicates_buy(signal):
                return StrategyDecision.BUY, signal
        return StrategyDecision.IGNORE, None

    def belongs_to_this_strategy(self, signal):
        return self._strategy.belongs_to_this_strategy(signal)

    def get_short_summary(self):
        return self._strategy.get_short_summary()


class RandomTradingStrategy(SimpleTrendBasedStrategy):

    def __init__(self, max_num_signals, start_time, end_time, transaction_currency, counter_currency, source=0):
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.max_num_signals = max_num_signals
        self.source = source
        self.signal_type = "Generic"
        self.signals = self.build_signals()

    def get_orders(self, signals, start_cash, start_crypto, source, time_delay=0, slippage=0):
        return SignalStrategy.get_orders(self, self.signals, start_cash, start_crypto, source, time_delay, slippage)

    def build_signals(self):
        num_signals = random.randint(1, self.max_num_signals)
        prices = postgres_db.get_prices_in_range(self.start_time, self.end_time, self.transaction_currency, self.counter_currency,
                                                 self.source)
        selected_prices = prices.sample(num_signals)
        selected_prices = selected_prices.sort_index()
        signals = []

        for timestamp, row in selected_prices.iterrows():
            price = row["price"]
            buy = random.random() < 0.5
            signal = Signal(self.signal_type, 1 if buy else -1, Horizon.any, 3, 3, # "RSI" is just a placeholder
                 price/1E8, 0, timestamp, None, self.transaction_currency, self.counter_currency, self.source, None)
            signals.append(signal)
        return signals

    def belongs_to_this_strategy(self, signal):
        return True

    def get_short_summary(self):
        return "Random trading strategy, max number of signals: {}".format(self.max_num_signals)

