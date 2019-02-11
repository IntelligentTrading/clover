import logging
import empyrical
import numpy as np
import pandas as pd
import copy

from apps.backtesting.orders import OrderType
from apps.backtesting.utils import get_distinct_signal_types, datetime_from_timestamp
from apps.backtesting.config import transaction_cost_percents
from abc import ABC, abstractmethod
from apps.backtesting.charting import BacktestingChart

from apps.backtesting.order_generator import OrderGenerator
from apps.backtesting.config import INF_CRYPTO, INF_CASH
from apps.backtesting.data_sources import DB_INTERFACE, NoPriceDataException

logging.getLogger().setLevel(logging.INFO)
pd.options.mode.chained_assignment = None


class Evaluation(ABC):

    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source="binance",
                 resample_period=60, evaluate_profit_on_last_order=True, verbose=True,
                 benchmark_backtest=None, time_delay=0, slippage=0, order_generator=OrderGenerator.ALTERNATING,
                 database=DB_INTERFACE):

        self._strategy = strategy
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._start_cash = start_cash
        self._start_crypto = start_crypto
        self._start_time = start_time
        self._end_time = end_time
        self._source = source
        self._resample_period = resample_period
        self._evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self._transaction_cost_percent = transaction_cost_percents[source]
        self._verbose = verbose
        self._benchmark_backtest = benchmark_backtest
        self._time_delay = time_delay
        self._slippage = slippage
        self._order_generator_type = order_generator
        self.database = database

        if order_generator == OrderGenerator.POSITION_BASED and not \
            (start_cash == INF_CASH and start_crypto == INF_CRYPTO):
            logging.warning('Position-based generator selected, and cash not set to infinite. '
                            'Be careful of currency scale errors!')

        if self._evaluate_profit_on_last_order is True:
            logging.warning('Evaluating profit on last order. '
                            'Strategies generating only one buy and no sells will have zero profit.')
        self._order_generator = OrderGenerator.create(
            generator_type=order_generator,
            start_cash=start_cash,
            start_crypto=start_crypto,
            time_delay=time_delay,
            slippage=slippage,
            database=self.database
        )

        if benchmark_backtest is not None:
            pass # TODO: fix assertions and delayed buy&hold
            # assert benchmark_backtest._start_time == self._start_time
            # assert benchmark_backtest._end_time == self._end_time

        # Init backtesting variables
        self._init_backtesting(start_cash, start_crypto)
        #self.trading_df = pd.DataFrame(columns=['close_price', 'signal', 'order', 'cash', 'crypto', 'total_value'])


    def _reevaluate_inf_bank(self):
        start_cash = self._start_cash
        start_crypto = self._start_crypto
        if self._start_cash == INF_CASH:
            # simulate how the trading would have gone
            evaluation = copy.deepcopy(self)
            evaluation._start_cash = evaluation._cash = 0
            evaluation._verbose = False
            evaluation.run()
            if evaluation._end_cash < 0:
                start_cash = -evaluation.end_cash
                #self._start_cash = self._cash = -evaluation.end_cash
        if self._start_crypto == INF_CRYPTO:
            evaluation = copy.deepcopy(self)
            evaluation._start_crypto = evaluation._crypto = 0
            evaluation._verbose = False
            evaluation.run()
            if evaluation._end_crypto < 0:
                start_crypto = -evaluation.end_crypto
                #self._start_crypto = self._crypto = -evaluation.end_crypto
        self._init_backtesting(start_cash, start_crypto)

    def _init_backtesting(self, start_cash, start_crypto):
        self._cash = start_cash
        self._crypto = start_crypto
        self._num_trades = 0
        self._num_buys = 0
        self._num_sells = 0
        self._end_price = None
        self.orders = []
        self.order_signals = []
        self.trading_df_rows = []  # optimization for speed
        self._init_start_value()
        self._init_start_value_USDT()


    @property
    def noncumulative_returns(self):
        return self.trading_df['return_relative_to_past_tick']

    @abstractmethod
    def run(self):
        pass

    @property
    def num_buys(self):
        return self._num_buys

    @property
    def num_sells(self):
        return self._num_sells

    @property
    def num_orders(self):
        return len(self.orders)

    @property
    def num_trades(self):
        return self._num_trades

    @property
    def start_value_usdt(self):
        return self._start_value_USDT

    def _init_start_value_USDT(self):
        try:
            self._start_value_USDT = self.database.convert_value_to_USDT(self._start_cash, self._start_time,
                                                                       self._counter_currency, self._source)
            if self._start_crypto > 0 and self._transaction_currency is not None:
                self._start_value_USDT += self.database.convert_value_to_USDT(self._start_crypto, self._start_time,
                                                                            self._transaction_currency, self._source)
        except NoPriceDataException:
            self._start_value_USDT = None

    @property
    def end_value_usdt(self):
        return self._end_value_USDT

    @property
    def profit_usdt(self):
        if self.start_value_usdt is None or self.end_value_usdt is None:
            return None
        else:
            return self.end_value_usdt - self.start_value_usdt

    @property
    def profit_percent_usdt(self):
        if self.profit_usdt is None or self.start_value_usdt is None:
            return None
        else:
            return self.profit_usdt / self.start_value_usdt * 100

    @property
    def start_value(self):
        return self._start_value

    def _init_start_value(self):
        try:
            self._start_value = self._start_cash + \
                   (self._start_crypto * self.database.get_price(
                       self._transaction_currency,
                       self._start_time,
                       self._source,
                       self._counter_currency) if self._start_crypto > 0 else 0)
            # because more often than not we start with 0 crypto and at the "beginning of time"
        except NoPriceDataException:
            self._start_value = None

    @property
    def end_cash(self):
        return self._end_cash

    @property
    def end_crypto(self):
        return self._end_crypto

    @property
    def end_value(self):
        try:
            return self.end_cash + (self.end_price * self.end_crypto) * (1-transaction_cost_percents[self._source])
        except:
            return None

    @property
    def end_price(self):
        return self._end_price

    @property
    def profit(self):
        if self.end_value is None or self.start_value is None:
            return None
        else:
            return self.end_value - self.start_value

    @property
    def profit_percent(self):
        if self.profit is None or self.start_value is None:
            return None
        else:
            return self.profit/self.start_value*100

    @property
    def max_drawdown(self):
        return self._max_drawdown

    @property
    def max_drawdown_duration(self):
        return self._max_drawdown_duration

    def _compute_max_drawdown(self):
        returns = self.noncumulative_returns
        if len(returns) == 0:
            return np.nan, np.nan, np.nan
        try:
            r = returns.add(1).cumprod()
            dd = r.div(r.cummax()).sub(1)
            mdd = dd.min()
            end = dd.idxmin()
            start = r.loc[:end].idxmax()
            return mdd, start, end
        except:
            return np.nan, np.nan, np.nan

    @property
    def sharpe_ratio(self):
        return self._sharpe_ratio

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def min_buy_sell_pair_return(self):
        return self._buy_sell_pair_returns.min()

    @property
    def max_buy_sell_pair_return(self):
        return self._buy_sell_pair_returns.max()

    @property
    def mean_buy_sell_pair_return(self):
        return self._buy_sell_pair_returns.mean()

    @property
    def std_buy_sell_pair_return(self):
        return self._buy_sell_pair_returns.std()

    @property
    def min_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.min()

    @property
    def max_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.max()

    @property
    def mean_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.mean()

    @property
    def std_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.std()

    @property
    def min_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.min()

    @property
    def max_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.max()

    @property
    def mean_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.mean()

    @property
    def std_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.std()

    @property
    def num_buy_sell_pairs(self):
        return self._num_sells

    @property
    def num_profitable_trades(self):
        return self._num_gains

    @property
    def num_unprofitable_trades(self):
        return self._num_losses

    @property
    def percent_profitable_trades(self):
        return self.num_profitable_trades / self.num_buy_sell_pairs if self.num_buy_sell_pairs != 0 else np.nan

    @property
    def percent_unprofitable_trades(self):
        self.num_unprofitable_trades / self.num_buy_sell_pairs if self.num_buy_sell_pairs != 0 else np.nan

    @property
    def benchmark_backtest(self):
        return self._benchmark_backtest



    def _write_trading_df_row(self):
        total_value = self._crypto * self._current_price + self._cash

        """
        # fill a row in the trading dataframe
        self.trading_df.loc[self._current_timestamp] = pd.Series({
            'close_price': self._current_price,
            'cash': self._cash,
            'crypto': self._crypto,
            'total_value': total_value,
            'order': "" if self._current_order is None else self._current_order.order_type.value,
            'signal': "" if self._current_order is None or self._current_signal is None else self._current_signal.signal_type})
        """
        self.trading_df_rows.append({
            'timestamp': self._current_timestamp,
            'close_price': self._current_price,
            'cash': self._cash,
            'crypto': self._crypto,
            'total_value': total_value,
            'order': "" if self._current_order is None else self._current_order.order_type.value,
            'signal': "" if self._current_order is None or self._current_signal is None else self._current_signal.signal_type,
            'order_obj': self._current_order,
            'signal_obj': self._current_signal
        }
        )

    def _finalize_backtesting(self):

        # self.trading_df = pd.DataFrame(columns=['close_price', 'signal', 'order', 'cash', 'crypto', 'total_value'],
        # columns=['close_price', 'signal', 'order', 'cash', 'crypto', 'total_value'])
        self.trading_df = pd.DataFrame(self.trading_df_rows,
                                       columns=['timestamp', 'close_price', 'signal', 'order', 'cash', 'crypto', 'total_value',
                                                'order_obj', 'signal_obj'])
        self.trading_df = self.trading_df.set_index('timestamp')
        assert self.trading_df.index.is_monotonic_increasing
        # set finishing variable values
        self._end_cash = self._cash
        self._end_crypto = self._crypto

        # compute returns for stats
        self.trading_df = self._fill_returns(self.trading_df)
        returns = np.array(self.trading_df['return_relative_to_past_tick'])
        # self._max_drawdown = empyrical.max_drawdown(np.array(returns))
        self._max_drawdown, start_dd, end_dd = self._compute_max_drawdown()
        self._max_drawdown_duration = end_dd - start_dd
        self._sharpe_ratio = empyrical.sharpe_ratio(returns)

        # extract only rows that have orders
        self.orders_df = self.trading_df[self.trading_df['order'] != ""]
        # recalculate returns
        self.orders_df = self._fill_returns(self.orders_df)
        # get profits on sell
        orders_sell_df = self.orders_df[self.orders_df['order'] == "SELL"]
        self._buy_sell_pair_returns = np.array(orders_sell_df['return_relative_to_past_tick'])
        self._buy_sell_pair_gains = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns > 0)]
        self._buy_sell_pair_losses = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns < 0)]

        self._num_gains = len(self._buy_sell_pair_gains)
        self._num_losses = len(self._buy_sell_pair_losses)

        # if no returns, no gains or no losses, stat functions will return nan
        if len(self._buy_sell_pair_returns) == 0:
            self._buy_sell_pair_returns = np.array([np.nan])

        if len(self._buy_sell_pair_gains) == 0:
            self._buy_sell_pair_gains = np.array([np.nan])

        if len(self._buy_sell_pair_losses) == 0:
            self._buy_sell_pair_losses = np.array([np.nan])

        if self._benchmark_backtest is not None:
            if len(self.benchmark_backtest.noncumulative_returns) != len(self.noncumulative_returns):
                logging.debug('Incompatible noncumulative returns fields of backtester and benchmark! '
                              'Alpha and beta not calculated.')
                self._alpha = None
                self._beta = None
            else:
                self._alpha, self._beta = \
                    empyrical.alpha_beta(self.noncumulative_returns, self._benchmark_backtest.noncumulative_returns)
        else:
            self._alpha = self._beta = np.nan

        # fill end price
        self._fill_end_price()

        # fill end USDT value
        self._fill_end_usdt_value()

        if self._verbose:
            logging.info(self.get_report())
            # logging.info(self.trading_df)
            # self.plot_portfolio()


    def _fill_end_usdt_value(self):
        try:
            self._end_value_USDT = self.database.convert_value_to_USDT(self.end_cash, self._end_time, self._counter_currency,
                                                                     self._source) + \
                                   self.database.convert_value_to_USDT(self.end_crypto, self._end_time, self._end_crypto_currency,
                                                                     self._source)
        except NoPriceDataException:
            self._end_value_USDT = None

    def _fill_end_price(self):
        if not self.orders_df.empty and (
                self._evaluate_profit_on_last_order or self.orders_df.tail(1)['order'].item() == "SELL"):
            self._end_price = self.orders_df.tail(1)['close_price'].item()
        else:
            try:
                self._end_price = self.database.get_price(self._transaction_currency, self._end_time, self._source,
                                                        self._counter_currency)
            except:
                self._end_price = None

    def _fill_returns(self, df):
        df['return_from_initial_investment'] = (df['total_value'] - self.start_value) / self.start_value
        df['return_relative_to_past_tick'] = df['total_value'].diff() / df['total_value'].shift(1)
        return df


    def get_orders(self):
        return self.orders

    def _format_price_dependent_value(self, value):
        if value is None:
            return float('nan')
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self._strategy))

        # output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self._start_cash, self._counter_currency,
                                                                           self._start_crypto, self._transaction_currency
                                                                           if self._start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self._start_time)))
        output.append("--")

        '''
        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals and len(self.order_signals) == len(self.orders): # for buy & hold we don't have signals
                output.append("   signal: {}".format(self.order_signals[i]))
        '''
        for i, row in self.orders_df.iterrows():
            order = row.order_obj
            signal = row.signal_obj
            output.append(str(order))
            if include_order_signals and signal is not None:  # for buy & hold we don't have signals
                output.append("   signal: {}".format(signal))
            output.append(f'   cash: {row.cash}    crypto: {row.crypto}')


        output.append("End time: {}".format(datetime_from_timestamp(self._end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self._num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self._counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self._transaction_currency))
        output.append("End price: {}".format(self.end_price))

        sign = "+" if self.profit != None and self.profit >= 0 else ""
        output.append("Total value invested: {} {}".format(self._format_price_dependent_value(self.start_value),
                                                           self._counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self._format_price_dependent_value(self.end_value),
                                                                             self._counter_currency,
                                                                             sign,
                                                                             self._format_price_dependent_value(self.profit_percent)))
        output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit), self._counter_currency))

        if self._counter_currency != "USDT":
            sign = "+" if self.profit_usdt is not None and self.profit_usdt >= 0 else ""
            output.append("Total value invested: {:.2f} {} (conversion on {})".format(
                self._format_price_dependent_value(self.start_value_usdt),
                "USDT",
                datetime_from_timestamp(self._start_time)))
            output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self._format_price_dependent_value(self.end_value_usdt), "USDT", sign,
                        self._format_price_dependent_value(self.profit_percent_usdt),
                        datetime_from_timestamp(self._end_time)))
            output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit_usdt),
                                                       "USDT"))

        output.append("\nAdditional stats:")
        output.append("  Max drawdown: {}".format(self.max_drawdown))
        output.append("  Max drawdown duration: {}".format(self.max_drawdown_duration))
        output.append("  Sharpe ratio: {}".format(self.sharpe_ratio))
        output.append("  Alpha: {}".format(self.alpha))
        output.append("  Beta: {}".format(self.beta))

        output.append("  Buy-sell pair gains - overall stats")
        output.append("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_gain,
            self.max_buy_sell_pair_gain,
            self.mean_buy_sell_pair_gain,
            self.std_buy_sell_pair_gain
        ))

        output.append("  Buy-sell pair losses - overall stats")
        output.append("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_loss,
            self.max_buy_sell_pair_loss,
            self.mean_buy_sell_pair_loss,
            self.std_buy_sell_pair_loss
        ))

        output.append("  Buy-sell pair returns - overall stats")
        output.append("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_return,
            self.max_buy_sell_pair_return,
            self.mean_buy_sell_pair_return,
            self.std_buy_sell_pair_return
        ))


        output.append("  Total buy-sell pairs: {}".format(self.num_buy_sell_pairs))
        output.append("  Total profitable trades: {}".format(self.num_profitable_trades))
        output.append("  Percent profitable trades: {}".format(self.percent_profitable_trades))
        output.append("  Percent unprofitable trades: {}".format(self.percent_unprofitable_trades))
        
        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self._strategy.get_short_summary(),
            self._start_cash, self._counter_currency, self._start_crypto, self.start_crypto_currency,
            self.end_cash, self._counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.profit_percent is not None and self.profit_percent >= 0 else "",
            self._format_price_dependent_value(self.profit_percent)))

    def execute_order(self, order):
        assert order.transaction_currency == self._transaction_currency
        delta_crypto, delta_cash = order.execute()
        self._cash += delta_cash
        self._crypto += delta_crypto
        self._num_trades += 1
        if order.order_type == OrderType.BUY:
            self._num_buys += 1
        elif order.order_type == OrderType.SELL:
            # the currency we're selling must match the bought currency
            self._num_sells += 1




    def to_primitive_types_dictionary(self):
        import inspect
        result = {}
        members = inspect.getmembers(self)
        for (name, value) in members:
            if name.startswith("__"):
                continue
            if type(value) not in (int, str, bool, float, np.float64):
                continue
            if name.startswith("_"):
                name = name[1:]
            result[name] = value
        if self._benchmark_backtest is not None:
            result['benchmark_profit_percent'] = self._benchmark_backtest.profit_percent
            result['benchmark_profit_percent_usdt'] = self._benchmark_backtest.profit_percent_usdt

        return result


    def _print_dict(self):
        dictionary = vars(self).copy()
        del dictionary["orders"]
        if "signals" in dictionary:
            del dictionary["signals"]
        for k in dictionary:
            if k.startswith("_"):
                print(f"'{k[1:]}':self.{k},")
            else:
                print(f"'{k}':self.{k},")


    def to_dictionary(self):
        dictionary = {
            'strategy': self._strategy,
            'transaction_currency': self._transaction_currency,
            'counter_currency': self._counter_currency,
            'start_cash': self._start_cash,
            'start_crypto': self._start_crypto,
            'start_time': self._start_time,
            'end_time': self._end_time,
            'source': self._source,
            'resample_period': self._resample_period,
            'evaluate_profit_on_last_order': self._evaluate_profit_on_last_order,
            'transaction_cost_percent': self._transaction_cost_percent,
            'benchmark_backtest': self._benchmark_backtest,
            'time_delay': self._time_delay,
            'slippage': self._slippage,
            'order_generator_type': self._order_generator_type,
            'cash': self._cash,
            'crypto': self._crypto,
            'num_trades': self._num_trades,
            'num_buys': self._num_buys,
            'num_sells': self._num_sells,
            'order_signals': self.order_signals,
            'trading_df': self.trading_df,
            'end_cash': self._end_cash,
            'end_crypto': self._end_crypto,
            'max_drawdown': self._max_drawdown,
            'max_drawdown_duration': self._max_drawdown_duration,
            'sharpe_ratio': self._sharpe_ratio,
            'orders_df': self.orders_df,
            'buy_sell_pair_returns': self._buy_sell_pair_returns,
            'buy_sell_pair_gains': self._buy_sell_pair_gains,
            'buy_sell_pair_losses': self._buy_sell_pair_losses,
            'num_gains': self._num_gains,
            'num_losses': self._num_losses,
            'alpha': self._alpha,
            'beta': self._beta,
        }

        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])
        dictionary["mean_buy_sell_pair_return"] = self.mean_buy_sell_pair_return

        if self.end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.profit
                dictionary["profit_percent"] = self.profit_percent
                dictionary["profit_USDT"] = self.profit_usdt
                dictionary["profit_percent_USDT"] = self.profit_percent_usdt
            except NoPriceDataException:
                logging.error("No price data!")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary




    def to_dictionary_slow(self):
        dictionary = vars(self).copy()
        # remove trailing underscores
        tmp = {(k[1:] if k.startswith("_") else k): dictionary[k] for k in dictionary.keys()}
        dictionary = tmp
        del dictionary["orders"]
        if "signals" in dictionary:
            del dictionary["signals"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])
        dictionary["mean_buy_sell_pair_return"] = self.mean_buy_sell_pair_return

        dictionary["transaction_currency"] = self._end_crypto_currency
        if "horizon" not in vars(self._strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self._strategy.horizon.name

        if self.end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.profit
                dictionary["profit_percent"] = self.profit_percent
                dictionary["profit_USDT"] = self.profit_usdt
                dictionary["profit_percent_USDT"] = self.profit_percent_usdt
            except NoPriceDataException:
                logging.error("No price data!")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary

    def to_excel(self, output_file):
        writer = pd.ExcelWriter(output_file, datetime_format='mmm d yyyy hh:mm:ss',
                        date_format='mmmm dd yyyy')
        trading_df = self.trading_df.copy()
        trading_df.index = pd.to_datetime(self.trading_df.index, unit='s', utc=True)
        trading_df.to_excel(writer, 'Results')
        writer.save()

    def plot_cumulative_returns(self):
        if self.trading_df.empty:
            return
        chart = BacktestingChart(self, self._benchmark_backtest)
        chart.draw_price_and_cumulative_returns()

    def plot_returns_tear_sheet(self):
        if self.trading_df.empty:
            return
        chart = BacktestingChart(self, self._benchmark_backtest)
        chart.draw_returns_tear_sheet()

    def plot_price_and_orders(self):
        if self.trading_df.empty:
            return
        chart = BacktestingChart(self, self._benchmark_backtest)
        chart.draw_price_chart()


    @staticmethod
    def signature_key(**kwargs):
        return (
            kwargs['strategy'].get_short_summary(),
            kwargs['transaction_currency'],
            kwargs['counter_currency'],
            kwargs['start_cash'],
            kwargs['start_crypto'],
            kwargs['start_time'],
            kwargs['end_time'],
            kwargs['source'],
            kwargs['resample_period'],
            kwargs['evaluate_profit_on_last_order'],
            kwargs['benchmark_backtest'].signature_key if kwargs['benchmark_backtest'] is not None else None,
            kwargs['time_delay'],
            kwargs['slippage'],
            kwargs['order_generator']
        )



