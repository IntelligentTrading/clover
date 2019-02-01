from apps.backtesting.evaluation import Evaluation
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.tick_provider_itf_db import TickProviderITFDB
from apps.backtesting.config import INF_CASH, INF_CRYPTO
from apps.backtesting.strategies import BuyAndHoldTimebasedStrategy
from apps.backtesting.order_generator import OrderGenerator
from apps.backtesting.caching import memoize
from apps.backtesting.data_sources import DB_INTERFACE


@memoize
class TickDrivenBacktester(Evaluation, TickListener):

    def __init__(self, tick_provider, **kwargs):
        super().__init__(**kwargs)
        self.tick_provider = tick_provider
        # reevaluate infinite bank
        self._reevaluate_inf_bank()

        self.run()

    def run(self):
        # register at tick provider
        self.tick_provider.add_listener(self)
        # ingest ticks
        self.tick_provider.run()

        # the provider will call the broadcast_ended() method when no ticks remain

    def process_event(self, ticker_data):
        if ticker_data.timestamp < self._start_time or ticker_data.timestamp > self._end_time:
            return

        self._current_timestamp = ticker_data.timestamp
        self._current_price = ticker_data.close_price

        decision = self._strategy.get_decision(ticker_data)
        order = self._order_generator.generate_order(decision)
        if order is not None:
            self.orders.append(order)
            self.order_signals.append(decision.signal)
            self.execute_order(order)

        self._current_order = order
        self._current_signal = decision.signal

        self._write_trading_df_row()

    def broadcast_ended(self):
        self._end_crypto_currency = self._transaction_currency
        self._finalize_backtesting()

    @property
    def end_price(self):
        if not self.trading_df.empty:
            return self.trading_df.tail(1)['close_price'].item()
        else:
            return Evaluation.end_price.fget(self)

    @staticmethod
    def build_benchmark(transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time,
                        source, tick_provider=None, time_delay=0, slippage=0, database=DB_INTERFACE):
        if tick_provider is None:
            tick_provider = TickProviderITFDB(transaction_currency,
                                                    counter_currency,
                                                    start_time,
                                                    end_time, database=database)
        benchmark_strategy = BuyAndHoldTimebasedStrategy(start_time, end_time, transaction_currency, counter_currency,
                                                         source=source)
        benchmark_order_generator = OrderGenerator.ALTERNATING

        benchmark = TickDrivenBacktester(
                tick_provider=tick_provider,
                strategy=benchmark_strategy,
                transaction_currency=transaction_currency,
                counter_currency=counter_currency,
                start_cash=start_cash,
                start_crypto=start_crypto,
                start_time=start_time,
                end_time=end_time,
                source=source,
                resample_period=None,
                verbose=False,
                time_delay=time_delay,
                slippage=slippage,
                order_generator=benchmark_order_generator,
                database=database
            )
        return benchmark

