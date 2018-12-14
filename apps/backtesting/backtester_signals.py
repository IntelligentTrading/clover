from apps.backtesting.evaluation import Evaluation
from apps.backtesting.orders import OrderType

class SignalDrivenBacktester(Evaluation):

    def __init__(self, signals=None, **kwargs):
        """
        :param strategy: Backtested strategy (instance of SignalStrategy).        
        :param transaction_currency: Transaction currency for which we're backtesting.
        :param counter_currency: Counter currency for which we're backtesting.
        :param start_cash: Starting amount of counter_currency.
        :param start_crypto: Starting amount of transaction_currency.
        :param start_time: Beginning of timeframe for which signals are fetched (UTC timestamp).
        :param end_time: End of timeframe for which signals are fetched (UTC timestamp).
        :param source: ITF exchange code.
        :param resample_period: Duration of 1 candle (minutes).
        :param evaluate_profit_on_last_order: Evaluate gains at the time of the last order (vs. at end_time).
        :param verbose: Produce verbose output.
        :param time_delay: Parameter specifying the delay applied when fetching price info (in seconds).
        :param slippage: Parameter specifying the slippage percentage, applied in the direction of the trade.
        :param signals: A predefined list of signals passed into the strategy. If not supplied, the signals will be
                        pulled from the database.
        :param database: database connection (default: postgres_db)
        """
        super().__init__(**kwargs)
        if signals is None:
            self.signals = self.database.get_filtered_signals(
                start_time=self._start_time,
                end_time=self._end_time,
                counter_currency=self._counter_currency,
                transaction_currency=self._transaction_currency,
                source=self._source,
                resample_period=self._resample_period
            )
        else:
            self.signals = signals

        self._buy_currency = self._start_crypto_currency = self._transaction_currency
        self._reevaluate_inf_bank()

        self.run()



    def fill_trading_df(self, orders):
        for i, order in enumerate(orders):
            if i == 0: # first order
#                assert order.order_type == OrderType.BUY
                self._start_crypto_currency = self._buy_currency = order.transaction_currency

            if order.order_type == OrderType.BUY:
                self._buy_currency = order.transaction_currency
            elif order.order_type == OrderType.SELL:
                assert order.transaction_currency == self._buy_currency
            self.execute_order(order)
            self._current_timestamp = order.timestamp
            self._current_price = order.unit_price
            self._current_order = order
            self._current_signal = self.order_signals[i] if len(self.order_signals) > 0 else None
            self._write_trading_df_row()
        self._end_crypto_currency = self._buy_currency
        self._finalize_backtesting()


    def get_decisions(self):
        decisions = []
        # special case for buy and hold - get decision at the beginning of time
        if len(self.signals) == 0 or self.signals[0].timestamp != self._start_time:
            decisions.append(self._strategy.get_decision(self._start_time, None, []))

        for signal in self.signals:
            timestamp = signal.timestamp
            price = signal.price
            decisions.append(self._strategy.get_decision(timestamp, price, [signal]))

        # again, let the strategy decide at the end of time
        if len(self.signals) == 0 or self.signals[-1].timestamp != self._end_time:
            decisions.append(self._strategy.get_decision(self._end_time, None, []))

        return decisions

    def run(self):
        self.orders, self.order_signals = self._order_generator.get_orders(
            decisions=self.get_decisions()
        )

        self.fill_trading_df(self.orders)