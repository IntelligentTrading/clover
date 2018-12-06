from enum import Enum
from apps.backtesting.utils import datetime_from_timestamp


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

    def __eq__(self, other):
        return self.value == other.value


class Order:
    def __init__(self, order_type, transaction_currency, counter_currency, timestamp, value, unit_price,
                 transaction_cost_percent, time_delay=0, slippage=0, original_price=None):
        self.order_type = order_type
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.timestamp = timestamp
        self.value = value
        self.unit_price = unit_price
        self.transaction_cost_percent = transaction_cost_percent
        self.time_delay = time_delay
        self.original_price = original_price
        self.slippage = slippage

    def execute(self):
        if self.order_type == OrderType.BUY:
            return (self.value * (1 - self.transaction_cost_percent)) / (self.unit_price * (1 + self.slippage)), -self.value
        elif self.order_type == OrderType.SELL:
            return -self.value, (self.value * self.unit_price * (1 - self.slippage)) * (1 - self.transaction_cost_percent)

    def __str__(self):
        delta_currency, delta_cash = self.execute()
        return "{0}  \t {1: <16} \t delta_cash -> {2:13.2f} {3} \t " \
               "delta_currency -> {4:13.6f} {5} \t (1 {6} = {7:.8f} {8} {9}), slippage = {10:.2f}". format(
                datetime_from_timestamp(self.timestamp),
                self.order_type,
                delta_cash,
                self.counter_currency,
                delta_currency,
                self.transaction_currency,
                self.transaction_currency,
                self.unit_price,
                self.counter_currency,
                " -> delayed trading with delay = {} seconds, original price = {}".
                    format(self.time_delay, self.original_price) if self.time_delay != 0 else "",
                self.slippage
                )