import logging
from apps.backtesting.data_sources import DB_INTERFACE
from apps.backtesting.legacy_postgres import PostgresDatabaseConnection
from collections import namedtuple, OrderedDict

Allocation = namedtuple('Allocation', 'amount coin portion unit_price value')

POSTGRES = PostgresDatabaseConnection()

class PortfolioSnapshot:

    def __init__(self, timestamp, allocations_dict, db_interface=POSTGRES):
        self._timestamp = timestamp
        self._allocations = {}
        self.db_interface = db_interface
        self._fill_internals(allocations_dict)

    def _fill_internals(self, allocations_dict):
        self._held_coins = set()
        self._portion_sum = 0
        self._total_value = 0
        for item in allocations_dict:
            # figure out the value of this particular item
            unit_price = self._get_price(item['coin'])
            value = unit_price * item['amount']
            allocation = Allocation(**item, unit_price=unit_price, value=value)
            self._held_coins.add(allocation.coin)
            self._portion_sum += float(allocation.portion)
            self._total_value += allocation.value
            self._allocations[allocation.coin] = allocation

    def _get_price(self, coin):
        if coin == 'BTC':
            return 1
        elif coin == 'USDT':
            # get BTC_USDT price and invert
            btc_usdt_price = self.db_interface.get_price_nearest_to_timestamp(currency='BTC',
                                                                    timestamp=self._timestamp,
                                                                    source=2,  # Binance
                                                                    counter_currency='USDT')
            return 1.0 / btc_usdt_price

        return self.db_interface.get_price_nearest_to_timestamp(currency=coin,
                                                                timestamp=self._timestamp,
                                                                source=2,               # Binance
                                                                counter_currency='BTC')

    def get_allocation(self, coin):
        return self._allocations.get(coin, None)

    def report(self):
        logging.info(f'At timestamp {self._timestamp}, portfolio stats are:')
        logging.info(f'    -> total value: {self._total_value}')
        for coin, allocation in self._allocations.items():
            logging.info(f'       {allocation.amount} {allocation.coin} worth {allocation.value} BTC, {allocation.portion*100:2}% total')



class PortfolioBacktester:

    def __init__(self):
        self._portfolio_snapshots = OrderedDict()

    def process_allocations(self, timestamp, allocations_dict):
        self._portfolio_snapshots[timestamp] = PortfolioSnapshot(timestamp, allocations_dict)

    def value_report(self):
        for timestamp, snapshot in self._portfolio_snapshots.items():
            snapshot.report()



class DummyDataProvider:
    sample_allocations = """
    [{
        "amount": 0.00695246,
        "coin": "BTC",
        "portion": 0.3954
    },{
        "amount": 0.05294586,
        "coin": "ETH",
        "portion": 0.0995
    },{
        "amount": 0.04120943,
        "coin": "BNB",
        "portion": 0.0034
    },{
        "amount": 0.005,
        "coin": "OMG",
        "portion": 0.0246
    },{
        "amount": 17.19363945,
        "coin": "USDT",
        "portion": 0.1511
    },{
        "amount": 1,
        "coin": "TRX",
        "portion": 0.0002
    }]

    """

    def run(self):
        import json, time
        from apps.backtesting.utils import datetime_to_timestamp
        backtester = PortfolioBacktester()
        timestamp = datetime_to_timestamp('2018/06/01 00:00:00 UTC')
        for i in range(10):
            backtester.process_allocations(timestamp+i*60*60*24, json.loads(self.sample_allocations))
        backtester.value_report()




