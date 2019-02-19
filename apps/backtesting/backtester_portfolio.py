import logging
import json
from apps.backtesting.data_sources import DB_INTERFACE
from apps.backtesting.legacy_postgres import PostgresDatabaseConnection
from collections import namedtuple, OrderedDict

Allocation = namedtuple('Allocation', 'amount coin portion unit_price value')

POSTGRES = PostgresDatabaseConnection()

"""
class Allocation:
    
    def __init__(self, amount, coin, portion, unit_price=None):
        self.amount = amount
        self.coin = coin
        self.portion = portion
        if unit_price is None:
            
        
    def _fill_internals(self, allocations_dict):
            unit_price = self._get_price(item['coin'])
            value = unit_price * item['amount']
            allocation = Allocation(**item, unit_price=unit_price, value=value)
            self._held_coins.add(allocation.coin)
            self._portion_sum += float(allocation.portion)
            self._total_value += allocation.value
            self._allocations[allocation.coin] = allocation
        

"""


def get_price(coin, timestamp, db_interface=POSTGRES):
    if coin == 'BTC':
        return 1
    elif coin == 'USDT':
        # get BTC_USDT price and invert
        btc_usdt_price = db_interface.get_price_nearest_to_timestamp(currency='BTC',
                                                                          timestamp=timestamp,
                                                                          source=2,  # Binance
                                                                          counter_currency='USDT') / 1E8
        return 1.0 / btc_usdt_price

    return db_interface.get_price_nearest_to_timestamp(currency=coin,
                                                            timestamp=timestamp,
                                                            source=2,  # Binance
                                                            counter_currency='BTC') / 1E8

class PortfolioSnapshot:

    def __init__(self, timestamp, allocations_data, db_interface=POSTGRES, load_from_json=True):
        self._timestamp = timestamp
        self._allocations_by_coin = {}
        self.db_interface = db_interface
        if load_from_json:
            self._parse_json_allocations(allocations_data)
        else:
            self._allocations = allocations_data
        self._fill_internals()

    def _parse_json_allocations(self, allocations_data):
        self._allocations = []
        allocations_dict = json.loads(allocations_data)
        for item in allocations_dict:
            # figure out the value of this particular item
            unit_price = get_price(item['coin'], self._timestamp)
            value = unit_price * item['amount']
            allocation = Allocation(**item, unit_price=unit_price, value=value)
            self._allocations.append(allocation)

    def _fill_internals(self):
        self._held_coins = set()
        self._portion_sum = 0
        self._total_value = 0
        for allocation in self._allocations:
            self._held_coins.add(allocation.coin)
            self._portion_sum += float(allocation.portion)
            self._total_value += allocation.value
            self._allocations_by_coin[allocation.coin] = allocation

    def get_allocation(self, coin):
        return self._allocations_by_coin.get(coin, None)

    @property
    def total_value(self):
        return self._total_value

    @property
    def held_coins(self):
        return self._held_coins

    @property
    def portion_sum(self):
        return self._portion_sum

    def report(self):
        logging.info(f'At timestamp {self._timestamp}, portfolio stats are:')
        logging.info(f'    -> total value: {self._total_value}, portion sum: {self._portion_sum}')
        for allocation in self._allocations:
            logging.info(f'       {allocation.amount} {allocation.coin} worth {allocation.value} BTC, {allocation.portion*100:2}% total')

    def update_to_timestamp(self, timestamp):
        updated_allocations = []
        for allocation in self._allocations:
            new_price = get_price(allocation.coin, timestamp)
            new_allocation = Allocation(amount=allocation.amount,
                                        coin=allocation.coin,
                                        portion=allocation.portion,
                                        unit_price=new_price,
                                        value=new_price*allocation.amount)
            updated_allocations.append(new_allocation)
        return PortfolioSnapshot(timestamp, updated_allocations, load_from_json=False)


class PortfolioBacktester:

    def __init__(self):
        self._portfolio_snapshots = OrderedDict()

    def simulate(self, start_time, end_time, step_seconds, portions_dict, start_value_of_portfolio):
        snapshot = None
        current_value_of_portfolio = start_value_of_portfolio
        for timestamp in range(start_time, end_time, step_seconds):
            if snapshot is not None:
                current_value_of_portfolio = snapshot.update_to_timestamp(timestamp).total_value
            allocations = []
            # calculate the held amount for each coin
            for coin in portions_dict:
                portion = portions_dict[coin]
                unit_price = get_price(coin, timestamp)
                value = portion * current_value_of_portfolio
                amount = value / unit_price
                allocation = Allocation(coin=coin, portion=portion, unit_price=unit_price, value=value, amount=amount)
                allocations.append(allocation)
            snapshot = PortfolioSnapshot(timestamp=timestamp, allocations_data=allocations, load_from_json=False)
            snapshot.report()
            current_value_of_portfolio = snapshot.total_value

    def process_allocations(self, timestamp, allocations_data):
        self._portfolio_snapshots[timestamp] = PortfolioSnapshot(timestamp, allocations_data)

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
        from apps.backtesting.utils import datetime_to_timestamp
        backtester = PortfolioBacktester()
        timestamp = datetime_to_timestamp('2018/06/01 00:00:00 UTC')
        for i in range(10):
            backtester.process_allocations(timestamp+i*60*60*24, self.sample_allocations)
        backtester.value_report()
        backtester.simulate(start_time=int(datetime_to_timestamp('2018/06/01 00:00:00 UTC')),
                            end_time=int(datetime_to_timestamp('2018/06/02 00:00:00 UTC')),
                            step_seconds=60*60,
                            portions_dict={
                                'BTC': 0.5,
                                'ETH': 0.25,
                                'OMG': 0.25
                            },
                            start_value_of_portfolio=1000)




