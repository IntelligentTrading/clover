import json
import logging

from apps.backtesting.legacy_postgres import POSTGRES
from apps.backtesting.rebalancing.price_providers import PriceProvider


class Allocation:

    def __init__(self, amount, asset, portion, timestamp, counter_currency, db_interface, unit_price=None, value=None):
        self.amount = amount
        self.asset = asset
        self.portion = portion

        if unit_price is not None:
            self.unit_price = unit_price
        else:
            self.unit_price = PRICE_PROVIDER.get_price(asset, timestamp,
                                                       counter_currency=counter_currency,
                                                       db_interface=db_interface)

        if value is not None:
            self.value = value
        else:
            self.value = amount * self.unit_price if self.unit_price is not None else None

        try:
            assert self.value == self.unit_price * self.amount or self.value == None
        except:
            logging.warning(f'Numerical error in allocation, allocation value  is {self.value}, '
                            f'and unit_price*amount is {self.unit_price}*self.amount')

        self.timestamp = timestamp
        self.counter_currency = counter_currency

        if counter_currency != 'USDT':
            self.unit_price_usdt = PRICE_PROVIDER.get_price(asset, timestamp,
                                                            counter_currency='USDT',
                                                            db_interface=db_interface)
        else:
            self.unit_price_usdt = self.unit_price

        self.value_usdt = self.amount * self.unit_price_usdt if self.unit_price_usdt is not None else None
        return


    def to_dict(self, prefix=''):
        return {
            f'{prefix}amount': self.amount,
            f'{prefix}asset': self.asset,
            f'{prefix}portion': self.portion,
            f'{prefix}unit_price': self.unit_price,
            f'{prefix}value': self.value,
            f'{prefix}unit_price_usdt': self.unit_price_usdt,
            f'{prefix}value_usdt': self.value_usdt,
            f'{prefix}timestamp': self.timestamp
        }


class PortfolioSnapshot:
    '''
    A snapshot of a multi-asset portfolio at timestamp. Contains a list of assets with corresponding allocations.
    '''

    def __init__(self, timestamp, allocations_data, db_interface, counter_currency, load_from_json=True):
        self._timestamp = timestamp
        self._allocations_by_asset = {}
        self.db_interface = db_interface
        if load_from_json:
            self._parse_json_allocations(allocations_data, counter_currency)
        else:
            self._allocations = allocations_data
        self._fill_internals()
        self._counter_currency = counter_currency

    def _parse_json_allocations(self, allocations_data, counter_currency):
        self._allocations = []
        allocations_dict = json.loads(allocations_data)
        for item in allocations_dict:
            # figure out the value of this particular item
            allocation = Allocation(**item, unit_price=None, value=None, timestamp=self._timestamp,
                                    counter_currency=counter_currency, db_interface=self.db_interface)
            self._allocations.append(allocation)

    def _fill_internals(self):
        self._held_assets = set()
        self._portion_sum = 0
        self._total_value = 0
        self._total_value_usdt = 0
        for allocation in self._allocations:
            self._held_assets.add(allocation.asset)
            self._portion_sum += float(allocation.portion)
            self._total_value += allocation.value
            self._total_value_usdt += allocation.value_usdt
            self._allocations_by_asset[allocation.asset] = allocation

    def get_allocation(self, asset):
        return self._allocations_by_asset.get(asset, Allocation(amount=0, asset=asset, portion=0,
                                                                timestamp=self._timestamp,
                                                                counter_currency=self._counter_currency,
                                                                db_interface=self.db_interface))

    def total_value(self, counter_currency):
        if counter_currency == 'BTC':
            return self._total_value
        elif counter_currency == 'USDT':
            return self._total_value_usdt

    @property
    def held_assets(self):
        return self._held_assets

    @property
    def portion_sum(self):
        return self._portion_sum

    def report(self):
        logging.info(f'At timestamp {self._timestamp}, portfolio stats are:')
        logging.info(f'    -> total value: {self._total_value}, portion sum: {self._portion_sum}')
        for allocation in self._allocations:
            logging.info(f'       {allocation.amount} {allocation.asset} worth {allocation.value} BTC, {allocation.portion*100:2}% total')

    def update_to_timestamp(self, timestamp):
        '''
        Returns a new PortfolioSnapshot with prices and values of individual assets updated to target timestamp.
        :param timestamp: timestamp to which to update the portfolio
        :return: a new instance of PortfolioSnapshot with updated asset values
        '''
        updated_allocations = []
        for allocation in self._allocations:
            new_price = PRICE_PROVIDER.get_price(allocation.asset, timestamp,
                                                 counter_currency=allocation.counter_currency,
                                                 db_interface=self.db_interface)
            new_allocation = Allocation(amount=allocation.amount, asset=allocation.asset, portion=allocation.portion,
                                        timestamp=timestamp, counter_currency=allocation.counter_currency,
                                        unit_price=new_price, value=new_price * allocation.amount, db_interface=self.db_interface)
            updated_allocations.append(new_allocation)

        return PortfolioSnapshot(timestamp, updated_allocations, load_from_json=False, db_interface=self.db_interface,
                                 counter_currency=self._counter_currency)

    def to_dict(self):
        return {asset: self.get_allocation(asset).to_dict() for asset in self.held_assets}


PRICE_PROVIDER = PriceProvider()