import logging
import json
import pandas as pd
from apps.backtesting.tick_provider import TickerData, TickProvider
from apps.backtesting.legacy_postgres import PostgresDatabaseConnection, NoPriceDataException
from collections import OrderedDict

POSTGRES = PostgresDatabaseConnection()

class Allocation:
    
    def __init__(self, amount, coin, portion, unit_price, value, timestamp, counter_currency='BTC'):
        self.amount = amount
        self.coin = coin
        self.portion = portion
        self.unit_price = unit_price
        self.value = value
        self.timestamp = timestamp
        self.counter_currency = counter_currency

        if counter_currency != 'USDT':
            self.unit_price_usdt = get_price(coin, timestamp, counter_currency='USDT')
        else:
            self.unit_price_usdt = unit_price

        self.value_usdt = self.amount * self.unit_price_usdt


    def to_dict(self, prefix=''):
        return {
            f'{prefix}amount': self.amount,
            f'{prefix}coin': self.coin,
            f'{prefix}portion': self.portion,
            f'{prefix}unit_price': self.unit_price,
            f'{prefix}value': self.value,
            f'{prefix}timestamp': self.timestamp
        }


def get_price_old(coin, timestamp, db_interface=POSTGRES):
    if coin == 'BTC':
        return 1
    elif coin == 'USDT':
        # get BTC_USDT price and invert
        btc_usdt_price, _ = db_interface.get_price_nearest_to_timestamp(currency='BTC',
                                                                          timestamp=timestamp,
                                                                          source=2,  # Binance
                                                                          counter_currency='USDT')
        return 1E8 / btc_usdt_price

    price, _ = db_interface.get_price_nearest_to_timestamp(currency=coin,
                                                            timestamp=timestamp,
                                                            source=2,  # Binance
                                                            counter_currency='BTC')
    return price / 1E8


def get_price(coin, timestamp, db_interface=POSTGRES, counter_currency='BTC'):
    if coin == counter_currency:
        return 1
    elif coin == 'USDT' and counter_currency == 'BTC':
        # get BTC_USDT price and invert
        btc_usdt_price, _ = db_interface.get_price_nearest_to_timestamp(currency='BTC',
                                                                        timestamp=timestamp,
                                                                        source=2,  # Binance
                                                                        counter_currency='USDT',
                                                                        normalize=True)
        return 1.0 / btc_usdt_price

    try:
        price, retrieved_timestamp = db_interface.get_price_nearest_to_timestamp(currency=coin,
                                                           timestamp=timestamp,
                                                           source=2,  # Binance
                                                           counter_currency=counter_currency,
                                                           normalize=True)
        return price
    except NoPriceDataException:
        if counter_currency == 'USDT':
            btc_price, retrieved_timestamp = db_interface.get_price_nearest_to_timestamp(currency=coin,
                                                           timestamp=timestamp,
                                                           source=2,  # Binance
                                                           counter_currency='BTC',
                                                           normalize=True)
            return POSTGRES.convert_value_to_USDT(value=1, timestamp=retrieved_timestamp, transaction_currency=coin, source=2)


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
        self._total_value_usdt = 0
        for allocation in self._allocations:
            self._held_coins.add(allocation.coin)
            self._portion_sum += float(allocation.portion)
            self._total_value += allocation.value
            self._total_value_usdt += allocation.value_usdt
            self._allocations_by_coin[allocation.coin] = allocation

    def get_allocation(self, coin):
        return self._allocations_by_coin.get(coin, None)

    def total_value(self, counter_currency):
        if counter_currency == 'BTC':
            return self._total_value
        elif counter_currency == 'USDT':
            return self._total_value_usdt

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
                                        value=new_price*allocation.amount,
                                        timestamp=timestamp)
            updated_allocations.append(new_allocation)
        return PortfolioSnapshot(timestamp, updated_allocations, load_from_json=False)

    def to_dict(self):
        return {coin: self.get_allocation(coin).to_dict() for coin in self.held_coins}


class PortfolioBacktester:

    def __init__(self, start_time, end_time, step_seconds, portions_dict, start_value_of_portfolio, counter_currency):
        self._start_time = start_time
        self._end_time = end_time
        self._step_seconds = step_seconds
        self._portions_dict = portions_dict
        self._start_value_of_portfolio = start_value_of_portfolio
        self._counter_currency = counter_currency
        self._simulate()
        self._build_benchmark_baselines()

    def _build_portfolio(self, timestamp, total_value):
        allocations = []
        # calculate the held amount for each coin
        for coin in self._portions_dict:
            portion = self._portions_dict[coin]
            unit_price = get_price(coin, timestamp)
            value = portion * total_value
            amount = value / unit_price
            allocation = Allocation(coin=coin, portion=portion, unit_price=unit_price, value=value, amount=amount, timestamp=timestamp)
            allocations.append(allocation)
        return PortfolioSnapshot(timestamp=timestamp, allocations_data=allocations, load_from_json=False)

    def _build_benchmark_baselines(self):
        self._benchmarks = {}
        from apps.backtesting.backtester_ticks import TickDrivenBacktester
        for coin in self.held_coins:
            coin_df = self.get_dataframe_for_coin(coin)
            tick_provider = TickProviderDataframe(transaction_currency=coin,
                                                  counter_currency='BTC',
                                                  source='binance',
                                                  dataframe=coin_df,
                                                  close_price_column_name='unit_price')
            self._benchmarks[coin] = TickDrivenBacktester.build_benchmark(coin, 'BTC',
                                                                          self._start_value_of_portfolio*self._portions_dict[coin],
                                                                          0, self._start_time, self._end_time,
                                                                          source=2, tick_provider=tick_provider,
                                                                          database=POSTGRES)


    def _simulate(self):
        self._portfolio_snapshots = OrderedDict()
        self._dataframes = {}
        value_df_dicts = []
        current_snapshot = None
        for timestamp in range(self._start_time, self._end_time, self._step_seconds):
            try:
                if current_snapshot is None:
                    current_snapshot = self._build_portfolio(self._start_time, self._start_value_of_portfolio)
                else:
                    current_snapshot = self._build_portfolio(
                        timestamp, current_snapshot.update_to_timestamp(timestamp).total_value(self._counter_currency))
                current_snapshot.report()
                logging.info(current_snapshot.to_dict())
                current_value_of_portfolio = current_snapshot.total_value(self._counter_currency)
                coin_values_dict = {}
                for coin in current_snapshot.to_dict().keys():
                    self._dataframes.setdefault(coin, []).append(current_snapshot.to_dict()[coin])
                    allocation_dict = current_snapshot.get_allocation(coin).to_dict(prefix=f'{coin}_')
                    for key in allocation_dict:
                        coin_values_dict[key] = allocation_dict[key]
                    #coin_values_dict[coin] = current_snapshot.get_allocation(coin).value
                coin_values_dict['timestamp'] = timestamp
                coin_values_dict['total_value'] = current_value_of_portfolio

                coin_values_dict['total_value_usdt'] = current_snapshot.total_value('USDT')
                value_df_dicts.append(coin_values_dict)
            except NoPriceDataException as e:
                logging.error(e)
                continue
        self._dataframes = {coin: pd.DataFrame(self._dataframes[coin]).set_index(['timestamp']) for coin in self._dataframes.keys()}
        self._value_dataframe = pd.DataFrame(value_df_dicts).set_index(['timestamp'])
        # self._value_dataframe.index = pd.to_datetime(self._value_dataframe.index, unit='s')

    def process_allocations(self, timestamp, allocations_data):
        self._portfolio_snapshots[timestamp] = PortfolioSnapshot(timestamp, allocations_data)

    def value_report(self):
        for timestamp, snapshot in self._portfolio_snapshots.items():
            snapshot.report()

    def get_dataframe_for_coin(self, coin):
        return self._dataframes.get(coin, None)

    def get_rebalancing_vs_benchmark_dataframe(self):
        value_df = self.value_dataframe
        benchmark_value_df = self.get_benchmark_trading_df_for_all_coins()

        df = value_df.join(benchmark_value_df, lsuffix='_rebalancing', rsuffix='_benchmark')
        columns_list = [f'total_value_{coin}' for coin in self.held_coins]
        columns_list.append('total_value_rebalancing')
        columns_list.append('total_value_benchmark')
        df.index = pd.to_datetime(df.index, unit='s')
        return df

    @property
    def value_dataframe(self):
        return self._value_dataframe

    @property
    def held_coins(self):
        return self._portions_dict.keys()

    def get_benchmark_for_coin(self, coin):
        return self._benchmarks.get(coin, None)

    def get_benchmark_trading_dataframe_for_coin(self, coin):
        if coin not in self._benchmarks:
            return None
        return self._benchmarks[coin].trading_df

    def get_benchmark_trading_df_for_all_coins(self):
        df = None
        for coin in self.held_coins:
            if df is None:
                df = self._benchmarks[coin].trading_df.copy()
                df = df.add_suffix(f'_{coin}')
            else:
                right = self._benchmarks.get(coin, None).trading_df.add_suffix(f'_{coin}')
                df = df.join(right)
        sum_columns = [f'total_value_{coin}' for coin in self.held_coins]
        df['total_value'] = df[sum_columns].sum(axis=1)

        return df



class TickProviderDataframe(TickProvider):

    def __init__(self, transaction_currency, counter_currency, source, dataframe, close_price_column_name):
        super(TickProviderDataframe, self).__init__()
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._source = source
        self._dataframe = dataframe
        self._close_price_column_name = close_price_column_name


    def run(self):
        for timestamp, row in self._dataframe.iterrows():
            close_price = row[self._close_price_column_name]
            ticker_data = TickerData(
                timestamp=timestamp,
                transaction_currency=self._transaction_currency,
                counter_currency=self._counter_currency,
                source=self._source,
                resample_period=None,
                open_price=close_price,  # row.open_price,
                high_price=close_price,  # row.high_price,
                low_price=close_price,  # row.low_price,
                close_price=close_price,
                close_volume=0,
                signals=[],
            )
            self.notify_listeners(ticker_data)
        self.broadcast_ended()


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
        # backtester = PortfolioBacktester()
        # timestamp = datetime_to_timestamp('2018/06/01 00:00:00 UTC')
        # for i in range(10):
        #     backtester.process_allocations(timestamp+i*60*60*24, self.sample_allocations)
        # backtester.value_report()
        backtester = PortfolioBacktester(start_time=int(datetime_to_timestamp('2018/06/01 00:00:00 UTC')),
                            end_time=int(datetime_to_timestamp('2018/06/02 00:00:00 UTC')),
                            step_seconds=60*60,
                            portions_dict={
                                'BTC': 0.5,
                                'ETH': 0.25,
                                'OMG': 0.25
                            },
                            start_value_of_portfolio=1000,
                            counter_currency='BTC')
        backtester.get_benchmark_trading_dataframe_for_coin('ETH')
        backtester.get_benchmark_trading_df_for_all_coins()
        backtester.get_rebalancing_vs_benchmark_dataframe()




