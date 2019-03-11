import logging
import json
import pandas as pd
import itertools
from apps.backtesting.tick_provider import TickerData, TickProvider
from apps.backtesting.legacy_postgres import PostgresDatabaseConnection, NoPriceDataException
from collections import OrderedDict
from apps.backtesting.utils import datetime_to_timestamp
from abc import ABC, abstractmethod


POSTGRES = PostgresDatabaseConnection()


class PriceProvider:

    def __init__(self):
        self.providers = [cls() for cls in PriceProvider.__subclasses__()]
        self.default_provider = CryptoPriceProvider()

    def get_price(self, asset, timestamp, **kwargs):
        for provider in self.providers:
            if provider.can_handle(asset):
                provider.get_price(asset, timestamp, **kwargs)
        return self.default_provider.get_price(asset, timestamp, **kwargs)

    def can_handle(self, asset_name):
        pass


class CryptoPriceProvider(PriceProvider):

    def __init__(self):
        pass

    def get_price(self, coin, timestamp, db_interface=POSTGRES, counter_currency='BTC'):
        price = None
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
                return POSTGRES.convert_value_to_USDT(value=1, timestamp=retrieved_timestamp, transaction_currency=coin,
                                                      source=2)

        if price is None:
            raise NoPriceDataException

    def can_handle(self, asset_name):
        return False


class GoldPriceProvider(PriceProvider):

    def __init__(self, csv_path='../../GLD.csv'):
        # load the CSV file data
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        self.price_df = df

    def get_price(self, timestamp):
        dt = pd.to_datetime(timestamp, unit='s')
        result = self.price_df.iloc[self.price_df.index.get_loc(dt, method='nearest')]
        delta_time = abs(result.name.timestamp() - dt.timestamp())
        if delta_time > 60*60*24*3:
            logging.warning(f'Retrieved price of gold more than +-3 days from the requested timestamp! '
                            f'(requested: {dt}, retrieved: {result.name})')
        return result.Close

    def can_handle(self, asset_name):
        return asset_name == 'GLD_ETF'


PRICE_PROVIDER = PriceProvider()

class Allocation:
    
    def __init__(self, amount, asset, portion, unit_price, value, timestamp, counter_currency='BTC'):
        self.amount = amount
        self.asset = asset
        self.portion = portion
        self.unit_price = unit_price
        self.value = value
        self.timestamp = timestamp
        self.counter_currency = counter_currency

        if counter_currency != 'USDT':
            self.unit_price_usdt = PRICE_PROVIDER.get_price(asset, timestamp, counter_currency='USDT')
        else:
            self.unit_price_usdt = unit_price

        self.value_usdt = self.amount * self.unit_price_usdt


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

    def __init__(self, timestamp, allocations_data, db_interface=POSTGRES, load_from_json=True):
        self._timestamp = timestamp
        self._allocations_by_asset = {}
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
            unit_price = PRICE_PROVIDER.get_price(item['asset'], self._timestamp)
            value = unit_price * item['amount']
            allocation = Allocation(**item, unit_price=unit_price, value=value)
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
        return self._allocations_by_asset.get(asset, None)

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
        updated_allocations = []
        for allocation in self._allocations:
            new_price = PRICE_PROVIDER.get_price(allocation.asset, timestamp)
            new_allocation = Allocation(amount=allocation.amount,
                                        asset=allocation.asset,
                                        portion=allocation.portion,
                                        unit_price=new_price,
                                        value=new_price*allocation.amount,
                                        timestamp=timestamp)
            updated_allocations.append(new_allocation)

        return PortfolioSnapshot(timestamp, updated_allocations, load_from_json=False)

    def to_dict(self):
        return {asset: self.get_allocation(asset).to_dict() for asset in self.held_assets}


class PortfolioBacktester:

    def __init__(self, start_time, end_time, step_seconds, portions_dict,
                 start_value_of_portfolio, counter_currency, verbose=False, trading_cost_percent=0):
        if isinstance(start_time, str):
            start_time = int(datetime_to_timestamp(start_time))
        if isinstance(end_time, str):
            end_time = int(datetime_to_timestamp(end_time))
        self._start_time = start_time
        self._end_time = end_time
        self._step_seconds = step_seconds
        self._portions_dict = portions_dict
        self._start_value_of_portfolio = start_value_of_portfolio
        self._start_value_of_portfolio_usdt = start_value_of_portfolio * PRICE_PROVIDER.get_price('BTC', start_time, counter_currency='USDT')
        self._counter_currency = counter_currency
        self._verbose = verbose
        self._trading_cost_percent = trading_cost_percent
        self._simulate()
        self._build_benchmark_baselines()
        self._fill_benchmark_dataframe()

    def _build_portfolio(self, timestamp, total_value):
        allocations = []
        # calculate the held amount for each asset
        for asset in self._portions_dict:
            portion = self._portions_dict[asset]
            unit_price = PRICE_PROVIDER.get_price(asset, timestamp)
            value = (portion * total_value) * (1 - self._trading_cost_percent/100)
            amount = value / unit_price
            allocation = Allocation(asset=asset, portion=portion, unit_price=unit_price, value=value, amount=amount, timestamp=timestamp)
            allocations.append(allocation)
        return PortfolioSnapshot(timestamp=timestamp, allocations_data=allocations, load_from_json=False)


    def _build_portfolio_with_trading_fee(self, timestamp, total_value, previous_portfolio):
        if previous_portfolio is None:
            return self._build_portfolio(timestamp, total_value)
        allocations = []
        trading_fee = self._trading_cost_percent / 100
        # calculate the held amount for each asset
        for asset in self._portions_dict:
            previous = previous_portfolio.get_allocation(asset)
            new_unit_price = PRICE_PROVIDER.get_price(asset, timestamp)
            portion = self._portions_dict[asset]
            delta_value = abs(total_value*portion - previous.amount*new_unit_price)
            fee = delta_value * trading_fee
            obtained_amount = (delta_value - fee) / new_unit_price
            new_amount = previous.amount + (obtained_amount if total_value*portion > previous.amount*new_unit_price else -obtained_amount)
            new_value = new_amount * new_unit_price
            portion = new_value / total_value
            # amount = value / unit_price
            allocation = Allocation(asset=asset, portion=portion, unit_price=new_unit_price, value=new_value, amount=new_amount, timestamp=timestamp)
            allocations.append(allocation)
        total_value = sum([allocation.value for allocation in allocations])
        for allocation in allocations:
            allocation.portion = allocation.value / total_value

        p = PortfolioSnapshot(timestamp=timestamp, allocations_data=allocations, load_from_json=False)
        return p


    def _build_benchmark_baselines(self):
        self._benchmarks = {}
        self._usdt_benchmarks = {}
        from apps.backtesting.backtester_ticks import TickDrivenBacktester
        for asset in self.held_assets:
            asset_df = self.get_dataframe_for_asset(asset)
            tick_provider = TickProviderDataframe(transaction_currency=asset,
                                                  counter_currency='BTC',
                                                  source='binance',
                                                  dataframe=asset_df,
                                                  close_price_column_name='unit_price')
            self._benchmarks[asset] = TickDrivenBacktester.build_benchmark(asset, 'BTC',
                                                                          self._start_value_of_portfolio*self._portions_dict[asset],
                                                                          0, self._start_time, self._end_time,
                                                                          source=2, tick_provider=tick_provider,
                                                                          database=POSTGRES)
            tick_provider_usdt = TickProviderDataframe(transaction_currency=asset,
                                                  counter_currency='USDT',
                                                  source='binance',
                                                  dataframe=asset_df,
                                                  close_price_column_name='unit_price_usdt')
            self._usdt_benchmarks[asset] = TickDrivenBacktester.build_benchmark(asset, 'USDT',
                                                                          self._start_value_of_portfolio_usdt*self._portions_dict[asset],
                                                                          0, self._start_time, self._end_time,
                                                                          source=2, tick_provider=tick_provider_usdt,
                                                                          database=POSTGRES)


    def _simulate(self):
        self._portfolio_snapshots = OrderedDict()
        self._dataframes = {}
        value_df_dicts = []
        current_snapshot = None
        for timestamp in range(self._start_time, self._end_time+1, self._step_seconds):
            try:
                if current_snapshot is None:
                    current_snapshot = self._build_portfolio_with_trading_fee(self._start_time, self._start_value_of_portfolio, previous_portfolio=None)
                else:
                    current_snapshot = self._build_portfolio_with_trading_fee(
                        timestamp, current_snapshot.update_to_timestamp(timestamp).total_value(self._counter_currency),
                        previous_portfolio=previous_snapshot)
                if self._verbose:
                    current_snapshot.report()
                logging.info(current_snapshot.to_dict())
                current_value_of_portfolio = current_snapshot.total_value(self._counter_currency)
                asset_values_dict = {}
                for asset in current_snapshot.to_dict().keys():
                    self._dataframes.setdefault(asset, []).append(current_snapshot.to_dict()[asset])
                    allocation_dict = current_snapshot.get_allocation(asset).to_dict(prefix=f'{asset}_')
                    for key in allocation_dict:
                        asset_values_dict[key] = allocation_dict[key]
                    #asset_values_dict[asset] = current_snapshot.get_allocation(asset).value
                asset_values_dict['timestamp'] = timestamp
                asset_values_dict['total_value'] = current_value_of_portfolio
                asset_values_dict['total_value_usdt'] = current_snapshot.total_value('USDT')
                value_df_dicts.append(asset_values_dict)
                previous_snapshot = current_snapshot
            except NoPriceDataException as e:
                logging.error(e)
                continue
        self._dataframes = {asset: pd.DataFrame(self._dataframes[asset]).set_index(['timestamp']) for asset in self._dataframes.keys()}
        self._value_dataframe = pd.DataFrame(value_df_dicts).set_index(['timestamp'])
        self._value_dataframe = self._fill_relative_returns(self._value_dataframe,
                                                            total_value_column_name='total_value',
                                                            relative_returns_column_name='return_relative_to_past_tick')
        self._value_dataframe = self._fill_relative_returns(self._value_dataframe,
                                                            total_value_column_name='total_value_usdt',
                                                            relative_returns_column_name='return_relative_to_past_tick_usdt')

        # self._value_dataframe.index = pd.to_datetime(self._value_dataframe.index, unit='s')

    def _fill_relative_returns(self, df, total_value_column_name='total_value', relative_returns_column_name='return_relative_to_past_tick'):
        df['return_relative_to_past_tick'] = df[total_value_column_name].diff() / df[total_value_column_name].shift(1)
        return df

    def process_allocations(self, timestamp, allocations_data):
        self._portfolio_snapshots[timestamp] = PortfolioSnapshot(timestamp, allocations_data)

    def value_report(self):
        for timestamp, snapshot in self._portfolio_snapshots.items():
            snapshot.report()

    def get_dataframe_for_asset(self, asset):
        return self._dataframes.get(asset, None)

    def get_rebalancing_vs_benchmark_dataframe(self):
        value_df = self.value_dataframe
        benchmark_value_df = self.get_benchmark_trading_df_for_all_assets()

        df = value_df.join(benchmark_value_df, lsuffix='_rebalancing', rsuffix='_benchmark')
        columns_list = [f'total_value_{asset}' for asset in self.held_assets]
        columns_list.append('total_value_rebalancing')
        columns_list.append('total_value_benchmark')
        df.index = pd.to_datetime(df.index, unit='s')
        return df

    @property
    def value_dataframe(self):
        return self._value_dataframe

    @property
    def held_assets(self):
        return self._portions_dict.keys()

    def get_benchmark_for_asset(self, asset):
        return self._benchmarks.get(asset, None)

    def get_benchmark_trading_dataframe_for_asset(self, asset):
        if asset not in self._benchmarks:
            return None
        return self._benchmarks[asset].trading_df

    def _fill_benchmark_dataframe(self):
        df = None
        for asset in self.held_assets:
            if df is None:
                df = self._benchmarks[asset].trading_df.copy()
                df = df.add_suffix(f'_{asset}')
                right_usdt = self._usdt_benchmarks[asset].trading_df.copy().add_suffix(f'_usdt_{asset}')
                df = df.join(right_usdt)
            else:
                right = self._benchmarks.get(asset, None).trading_df.add_suffix(f'_{asset}')
                df = df.join(right)
                right_usdt = self._usdt_benchmarks[asset].trading_df.copy().add_suffix(f'_usdt_{asset}')
                df = df.join(right_usdt)
        sum_columns = [f'total_value_{asset}' for asset in self.held_assets]
        df['total_value'] = df[sum_columns].sum(axis=1)
        df = self._fill_relative_returns(df, total_value_column_name='total_value', 
                                         relative_returns_column_name='return_relative_to_past_tick')

        sum_columns_usdt = [f'total_value_usdt_{asset}' for asset in self.held_assets]
        df['total_value_usdt'] = df[sum_columns_usdt].sum(axis=1)
        df = self._fill_relative_returns(df, total_value_column_name='total_value_usdt', 
                                         relative_returns_column_name='return_relative_to_past_tick_usdt')

        self._benchmark_dataframe = df

    def get_benchmark_trading_df_for_all_assets(self):
        return self._benchmark_dataframe

    @property
    def profit(self):
        end_value = self.value_dataframe.iloc[-1].total_value
        return end_value - self._start_value_of_portfolio

    @property
    def profit_usdt(self):
        end_value = self.value_dataframe.iloc[-1].total_value_usdt
        return end_value - self._start_value_of_portfolio_usdt

    @property
    def profit_percent(self):
        return self.profit / float(self._start_value_of_portfolio)

    @property
    def profit_percent_usdt(self):
        return self.profit_usdt / float(self._start_value_of_portfolio_usdt)

    @property
    def benchmark_profit(self):
        end_value = self._benchmark_dataframe.iloc[-1].total_value
        return end_value - self._start_value_of_portfolio

    @property
    def benchmark_profit_usdt(self):
        end_value = self._benchmark_dataframe.iloc[-1].total_value_usdt
        return end_value - self._start_value_of_portfolio_usdt

    @property
    def benchmark_profit_percent(self):
        return self.benchmark_profit / float(self._start_value_of_portfolio)

    @property
    def benchmark_profit_percent_usdt(self):
        return self.benchmark_profit_usdt / float(self._start_value_of_portfolio_usdt)

    @property
    def gain_over_benchmark(self):
        return self.profit - self.benchmark_profit

    @property
    def gain_over_benchmark_usdt(self):
        return self.profit_usdt - self.benchmark_profit_usdt

    @property
    def percent_gain_over_benchmark(self):
        return self.gain_over_benchmark / self._start_value_of_portfolio

    @property
    def percent_gain_over_benchmark_usdt(self):
        return self.gain_over_benchmark_usdt / self._start_value_of_portfolio_usdt

    @property
    def summary_dict(self):
        return {
            'allocations': ', '.join([f'{asset} ({self.get_portion(asset)*100:.0f}%)' for asset in self.held_assets]),
            'profit_percent': self.profit_percent,
            'profit_percent_usdt': self.profit_percent_usdt,
            'benchmark_profit_percent': self.benchmark_profit_percent,
            'benchmark_profit_percent_usdt': self.benchmark_profit_percent_usdt,
            'percent_gain_over_benchmark': self.percent_gain_over_benchmark,
            'percent_gain_over_benchmark_usdt': self.percent_gain_over_benchmark_usdt,
            'gain_over_benchmark': self.gain_over_benchmark,
            'gain_over_benchmark_usdt': self.gain_over_benchmark_usdt
        }

    def get_portion(self, asset):
        return self._portions_dict.get(asset, None)

    def draw_returns_tear_sheet(self, save_file=True, out_filename='pyfolio_returns_tear_sheet.png'):
        import pyfolio as pf
        import matplotlib
        # if save_file:
        #   matplotlib.use('Agg')

        df = self.get_rebalancing_vs_benchmark_dataframe()
        df = df.rename(columns={"return_relative_to_past_tick_benchmark": "Buy & hold"})
        f = pf.create_returns_tear_sheet(returns=df['return_relative_to_past_tick_rebalancing'],
                                         return_fig=True,
                                         bootstrap=None,
                                         benchmark_rets=df['Buy & hold'])

        if save_file:
            f.savefig(out_filename)
        return f

    def plot_returns(self, title=None):
        if title is None:
            title = self.summary_dict['allocations']
        self.get_rebalancing_vs_benchmark_dataframe()[
            ['total_value_usdt_rebalancing', 'total_value_usdt_benchmark']].plot(title=title)


class ComparativePortfolioEvaluation:
    def __init__(self, portion_dicts, start_time, end_time, rebalancing_periods, start_value_of_portfolio, counter_currency,
                 trading_cost_percent=0):
        self._portion_dicts = portion_dicts
        self._start_time = start_time
        self._end_time = end_time
        self._rebalancing_periods = rebalancing_periods
        self._start_value_of_portfolio = start_value_of_portfolio
        self._counter_currency = counter_currency
        self._trading_cost_percent = trading_cost_percent
        self._run()

    def _run(self):
        df_rows = []
        self._portfolio_backtests = {}
        for portfolio_name, rebalancing_period in itertools.product(self._portion_dicts.keys(), self._rebalancing_periods):
            portions_dict = self._portion_dicts[portfolio_name]
            portfolio_backtest = PortfolioBacktester(start_time=self._start_time,
                                                     end_time=self._end_time,
                                                     step_seconds=rebalancing_period,
                                                     portions_dict=portions_dict,
                                                     start_value_of_portfolio=self._start_value_of_portfolio,
                                                     counter_currency=self._counter_currency,
                                                     trading_cost_percent=self._trading_cost_percent)
            result = portfolio_backtest.summary_dict
            result['portfolio'] = portfolio_name
            result['rebalancing_period_hours'] = rebalancing_period / 60 / 60
            df_rows.append(result)
            self._portfolio_backtests[(portfolio_name, rebalancing_period)] = portfolio_backtest
        self._comparative_df = pd.DataFrame(df_rows)

    @property
    def comparative_df(self):
        return self._comparative_df

    def plot_all_returns(self):
        for (portfolio_name, rebalancing_period), portfolio_backtest in self._portfolio_backtests.items():
            title = portfolio_backtest.summary_dict['allocations']
            portfolio_backtest.plot_returns(title=f'{portfolio_name} / {title} / rebalanced every {rebalancing_period /60/60:.0f} hours')



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
        "asset": "BTC",
        "portion": 0.3954
    },{
        "amount": 0.05294586,
        "asset": "ETH",
        "portion": 0.0995
    },{
        "amount": 0.04120943,
        "asset": "BNB",
        "portion": 0.0034
    },{
        "amount": 0.005,
        "asset": "OMG",
        "portion": 0.0246
    },{
        "amount": 17.19363945,
        "asset": "USDT",
        "portion": 0.1511
    },{
        "amount": 1,
        "asset": "TRX",
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
        backtester = PortfolioBacktester(start_time=int(datetime_to_timestamp('2018/10/01 00:00:00 UTC')),
                            end_time=int(datetime_to_timestamp('2018/10/30 00:00:00 UTC')),
                            step_seconds=60*60,
                            portions_dict={
                                'BTC': 0.25,
                                'ETH': 0.25,
                                'XRP': 0.25,
                                'EOS': 0.25,
                            },
                            start_value_of_portfolio=1000,
                            counter_currency='BTC',
                            trading_cost_percent=0.1)
        backtester.draw_returns_tear_sheet()
        backtester.get_benchmark_trading_dataframe_for_asset('ETH')
        backtester.get_benchmark_trading_df_for_all_assets()
        backtester.get_rebalancing_vs_benchmark_dataframe()




