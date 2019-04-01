import itertools

import pandas as pd

from apps.backtesting.rebalancing.backtester_portfolio import FixedRatiosPortfolioBacktester


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
            portfolio_backtest = FixedRatiosPortfolioBacktester(start_time=self._start_time,
                                                     end_time=self._end_time,
                                                     step_seconds=rebalancing_period,
                                                     rebalancing_period_seconds=rebalancing_period,
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

    def save_return_figs(self, out_folder, img_format='svg'):
        import os
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        for (portfolio_name, rebalancing_period), portfolio_backtest in self._portfolio_backtests.items():
            title = portfolio_backtest.summary_dict['allocations']
            hours = int(rebalancing_period /60/60)
            chart_title = f'{portfolio_name} / {title} / rebalanced every {hours} {"hours" if hours != 1 else "hour"}'
            out_path = os.path.join(out_folder, f'{portfolio_name}_{hours}.{img_format}')
            portfolio_backtest.save_returns_plot(out_path, title=chart_title)