import itertools
import numpy as np
from apps.backtesting.data_sources import DB_INTERFACE
from apps.backtesting.backtester_signals import SignalDrivenBacktester
from apps.backtesting.config import backtesting_report_columns, backtesting_report_column_names, COINMARKETCAP_TOP_20_ALTS
from apps.backtesting.signals import ALL_SIGNALS
from apps.backtesting.strategies import BuyAndHoldTimebasedStrategy, SignalSignatureStrategy, SimpleRSIStrategy
from enum import Enum
from apps.backtesting.order_generator import OrderGenerator
from apps.backtesting.config import POOL_SIZE
from apps.backtesting.utils import time_performance
from collections import namedtuple
import pandas.io.formats.excel
from apps.backtesting.utils import parallel_run
import xlsxwriter

Ticker = namedtuple("Ticker", "source transaction_currency counter_currency")



pandas.io.formats.excel.header_style = None


class SignalCombinationMode(Enum):
    ANY = 0
    SAME_TYPE = 1
    SAME_TYPE_AND_STRENGTH = 2


class StrategyEvaluationSetBuilder:
    """
    Builds a set of strategies for comparative backtest by combining signals.
    """

    @staticmethod
    def build_from_signal_set(buy_signals, sell_signals, num_buy, num_sell, signal_combination_mode):
        """
        Builds a set of strategies based on a set of buy signals and a set of sell signals.
        :param buy_signals: A list of signal signatures to be used for buying (see ALL_SIGNALS).
        :param sell_signals: A list of signal signatures to be used for selling (see ALL_SIGNALS).
        :param num_buy: Number of different signals a strategy should use to buy.
        :param num_sell: Number of different signals a strategy should use to sell.
        :param signal_combination_mode: Indicates how to build strategies.
               Options: SignalCombinationMode.ANY: the signal type of buy and sell signals need not match (e.g. RSI buy, SMA sell)
                        SignalCombinationMode.SAME_TYPE: the signal type of buy and sell signals must match.
                        SignalCombinationMode.SAME_TYPE_AND_STRENGTH: the signal type and strength of buy and sell signals must match.
        :return: A list of all SignalSignature strategies satisfying the building constraints.
        """
        strategies = []

        # create all possible strategies
        buy_combinations = []
        for i in range(1, num_buy+1):
            sample = [list(x) for x in itertools.combinations(buy_signals, i)]
            buy_combinations.extend(sample)

        sell_combinations = []
        for i in range(1, num_sell+1):
            sample = [list(x) for x in itertools.combinations(sell_signals, i)]
            sell_combinations.extend(sample)

        buy_sell_pairs = itertools.product(buy_combinations, sell_combinations)

        # filter out those not satisfying the combination criteria
        for buy, sell in buy_sell_pairs:
            if not StrategyEvaluationSetBuilder.check_signal_combination_mode(buy, sell, signal_combination_mode):
                continue
            signal_set = list(buy)
            signal_set.extend(list(sell))
            if len(signal_set) == 0:
                continue

            strategy = SignalSignatureStrategy(signal_set)
            strategies.append(strategy)

        return strategies

    @staticmethod
    def check_signal_combination_mode(buy_signal_set, sell_signal_set, signal_combination_mode):
        """
        Checks if a particular set of buy signals and set signals satisfies the combination criterion.
        :param buy_signal_set: A set of buy signal signatures.
        :param sell_signal_set: A set of sell signal signatures.
        :param signal_combination_mode: Instance of SignalCombinationMode.
        :return: True if the combination of buy and sell signals is valid according to
        """
        if signal_combination_mode == SignalCombinationMode.ANY:
            return True
        buy_types = set([ALL_SIGNALS[x].signal for x in buy_signal_set])
        sell_types = set([ALL_SIGNALS[x].signal for x in sell_signal_set])
        if signal_combination_mode == SignalCombinationMode.SAME_TYPE:
            return buy_types == sell_types and len(buy_types) == 1
        elif signal_combination_mode == SignalCombinationMode.SAME_TYPE_AND_STRENGTH:
            buy_strengths = set([ALL_SIGNALS[x].strength for x in buy_signal_set])
            sell_strengths = set([ALL_SIGNALS[x].strength for x in sell_signal_set])

            return buy_types == sell_types and len(buy_types) == 1 and \
                   buy_strengths == sell_strengths and len(buy_strengths) == 1

    @staticmethod
    def build_from_rsi_thresholds(signal_type, overbought_thresholds, oversold_thresholds):
        """
        Builds a set of strategies as a Cartesian product of lists of overbought and oversold thresholds.
        :param signal_type: "RSI" or "RSI_Cumulative".
        :param overbought_thresholds: A list of overbought thresholds.
        :param oversold_thresholds: A list of oversold thresholds.
        :return: A list of strategies.
        """

        strategies = []
        for overbought in overbought_thresholds:
            for oversold in oversold_thresholds:
                strategy = SimpleRSIStrategy(overbought, oversold, signal_type)
                strategies.append(strategy)
        return strategies


class ComparativeEvaluation:
    """
    Comparatively backtests a set of strategies on given currency pairs, resample periods and exchanges.
    Uses SignalDrivenBacktester.
    """

    def __init__(self, strategy_set, start_cash, start_crypto, start_time, end_time, resample_periods,
                 counter_currencies=None, sources=None, tickers=None, output_file=None, time_delay=0, debug=False,
                 order_generator=OrderGenerator.ALTERNATING, parallelize=True):

        self.strategy_set = sorted(strategy_set)
        self.counter_currencies = counter_currencies
        self.resample_periods = resample_periods
        self.sources = sources
        self.start_time = start_time
        self.end_time = end_time
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.evaluate_profit_on_last_order = False
        self.buy_first_and_hold = False
        self.output_file = output_file
        self.time_delay = time_delay
        self.order_generator = order_generator
        self._parallelize = parallelize

        if tickers is None:
            self.tickers = self.build_tickers(counter_currencies, sources)
        else:
            self.tickers = tickers

        self._run_backtests(debug)
        self._report = ComparativeReportBuilder(self.backtests, self.baselines)
        try:
            best_backtest = self._report.get_best_performing_backtest()
            logging.info(best_backtest.get_report())
        except Exception as e:
            logging.error(f'Error in finding best backtest: {str(e)}')

        if output_file is not None:
            self._report.write_summary(output_file)

    @staticmethod
    def build_tickers( counter_currencies, sources):
        currency_tuples = []
        for source, counter_currency in itertools.product(sources, counter_currencies):
            currency_tuples += [Ticker(source, transaction_currency, counter_currency)
                                for transaction_currency in DB_INTERFACE.get_currencies_trading_against_counter(counter_currency, source)]
        return currency_tuples


    @time_performance
    def _run_backtests(self, debug):
        self.backtests = []
        self.baselines = []
        param_list = []

        for strategy, resample_period, ticker in \
                itertools.product(self.strategy_set, self.resample_periods, self.tickers):

            params = {}
            params['strategy'] = strategy
            params['start_time'] = self.start_time
            params['end_time'] = self.end_time
            params['transaction_currency'] = ticker.transaction_currency
            params['counter_currency'] = ticker.counter_currency
            params['resample_period'] = resample_period
            params['start_cash'] = self.start_cash
            params['start_crypto'] = self.start_crypto
            params['evaluate_profit_on_last_order'] = self.evaluate_profit_on_last_order
            params['verbose'] = False
            params['source'] = ticker.source
            params['order_generator'] = self.order_generator

            param_list.append(params)

        if self._parallelize:
            # from pathos.multiprocessing import ProcessingPool as Pool
            # with Pool(POOL_SIZE) as pool:
            #    backtests = pool.map(self._evaluate, param_list)
            #    pool.close()
            #    pool.join()
            backtests = parallel_run(self._evaluate, param_list)
            logging.info("Parallel processing finished.")
        else:
            backtests = map(self._evaluate, param_list)

        for backtest in backtests:
            if backtest is None:
                continue
            if backtest.profit_percent is None or backtest.benchmark_backtest.profit_percent is None:
                    continue
            self.backtests.append(backtest)
            self.baselines.append(backtest.benchmark_backtest)
            if debug:
                break

        logging.info("Finished backtesting, building report...")


    def _evaluate(self, params):
        logging.info(f"Evaluating strategy {params['strategy'].get_short_summary()}, "
                     f"{params['transaction_currency']}-{params['counter_currency']}, "
                     f"start_time {self.start_time}, end_time {self.end_time}...")
        strategy = params['strategy']
        strategy.clear_state()
        del params['strategy']
        try:
            baseline = BuyAndHoldTimebasedStrategy(self.start_time, self.end_time, params['transaction_currency'],
                                                   params['counter_currency'], params['source'])

            baseline_evaluation = SignalDrivenBacktester(strategy=baseline, **params)
            return SignalDrivenBacktester(strategy=strategy,
                                          benchmark_backtest=baseline_evaluation, **params)

        except NoPriceDataException as e:
            logging.info('Error while fetching price, skipping...')
            return None
        except Exception as e:
            logging.info(f'Error during strategy evaluation: {str(e)}')
            return None

    @property
    def report(self):
        return self._report


class ComparativeReportBuilder:
    """
    Based on a set of backtests and corresponding baseline backtests, builds a comparative report of their performance.
    """

    def __init__(self, backtests, baselines, additional_columns = []):
        """
        Initializes and builds the report dataframe.
        :param backtests: A collection of backtest objects.
        :param baselines: A collection of baselines corresponding to backtests.
        """
        self.backtests = backtests
        self.baselines = baselines
        self.additional_columns = additional_columns
        self._build_dataframe()

    @time_performance
    def _build_dataframe(self):
        evaluation_dicts = [self._create_row_dict(backtest, baseline) for backtest, baseline in zip(self.backtests, self.baselines)]
        output = pd.DataFrame(evaluation_dicts)
        if len(output) == 0:
            logging.warning("No strategies evaluated.")
            return
        # sort results by profit
        # output = output.sort_values(by=['profit_percent'], ascending=False)

        # drop index
        output.reset_index(inplace=True, drop=True)

        # reformat sources to human readable format
        output.source.replace([0, 1, 2], ['Poloniex', 'Bittrex', 'Binance'], inplace=True)

        # save full results
        self.results_df = output

    #from utils import time_performance
    #@time_performance
    def _create_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["evaluation_object"] = evaluation
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        return evaluation_dict


    def get_best_performing_backtest(self):
        sorted_df = self.results_df.sort_values(by=['profit_percent'], ascending=False)
        return sorted_df[self.results_df.num_trades > 0].iloc[0]["evaluation_object"]


    def write_summary(self, output_file):
        writer = pd.ExcelWriter(output_file)
        # filter so that only report columns remain
        #self.results_df['profit_percent'][self.results_df['profit_percent'] == 'N/A'] = np.nan
        self.results_df[backtesting_report_columns]. \
            sort_values(by=['profit_percent'], ascending=False).to_excel(writer, 'Results')

        writer.save()


    def write_comparative_summary(self, summary_path):
        writer = pd.ExcelWriter(summary_path)
        summary = self.results_df.copy()
        # remove empty trades
        summary = summary[summary.num_trades != 0]
        summary.groupby(["strategy", "resample_period"]).describe().to_excel(writer, 'Results')
        writer.save()


    def _filter_df_based_on_currency_pairs(self, df, currency_pairs_to_keep):
        return df[df[["transaction_currency", "counter_currency"]].apply(tuple, 1).isin(currency_pairs_to_keep)]


    def _get_description_stat_values(self, desc_df, field_name, row_name):
        mean = desc_df[[(field_name, 'mean')]].loc[row_name][0]
        std = desc_df[[(field_name, 'std')]].loc[row_name][0]
        min = desc_df[[(field_name, 'min')]].loc[row_name][0]
        max = desc_df[[(field_name, 'max')]].loc[row_name][0]
        return mean, std, min, max


    def _describe_and_write(self, filtered_df, writer, sheet_prefix, formats, additional_columns=[]):
        if filtered_df.empty:
            return

        # group by strategy, evaluate
        by_strategy_df = filtered_df[["source", "strategy", "resample_period", "profit_percent",
                                      "buy_and_hold_profit_percent", "num_trades", "sharpe_ratio"] + additional_columns]\
            .groupby(["source", "strategy", "resample_period"], sort=False).describe(percentiles=[])

        # add outperformance stat
        outperforms = pd.Series(by_strategy_df[('profit_percent', 'mean')]) > \
                      pd.Series(by_strategy_df[('buy_and_hold_profit_percent', 'mean')])

        by_strategy_df["outperforms"] = outperforms

        # reorder columns and write
        by_strategy_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades", "outperforms", "sharpe_ratio"] + additional_columns]\
            .to_excel(writer, f'{sheet_prefix} by strategy', header=False, startrow=4)

        # apply sheet formatting
        sheet = writer.sheets[f'{sheet_prefix} by strategy']
        self._reformat_sheet_grouped_by_strategy(formats, outperforms, sheet, additional_columns)

        # group by coin, evaluate
        by_coin_df = filtered_df[["source", "transaction_currency", "profit_percent",
                                  "buy_and_hold_profit_percent", "num_trades", ]]\
            .groupby(["source", "transaction_currency"]).describe(percentiles=[])

        # reorder columns and write
        # not sorted - commented out
        # by_coin_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades"]].to_excel(writer, f'{sheet_prefix} by coin')

        g = by_coin_df.groupby(level=0, group_keys=False)
        by_coin_sorted_df = g.apply(lambda x: x.sort_values([('profit_percent', 'mean')], ascending=False))
        by_coin_sorted_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades", ]].to_excel(writer, f'{sheet_prefix} by coin',
                                                                                                    header=False, startrow=4)
        # apply sheet formatting
        sheet = writer.sheets[f'{sheet_prefix} by coin']
        self._reformat_sheet_grouped_by_coin(formats, sheet)


    def _reformat_sheet_grouped_by_strategy(self, formats, outperforms, sheet, additional_columns = []):
        # add outperformance percent at the top
        sheet.write('V4', np.mean(outperforms), formats['large_bold_red'])

        # apply this to all header cells and reinit header
        sheet.merge_range('D3:I3', 'Profit percent', formats['header_format'])
        sheet.merge_range('J3:O3', 'Buy and hold profit percent', formats['header_format'])
        sheet.merge_range('P3:U3', 'Number of trades', formats['header_format'])
        sheet.write('V3', 'Outperforms', formats['header_format'])
        sheet.merge_range('W3:AB3', 'Sharpe ratio', formats['header_format'])

        start = 28
        for col in additional_columns:
            end = start+5
            start_col = xlsxwriter.utility.xl_col_to_name(start)
            end_col = xlsxwriter.utility.xl_col_to_name(end)
            sheet.merge_range(f'{start_col}3:{end_col}3', col, formats['header_format'])
            start = end+1

        sheet.write_row('D4', ['number of tests (coin, strategy)',
                               'mean', 'std', 'min', 'median', 'max'] * 3, formats['aux_header_format'])
        sheet.write_row('W4', ['number of tests (coin, strategy)',
                               'mean', 'std', 'min', 'median', 'max'], formats['aux_header_format'])

        sheet.write_row('AC4', ['number of tests (coin, strategy)',
                               'mean', 'std', 'min', 'median', 'max'] * len(additional_columns), formats['aux_header_format'])


        sheet.write_row('A5', ['exchange',
                               'strategy', 'resample period (minutes)'], formats['header_format'])
        # set up column formatting
        sheet.set_column('E:I', None, formats['percent_format'])
        sheet.set_column('J:J', None, formats['gray_bg_format'])
        sheet.set_column('L:O', None, formats['gray_percent_format'])
        sheet.set_column('Q:U', None, formats['number_format'])
        sheet.set_column('E:E', None, formats['bold_red_percent'])
        sheet.set_column('K:K', None, formats['gray_bold_red_percent'])
        sheet.set_column('V:V', 15, formats['gray_percent_format'])
        sheet.set_column('B:C', None, formats['mid_bold'])
        sheet.set_column('A:A', None, formats['top_bold'])
        sheet.set_column('Q:Q', None, formats['bold_red_number'])

        sheet.set_row(0, None, formats['white_bold'])
        sheet.set_row(1, None, formats['white_bold'])

        sheet.write('A1', 'Mean profits, buy and hold profits and trades for all coins and signals in our database, grouped by strategy')


    def _reformat_sheet_grouped_by_coin(self, formats, sheet):

        # apply this to all header cells and reinit header
        sheet.merge_range('C3:H3', 'Profit percent', formats['header_format'])
        sheet.merge_range('I3:N3', 'Buy and hold profit percent', formats['header_format'])
        sheet.merge_range('O3:T3', 'Number of trades', formats['header_format'])
        sheet.write_row('C4', ['number of tests (coin, strategy)',
                               'mean', 'std', 'min', 'median', 'max'] * 3, formats['aux_header_format'])
        sheet.write_row('A5', ['exchange', 'coin'], formats['header_format'])
        # set up column formatting
        sheet.set_column('D:H', None, formats['percent_format'])
        sheet.set_column('I:I', None, formats['gray_bg_format'])
        sheet.set_column('K:N', None, formats['gray_percent_format'])
        sheet.set_column('P:T', None, formats['number_format'])
        sheet.set_column('D:D', None, formats['bold_red_percent'])
        sheet.set_column('J:J', None, formats['gray_bold_red_percent'])
        sheet.set_column('P:P', 15, formats['gray_percent_format'])
        sheet.set_column('B:B', None, formats['mid_bold'])
        sheet.set_column('A:A', None, formats['top_bold'])
        sheet.set_column('P:P', None, formats['bold_red_number'])

        # clear formatting of the first two rows
        sheet.set_row(0, None, formats['white_bold'])
        sheet.set_row(1, None, formats['white_bold'])

        sheet.write('A1', 'Mean profits, buy and hold profits and number of trades for all coins and signals in our database, grouped by coin')


    def _reformat_original_data(self, sheet, formats):
        sheet.write_row('A1', backtesting_report_column_names, formats['header_format'])
        sheet.set_column('A:A', 20)
        sheet.set_column('F:I', None, formats['percent_format'])
        sheet.set_column('M:M', None, formats['percent_format'])


    def _init_worksheet_formats(self, workbook):
        formats = {}
        formats['percent_format'] = workbook.add_format({'num_format': '0.00\%'})
        formats['gray_percent_format'] = workbook.add_format({'num_format': '0.00\%', 'pattern': 1, 'bg_color': '#D9D9D9'})
        formats['gray_bg_format'] = workbook.add_format({'pattern': 1, 'bg_color': '#D9D9D9'})
        formats['number_format'] = workbook.add_format({'num_format': '0.00'})
        formats['bold_red_percent'] = workbook.add_format({'bold': True, 'font_color': 'red', 'num_format': '0.00\%'})
        formats['bold_red_number'] = workbook.add_format({'bold': True, 'font_color': 'red', 'num_format': '0.00'})
        formats['large_bold_red'] = workbook.add_format(
            {'bold': True, 'font_color': 'red', 'font_size': 16, 'num_format': '0.00%',
             'border': 1, 'pattern': 1, 'bg_color': '#BFBFBF', 'align': 'center',
             'valign': 'vcenter'})
        formats['gray_bold_red_percent'] = workbook.add_format({'bold': True, 'font_color': 'red', 'num_format': '0.00\%',
                                                          'pattern': 1, 'bg_color': '#D9D9D9'})
        formats['header_format'] = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#808080',
            'font_color': '#FFFFFF',
            'border': 1,
            'pattern': 1
        })
        formats['aux_header_format'] = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#BFBFBF',
            'font_color': '#FFFFFF',
            'border': 1,
            'pattern': 1
        })
        formats['mid_bold'] = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
        })
        formats['top_bold'] = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'top'
        })
        formats['white_bold'] = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#FFFFFF',
            'pattern': 1
        })

        return formats


    def all_coins_report(self, report_path=None, currency_pairs_to_keep=None, group_strategy_variants=True, writer=None,
                         sheet_prefix='', additional_columns=['num_gains', 'num_losses']):
        df = self.results_df.copy(deep=True)

        # remove strategies without any trades - very important for averaging!
        df = df[df.num_trades > 0]

        if group_strategy_variants:
            # clean up and structure strategy names
            df.loc[df['strategy'].str.contains('rsi_buy'), 'strategy'] = 'RSI variant'
            df.loc[df['strategy'].str.contains('rsi_cumulat'), 'strategy'] = 'RSI cumulative variant'
            df.loc[df['strategy'].str.contains('ann_'), 'strategy'] = 'ANN'
            df.loc[df['strategy'].str.contains('ichi_'), 'strategy'] = 'Ichimoku'

        if report_path is None and writer is None:
            logging.error('Please specify writer or a report path!')
            return

        close_on_exit = False
        if writer is None:
            writer = pd.ExcelWriter(report_path)
            close_on_exit = True

        workbook = writer.book
        formats = self._init_worksheet_formats(workbook)

        if currency_pairs_to_keep is not None:
            df = self._filter_df_based_on_currency_pairs(currency_pairs_to_keep)

        self._describe_and_write(df, writer, f"{sheet_prefix}All", formats, additional_columns)
        top_20_alts = zip(COINMARKETCAP_TOP_20_ALTS, ["BTC"] * len(COINMARKETCAP_TOP_20_ALTS))
        top20 = self._filter_df_based_on_currency_pairs(df, top_20_alts)
        self._describe_and_write(top20, writer, f"{sheet_prefix}Top 20", formats, additional_columns)

        df[backtesting_report_columns].to_excel(writer, f"{sheet_prefix}Original data", index=False, header=False, startrow=1)
        self._reformat_original_data(writer.sheets[f"{sheet_prefix}Original data"], formats)

        if close_on_exit:
            writer.save()
            writer.close()






