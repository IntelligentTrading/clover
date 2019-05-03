from apps.backtesting.comparative_evaluation import *
from apps.backtesting.strategies import ANNAnomalyStrategy
import numpy as np
import datetime
from apps.backtesting.utils import time_performance

@time_performance
def best_performing_signals_of_the_period(start_time=None, end_time=None, additional_strategies=[],
                                          best_performing_filename=None, full_report_filename=None,
                                          group_strategy_variants=False):
    if start_time is None or end_time is None:
        start_time = datetime.datetime(2018, 10, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        end_time = datetime.datetime(2018, 11, 1, 23, 59, tzinfo=datetime.timezone.utc).timestamp()

    if best_performing_filename is None:
        best_performing_filename = f"best_performing_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx"

    if full_report_filename is None:
        full_report_filename = f"full_report_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx"

    ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies = build_itf_baseline_strategies()
    strategies = basic_strategies + vbi_strategies + ann_rsi_strategies + ann_anomaly_strategies + additional_strategies

    comparison = ComparativeEvaluation(strategy_set=ann_anomaly_strategies, start_cash=1, start_crypto=0, start_time=start_time,
                                       end_time=end_time, resample_periods=[60, 240, 1440], counter_currencies=["BTC"],
                                       sources=[0, 1, 2], output_file=best_performing_filename, debug=False, parallelize=False)

    comparison.report.all_coins_report(full_report_filename, group_strategy_variants=group_strategy_variants)


def build_itf_baseline_strategies():
    basic_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['rsi_buy_3', 'rsi_buy_2', 'rsi_cumulat_buy_2', 'rsi_cumulat_buy_3', 'ichi_kumo_up',
                     'ann_simple_bull'],
        sell_signals=['rsi_sell_3', 'rsi_sell_2', 'rsi_cumulat_sell_2', 'rsi_cumulat_sell_3', 'ichi_kumo_down',
                      'ann_simple_bear'],
        num_buy=2,
        num_sell=2,
        signal_combination_mode=SignalCombinationMode.SAME_TYPE)
    vbi_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['vbi_buy'],
        sell_signals=['rsi_sell_1', 'rsi_sell_2', 'rsi_sell_3'],
        num_buy=1,
        num_sell=1,
        signal_combination_mode=SignalCombinationMode.ANY
    )
    ann_rsi_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['rsi_buy_1', 'rsi_buy_2', 'rsi_buy_3'],
        sell_signals=["ann_simple_bear"],
        num_buy=1,
        num_sell=1,
        signal_combination_mode=SignalCombinationMode.ANY
    )

    # ANN anomaly strategies
    comparative_signals = ['RSI', 'RSI_Cumulative', 'ANN_Simple', 'VBI', 'kumo_breakout']
    candle_periods = [0, 1, 3, 5]
    ann_anomaly_strategies = [ANNAnomalyStrategy(confirmation_signal, max_delta_period)
                              for confirmation_signal, max_delta_period in itertools.product(comparative_signals, candle_periods)]

    return ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies


def in_depth_signal_comparison(out_path, additional_strategies = []):
    ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies = build_itf_baseline_strategies()
    strategies = basic_strategies + vbi_strategies + ann_rsi_strategies + ann_anomaly_strategies + additional_strategies

    periods = {
        'Mar 2018': ('2018/03/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC'),
        'Apr 2018': ('2018/04/01 00:00:00 UTC', '2018/04/30 23:59:59 UTC'),
        'May 2018': ('2018/05/01 00:00:00 UTC', '2018/05/31 23:59:59 UTC'),
        'Jun 2018': ('2018/06/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC'),
        'Jul 2018': ('2018/07/01 00:00:00 UTC', '2018/07/31 23:59:59 UTC'),
        'Aug 2018': ('2018/08/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC'),
        'Sep 2018': ('2018/09/01 00:00:00 UTC', '2018/09/30 23:59:59 UTC'),
        'Oct 2018': ('2018/10/01 00:00:00 UTC', '2018/10/31 23:59:59 UTC'),
        'Q1 2018': ('2018/01/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC'),
        'Q2 2018': ('2018/04/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC'),
        '678 2018': ('2018/06/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC'),
    }

    writer = pd.ExcelWriter(out_path)

    from dateutil import parser
    for period in periods:
        start_time = parser.parse(periods[period][0]).timestamp()
        end_time = parser.parse(periods[period][1]).timestamp()

        comparison = ComparativeEvaluation(strategy_set=strategies, start_cash=1, start_crypto=0, start_time=start_time,
                                           end_time=end_time, resample_periods=[60, 240, 1440],
                                           counter_currencies=["BTC"], sources=[0, 1, 2],
                                           output_file=f"best_performing_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx",
                                           debug=False)

        comparison.report.all_coins_report(writer=writer, sheet_prefix=f'({period}) ', group_strategy_variants=False)

    writer.save()
    writer.close()

