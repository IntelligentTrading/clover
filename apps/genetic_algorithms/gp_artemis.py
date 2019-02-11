import logging
import json
import itertools
import numpy as np
import pandas as pd
from apps.backtesting.data_sources import DB_INTERFACE, Data, NoPriceDataException
from apps.backtesting.utils import datetime_to_timestamp

from artemis.experiments import experiment_root
from artemis.experiments.experiments import clear_all_experiments
from collections import OrderedDict
from apps.genetic_algorithms.genetic_program import GeneticProgram, FitnessFunction
from apps.genetic_algorithms.leaf_functions import TAProviderCollection
from apps.genetic_algorithms.grammar import Grammar
from apps.genetic_algorithms.chart_plotter import *
from apps.backtesting.order_generator import OrderGenerator
from apps.backtesting.config import INF_CASH, INF_CRYPTO
from apps.backtesting.comparative_evaluation import ComparativeEvaluation, ComparativeReportBuilder
#from data_sources import get_currencies_trading_against_counter
from apps.backtesting.backtesting_runs import build_itf_baseline_strategies
from apps.backtesting.utils import time_performance
from functools import partial
#from pathos.multiprocessing import ProcessingPool as Pool
from apps.backtesting.utils import parallel_run
from apps.genetic_algorithms.gp_utils import Period
from apps.genetic_algorithms.leaf_functions import TAProviderCollection


from apps.backtesting.utils import LogDuplicateFilter
dup_filter = LogDuplicateFilter()
logging.getLogger().addFilter(dup_filter)


class ExperimentManager:

    START_CASH = 1000
    START_CRYPTO = 0

    @experiment_root
    def run_evolution(experiment_id, data, function_provider, grammar_version, fitness_function, mating_prob,
                      mutation_prob, population_size, num_generations, premade_individuals, order_generator, tree_depth,
                      reseed_params, hof_size):
        grammar = Grammar.construct(grammar_version, function_provider, ephemeral_suffix=experiment_id)
        genetic_program = GeneticProgram(data_collection=data, function_provider=function_provider,
                                         grammar=grammar, fitness_function=fitness_function,
                                         premade_individuals=premade_individuals, order_generator=order_generator,
                                         tree_depth=tree_depth, reseed_params=reseed_params, hof_size=hof_size)
        hof, best = genetic_program.evolve(mating_prob, mutation_prob, population_size, num_generations, verbose=False)
        return hof, best


    def __init__(self, experiment_container, read_from_file=True, database=DB_INTERFACE, hof_size=10,
                 function_provider=None, rockstars=[], parallel_run=False):
        self.database = database
        self.hof_size = hof_size
        self.rockstars = rockstars
        self._parallel_run = parallel_run

        if read_from_file:
            with open(experiment_container) as f:
                self.experiment_json = json.load(f)
        else:
            self.experiment_json = json.loads(experiment_container)

        if self.experiment_json['order_generator'] == OrderGenerator.POSITION_BASED:
            self.START_CASH = INF_CASH
            self.START_CRYPTO = INF_CRYPTO


        # if function provider is set to none, using cached data
        if function_provider is None:
            # initialize data

            self.training_data = [
                DB_INTERFACE.build_data_object(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                               **dataset) for dataset in self.experiment_json["training_data"]
            ]

            self.validation_data = [Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                      **dataset) for dataset in self.experiment_json["validation_data"]]
            # create function provider objects based on data
            from apps.genetic_algorithms.leaf_functions import TAProvider
            self.function_provider = TAProvider(db_interface=DB_INTERFACE) #Collection(self.training_data + self.validation_data)

        else:
            self.training_data = [Data.to_string(dataset['transaction_currency'],
                                                 dataset['counter_currency'],
                                                 datetime_to_timestamp(dataset['start_time']),
                                                 datetime_to_timestamp(dataset['end_time']))
                                  for dataset in self.experiment_json["training_data"]]


            self.function_provider = function_provider
        # generate and register variants
        self._fill_experiment_db()
        self._register_variants()



    def _fill_experiment_db(self):
        self.experiment_db = ExperimentDB()

        variants = itertools.product(self.experiment_json["mating_probabilities"],
                                     self.experiment_json["mutation_probabilities"],
                                     self.experiment_json["population_sizes"],
                                     self.experiment_json["fitness_functions"])
        for mating_prob, mutation_prob, population_size, fitness_function in variants:
            self.experiment_db.add(
                data=self.training_data,
                function_provider=self.function_provider,
                grammar_version=self.experiment_json["grammar_version"],
                fitness_function=FitnessFunction.construct(fitness_function),
                mating_prob=mating_prob,
                mutation_prob=mutation_prob,
                population_size=population_size,
                num_generations=self.experiment_json["num_generations"],
                premade_individuals=self.experiment_json["premade_individuals"]+self.rockstars,
                order_generator=self.experiment_json["order_generator"],
                tree_depth=self.experiment_json["tree_depth"],
                reseed_params=self.experiment_json["reseed_initial_population"]
            )

    def _register_variants(self, rebuild_grammar=True):
        clear_all_experiments()

        if rebuild_grammar:
            for experiment_name, experiment in self.experiment_db.get_all():
                Grammar.construct(experiment['grammar_version'],
                                  experiment['function_provider'],
                                  experiment['experiment_id'])

        self.run_evolution.variants = OrderedDict() # for some reason, Artemis doesn't clear this

        for experiment_name, experiment in self.experiment_db.get_all():
            self.run_evolution.add_variant(variant_name=experiment_name, hof_size=self.hof_size, **experiment)
        self.variants = self.run_evolution.get_variants()

    def get_variants(self):
        return self.run_evolution.get_variants()

    @staticmethod
    def run_variant(variant, keep_record, display_results, rerun_existing, saved_figure_ext):
        if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
            logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
            return

        return variant.run(keep_record=keep_record, display_results=display_results, saved_figure_ext=saved_figure_ext)

    def run_experiments(self, rerun_existing=False, display_results=True, keep_record=True):
        if self._parallel_run:
            self.run_parallel_experiments(rerun_existing, display_results, keep_record)
        else:
            self.run_sequential_experiments(rerun_existing, display_results, keep_record)

    @time_performance
    def run_parallel_experiments(self, rerun_existing=False, display_results=True, keep_record=True):
        partial_run_func = partial(ExperimentManager.run_variant, keep_record=keep_record, display_results=display_results,
                           rerun_existing=rerun_existing, saved_figure_ext='.fig.png')

        records = parallel_run(partial_run_func, self.variants)
        return records

    @time_performance
    def run_sequential_experiments(self, rerun_existing=False, display_results=True, keep_record=True):
        records = []
        for i, variant in enumerate(self.variants):
            if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
                logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
                continue

            logging.info(f"Running variant {i}")
            record = variant.run(keep_record=keep_record, display_results=display_results, saved_figure_ext='.fig.png')
            records.append(record)
        return records


    def explore_records(self, use_validation_data=True):
        if use_validation_data:
            data = self.validation_data
        else:
            data = self.training_data
        #   variants = run_evolution.get_variants()
        for variant in self.variants:
            # for variant_name in variant_names:
            #   variant = run_evolution.get_variant(variant_name)
            records = variant.get_records()
            latest = variant.get_latest_record(only_completed=True)
            if latest is None:
                logging.warning(f">>> No records found for variant {variant.name}, skipping...")
                continue
            logging.info(f"\n\n===== Exploring experiment variant {variant} =====")
            logging.info("== Variant records:")
            logging.info(records)
            logging.info("== Last result:")
            logging.info(latest)
            hof, best = latest.get_result()
            for individual in hof:
                print(individual)
            best_individual = hof[0]
            evaluation = self._build_evaluation_object(best_individual, variant, data)
            evaluation.plot_price_and_orders()
            draw_tree(best_individual)

            logging.info("== Experiment output log:")
            logging.info(latest.get_log())


    def print_best_individual_statistics(self, variant, datasets):
        best_individuals = self.get_best_performing_individuals_and_dataset_performance(variant, datasets)
        best_individual, evaluations = best_individuals[0]
        print(f"Best performing individual across datasets in variant {variant}:\n")
        for evaluation in evaluations:
            self._print_individual_info(best_individual, evaluation)


    def get_best_performing_individuals_and_dataset_performance(self, variant, datasets, top_n=None):
        """
        Finds the best performing individual in an experiment variant, across all datasets.
        :param variant: experiment variant
        :param datasets: a collection of Data objects
        :return: individual (DEAP object) and its corresponding evaluations across datasets
                 (a list of Evaluation objects)
        """
        latest = variant.get_latest_record(only_completed=True)
        if latest is None:
            logging.warning(f">>> No records found for variant {variant.name}, skipping...")
            return
        hof, best = latest.get_result()

        # look at best of all time and best in every generation
        individuals = hof.items + best
        individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

        result = []
        if top_n is None:
            top_n = len(individuals)  # just take all of them
        for i in range(top_n):
            best_individual = individuals[i]
            evaluations = []
            for data in datasets:
                evaluation = self._build_evaluation_object(best_individual, variant, data)
                evaluations.append(evaluation)
            result.append((best_individual, evaluations))
        return result


    def get_best_performing_across_variants_and_datasets(self, datasets, sort_by=["mean_profit"], top_n_per_variant=None,
                                                         remove_duplicates=True):
        """
        Returns a list of best performing individuals, top_n_per_variant per experiment variant.
        :return:
        """
        df = pd.DataFrame(columns=["experiment_name", "doge", "fitness_function", "fitness_value",
                                   "mean_profit", "std_profit", "max_profit", "min_profit", "all_profits",
                                   "benchmark_profits", "differences", "variant",
                                   "evaluations", "individual"])
        for variant in self.get_variants():
            best_individuals = self.get_best_performing_individuals_and_dataset_performance(variant, datasets)

            if best_individuals is None:
                break

            for i, (best_individual, evaluations) in enumerate(best_individuals):
                if remove_duplicates and str(best_individual) in df.doge.values:
                    continue

                if top_n_per_variant is not None and i > top_n_per_variant: # already got enough from this variant
                    break

                profits = [evaluation.profit_percent for evaluation in evaluations]
                benchmark_profits = [evaluation.benchmark_backtest.profit_percent for evaluation in evaluations]
                differences = [x1-x2 for (x1, x2) in zip(profits, benchmark_profits)]
                df = df.append({"experiment_name": str(variant.name),
                           "doge": str(best_individual),
                           "fitness_function": self.get_db_record(variant)["fitness_function"]._name,
                           "fitness_value" : self._get_fitness(best_individual, variant, datasets),
                           "mean_profit": np.mean(profits),
                           "std_profit": np.std(profits),
                           "max_profit": np.max(profits),
                           "min_profit": np.min(profits),
                           "all_profits": ", ".join(map(str,profits)),
                           "benchmark_profits": ", ".join(map(str, benchmark_profits)),
                           "differences": ", ".join(map(str, differences)),
                           "variant" : variant,
                           "evaluations": evaluations,
                           "individual": best_individual}, ignore_index=True)

        df = df.sort_values(by=sort_by, ascending=False)
        return df


    def produce_report_by_periods(self, top_n, out_filename, periods=None, resample_periods=[60], counter_currencies=None,
                       sources=[0], tickers=None, start_cash=1000, start_crypto=0):

        if tickers is None:
            tickers = ComparativeEvaluation.build_tickers(counter_currencies, sources)

        if periods is None:
            periods = {
                Period('2018/03/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC', 'Mar 2018'),
                Period('2018/04/01 00:00:00 UTC', '2018/04/30 23:59:59 UTC', 'Apr 2018'),
                Period('2018/05/01 00:00:00 UTC', '2018/05/31 23:59:59 UTC', 'May 2018'),
                Period('2018/06/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC', 'Jun 2018'),
                Period('2018/07/01 00:00:00 UTC', '2018/07/31 23:59:59 UTC', 'Jul 2018'),
                Period('2018/08/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC', 'Aug 2018'),
                Period('2018/01/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC', 'Q1 2018'),
                Period('2018/04/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC', 'Q2 2018'),
                Period('2018/06/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC', '678 2018'),
            }

        writer = pd.ExcelWriter(out_filename)

        for period in periods:
            self.produce_report(top_n, out_filename, period.start_time, period.end_time, resample_periods,
                           counter_currencies,
                           sources, tickers, start_cash, start_crypto, writer, sheet_prefix=f"({period.name}) ")

        writer.save()
        writer.close()


    @time_performance
    def produce_report(self, top_n, out_filename, start_time, end_time, resample_periods=[60], counter_currencies=None,
                       sources=[0], tickers=None, start_cash=1000, start_crypto=0, writer=None, sheet_prefix=None):


        ann_rsi_strategies, basic_strategies, vbi_strategies = build_itf_baseline_strategies()
        strategies = ann_rsi_strategies + basic_strategies + vbi_strategies

        logging.info('Evaluating GP strategies...')

        genetic_backtests, genetic_baselines, _ = self._get_genetic_backtests(top_n, start_time, end_time,
                                                                           resample_periods, counter_currencies,
                                                                           sources, tickers, start_cash, start_crypto)

        logging.info("Evaluating baseline strategies...")

        comp = ComparativeEvaluation(strategies, start_cash=start_cash, start_crypto=start_crypto,
                                     start_time=start_time, end_time=end_time, resample_periods=resample_periods,
                                     counter_currencies=counter_currencies, sources=sources, debug=False, tickers=tickers)

        report = ComparativeReportBuilder(comp.backtests + genetic_backtests, comp.baselines + genetic_baselines)
        if writer is None:
            report.all_coins_report(out_filename, group_strategy_variants=False)
        else:
            report.all_coins_report(writer=writer, sheet_prefix=sheet_prefix,
                                               group_strategy_variants=False)

    def _get_genetic_backtests(self, top_n, start_time, end_time, resample_periods, counter_currencies, sources,
                               tickers, start_cash, start_crypto, filter_field_name=None, filter_field_value=None):

        performance_df = self.get_best_performing_across_variants_and_datasets(self.training_data)

        # if some filtering is set up, filter the obtained result accordingly
        if filter_field_name is not None:
            performance_df = performance_df[performance_df[filter_field_name] == filter_field_value]

        genetic_strategies = []
        genetic_strategy_variants = []

        # we will evaluate the performance of top n individuals
        for i in range(top_n):
            individual = performance_df.iloc[i].individual
            variant = performance_df.iloc[i].variant
            genetic_strategies.append(individual)
            genetic_strategy_variants.append(variant)

        genetic_backtests, genetic_baselines = self._run_genetic_backtests(genetic_strategies, genetic_strategy_variants, tickers, sources,
                                           resample_periods, start_cash, start_crypto, start_time, end_time,
                                           counter_currencies)
        return genetic_backtests, genetic_baselines, genetic_strategy_variants

    def _run_genetic_backtests(self, genetic_strategies, genetic_strategy_variants, tickers, sources, resample_periods,
                               start_cash, start_crypto, start_time, end_time, counter_currencies):

        genetic_backtests = []
        genetic_baselines = []
        # prepare tickers
        if tickers is None:
            tickers = ComparativeEvaluation.build_tickers(counter_currencies, sources)
        data_collection = []
        for (ticker, resample_period) in itertools.product(tickers, resample_periods):
            try:
                data = Data(
                    start_time=start_time,
                    end_time=end_time,
                    transaction_currency=ticker.transaction_currency,
                    counter_currency=ticker.counter_currency,
                    resample_period=resample_period,
                    source=ticker.source,
                    start_cash=start_cash,
                    start_crypto=start_crypto
                )
                data_collection.append(data)
            except Exception as e:  # in case of missing data
                logging.error(
                    f'Unable to load data for {ticker.transaction_currency}-{ticker.counter_currency}, skipping...')
                logging.error(str(e))
        # build a new TAProvider based on the loaded data
        ta_provider = TAProviderCollection(data_collection)
        param_list = []
        # do genetic program backtesting on the new data
        for data in data_collection:
            for individual, variant in zip(genetic_strategies, genetic_strategy_variants):
                param_list.append({'individual': individual,
                                   'db_record': self.get_db_record(variant),
                                   'data': data,
                                   'function_provider': ta_provider}

                                  )
        """
                        with Pool(POOL_SIZE) as pool:
                            backtests = pool.map(self._build_evaluation_object_parallel, param_list)
                            pool.close()
                            pool.join()
                            logging.info("Parallel processing finished.")
                        """
        backtests = map(self._build_evaluation_object_parallel, param_list)
        for evaluation in backtests:
            if evaluation is None:
                continue
            genetic_backtests.append(evaluation)
            genetic_baselines.append(evaluation.benchmark_backtest)
        return genetic_backtests, genetic_baselines

    def get_performance_df_for_dataset_and_variant(self, variant, data):
        """
        Produces a performance dataframe for the given training set and experiment variant.
        :param variant: experiment variant (get a list of all by invoking get_variants() on this object)
        :param data: a Data object, part of the training_data collection
        :return: a dataframe showing performance
        """
        performance_rows = []
        latest = variant.get_latest_record(only_completed=True)
        if latest is None:
            logging.warning(f">>> No records found for variant {variant.name}, skipping...")
            return
        hof, best = latest.get_result()

        print('HoF::::::')
        for item in hof.items:
            print(f'    {str(item)[:10]} {item.fitness.values[0]}')

        print('GenBest::::::')
        for item in best:
            print(f'    {str(item)[:10]} {item.fitness.values[0]}')

        for rank, individual in enumerate(hof):
            row = self.generate_performance_df_row(data, individual, variant, rank)
            performance_rows.append(row)

        performance_info = pd.DataFrame(performance_rows)
        performance_info = performance_info.sort_values(by=['profit_percent'], ascending=False)
        return performance_info

    def generate_performance_df_row(self, data, individual, variant, rank=None):
        evaluation = self._build_evaluation_object(individual, variant, data)
        row = evaluation.to_primitive_types_dictionary()
        row['experiment_id'] = variant.name
        row['hof_ranking'] = rank
        row["individual"] = individual
        row["individual_str"] = str(individual)
        row["evaluation"] = evaluation
        return row

    def evaluate_individual_on_data_collection(self, individual, variant, data_collection):
        performance_df = pd.DataFrame()
        for data in data_collection:
            row = self.generate_performance_df_row(data, individual, variant)
            performance_df = performance_df.append(row, ignore_index=True)
        return performance_df

    def get_performance_dfs_for_variant(self, variant):
        """
        Gets performance dataframes for an experiment variant, one per training subset.
        :param variant: Experiment variant
        :return: a list of performance dataframes
        """
        performance_dfs = []
        for data in self.training_data:
            performance_dfs.append(self.get_performance_df_for_dataset_and_variant(variant, data))
        return performance_dfs

    def get_joined_performance_dfs_over_all_variants(self, verbose=True):
        """
        Produces a list of N dataframes, where N is the number of training sets in the training collection.
        Each dataframe shows profits on the corresponding training set across all experiment variants.
        :return: a list of dataframes
        """
        joined_dfs = []

        for data in self.training_data:
            performance_info = pd.DataFrame()
            for variant in self.variants:
                df = self.get_performance_df_for_dataset_and_variant(variant, data)
                performance_info = performance_info.append(df)

            performance_info = performance_info.sort_values(by=['profit_percent'], ascending=False)
            if verbose:
                self.performance_df_row_info(performance_info.iloc[0], data)
            joined_dfs.append(performance_info)
        return joined_dfs

    def performance_df_row_info(self, performance_df_row, data=None):
        print(f'Experiment id: {performance_df_row.experiment_id}\n')
        individual = performance_df_row.individual
        evaluation = performance_df_row.evaluation
        self._print_individual_info(individual, evaluation, data)
        return individual

    def _print_individual_info(self, individual, evaluation, data=None):
        evaluation.plot_price_and_orders()
        if not data is None:
            data.plot(evaluation.orders, str(individual))
        print(f'String representation:\n{str(individual)}\n')
        if in_notebook():
            g = self.get_graph(individual)
            from IPython.display import display
            display(g)
        #g.view()
        # draw_tree(individual)
        # try:
        #    networkx_graph(individual)
        # except:
        #    logging.warning("Failed to plot networkx graph (not installed?), skipping...")
        print(f'Backtesting report:\n {evaluation.get_report()}\n')
        print(f'Benchmark backtesting report:\n {evaluation.benchmark_backtest.get_report()}\n')

    def browse_variants(self):
        self.run_evolution.browse()

    def get_db_record(self, variant):
        return self.experiment_db[variant.name[len("run_evolution."):]]

    def get_db_record_from_experiment_id(self, experiment_id):
        return self.experiment_db[experiment_id[len("run_evolution."):]]

    def _build_genetic_program(self, variant, data, function_provider=None):
        if function_provider is None:
            function_provider = self.function_provider
        db_record = self.get_db_record(variant)

        return self.build_genetic_program(data, function_provider, db_record)

    @staticmethod
    def build_genetic_program(data, function_provider, db_record):
        grammar = Grammar.construct(
            grammar_name=db_record['grammar_version'],
            function_provider=function_provider,
            ephemeral_suffix=db_record['experiment_id']
        )
        genetic_program = GeneticProgram(
            data_collection=data,
            function_provider=function_provider,
            grammar=grammar,
            fitness_function=db_record['fitness_function'],
            tree_depth=db_record['tree_depth'],
            order_generator=db_record['order_generator']

        )
        return genetic_program


    def _build_evaluation_object(self, individual, variant, data, function_provider=None):

        genetic_program = self._build_genetic_program(variant, data, function_provider)
        if function_provider is not None: # rebuild individual with the new data
            individual = genetic_program.individual_from_string(str(individual))
        return genetic_program.build_evaluation_object(individual, data)

    @staticmethod
    def _build_evaluation_object_parallel(params):
        try:
            genetic_program = ExperimentManager.build_genetic_program(params['data'], params['function_provider'], params['db_record'])
            individual = genetic_program.individual_from_string(str(params['individual']))
            return genetic_program.build_evaluation_object(individual, params['data'])
        except NoPriceDataException:
            return None

    def _get_fitness(self, individual, variant, data):
        genetic_program = self._build_genetic_program(variant, data)
        return genetic_program.compute_fitness_over_datasets(individual)[0]

    def plot_data(self):
        self.training_data[0].plot()

    def get_graph(self, individual):
        return get_dot_graph(individual)

    def analyze_performance_in_period(self, period, training_tickers, top_n, out_filename=None, resample_periods=[60],
                                      sources=[0], start_cash=1000, start_crypto=0, additional_fields={}, prefix=""):
        start_time, end_time = period.start_time, period.end_time
        results = {}
        for source, resample_period, fitness_function_name in itertools.product(sources, resample_periods, self.experiment_db.registered_fitness_functions()):

            self._fill_strategy_performance_dict(results, f"all_coins_trading_against_USDT", top_n, start_time,
                                                 end_time, resample_period, source, start_cash, start_crypto,
                                                 tickers=None, counter_currencies=['USDT'], fitness_function_name=fitness_function_name, prefix=prefix)
            self._fill_strategy_performance_dict(results, f"all_coins_trading_against_BTC", top_n, start_time,
                                                 end_time, resample_period, source, start_cash, start_crypto,
                                                 tickers=None, counter_currencies=['BTC'], fitness_function_name=fitness_function_name, prefix=prefix)
            self._fill_strategy_performance_dict(results, f"all_training_coins", top_n, start_time,
                                                 end_time, resample_period, source, start_cash, start_crypto,
                                                 tickers=training_tickers, counter_currencies=None, fitness_function_name=fitness_function_name, prefix=prefix)
            self._fill_strategy_performance_dict(results, f"training_coins_trading_against_BTC", top_n, start_time,
                                                 end_time, resample_period, source, start_cash, start_crypto,
                                                 tickers=[ticker for ticker in training_tickers if
                                                          ticker.counter_currency == 'BTC'],
                                                 counter_currencies=None, fitness_function_name=fitness_function_name, prefix=prefix)
            

            self._fill_strategy_performance_dict(results, f"training_coins_trading_against_USDT", top_n, start_time,
                                                 end_time, resample_period, source, start_cash, start_crypto,
                                                 tickers=[ticker for ticker in training_tickers if
                                                          ticker.counter_currency == 'USDT'],
                                                 counter_currencies=None, fitness_function_name=fitness_function_name, prefix=prefix)


        # unroll data into rows and add to a DataFrame
        row_dicts = []
        for key in results:
            item = results[key]
            item['source'] = key[0]
            item['resample_period'] = key[1]
            item['strategy'] = key[2]
            item['fitness_function'] = key[3]
            item['db_record'] = key[4]
            row_dicts.append(item)
            for field in additional_fields: # stuff like experiment name, etc, to add to dataframe
                item[field] = additional_fields[field]

        df = pd.DataFrame(row_dicts)
        df = df[list(additional_fields.keys()) + ['source', 'resample_period', 'strategy',
                                                  'fitness_function', 'db_record',
                                                  f'{prefix}all_coins_trading_against_BTC',
                                                  f'{prefix}baseline_all_coins_trading_against_BTC',
                                                  f'{prefix}all_coins_trading_against_USDT',
                                                  f'{prefix}baseline_all_coins_trading_against_USDT',
                                                  f'{prefix}all_training_coins',
                                                  f'{prefix}baseline_all_training_coins',
                                                  f'{prefix}training_coins_trading_against_BTC',
                                                  f'{prefix}baseline_training_coins_trading_against_BTC',
                                                  f'{prefix}training_coins_trading_against_USDT',
                                                  f'{prefix}baseline_training_coins_trading_against_USDT']]

        return df

    def build_training_and_validation_dataframe(self, training_period, validation_period, training_tickers, top_n,
                                                resample_periods=[60], sources=[0], start_cash=1000, start_crypto=0,
                                                additional_fields={}, out_filename=None):
        additional_fields = dict(additional_fields)
        additional_fields['training_period'] = str(training_period)
        additional_fields['validation_period'] = str(validation_period)
        training_df = self.analyze_performance_in_period(training_period, training_tickers, top_n, None,
                                                         resample_periods, sources, start_cash, start_crypto,
                                                         additional_fields=additional_fields, prefix="training_")
        validation_df = self.analyze_performance_in_period(validation_period, training_tickers, top_n, None,
                                                           resample_periods, sources, start_cash, start_crypto,
                                                           additional_fields=additional_fields, prefix="validation_")
        #df = pd.concat([training_df, validation_df], axis=1)
        df = pd.merge(training_df, validation_df, on=['training_period', 'validation_period',
                                                               'source', 'resample_period', 'strategy', 'fitness_function', 'grammar'])

        if out_filename is not None:
            writer = pd.ExcelWriter(out_filename)
            df.to_excel(writer, 'Sheet1')
            writer.save()
        return df

    def _fill_strategy_performance_dict(self, results_dict, field_name, top_n, start_time, end_time, resample_period, source,
                                        start_cash, start_crypto, tickers, counter_currencies, fitness_function_name, prefix=''):

        genetic_backtests, genetic_baselines, genetic_strategy_variants = self._get_genetic_backtests(
            top_n=top_n,
            start_time=start_time,
            end_time=end_time,
            resample_periods=[resample_period],
            counter_currencies=counter_currencies,
            sources=[source],
            tickers=tickers,
            start_cash=start_cash,
            start_crypto=start_crypto,
            filter_field_name='fitness_function',
            filter_field_value=fitness_function_name
        )
        backtest_results, baseline_results = self._get_evaluation_performance_arrays(genetic_backtests,
                                                                                     genetic_baselines)
        for i, strategy in enumerate(backtest_results):
            key = (source, resample_period, strategy, fitness_function_name, str(self.get_db_record(genetic_strategy_variants[i])))
            if not key in results_dict:
                results_dict[key] = {}
            results_dict[key][f'{prefix}{field_name}'] = np.mean(backtest_results[strategy])
            results_dict[key][f'{prefix}baseline_{field_name}'] = np.mean(baseline_results[strategy])


    def _get_evaluation_performance_arrays(self, backtests, baselines):
        backtest_results = OrderedDict()
        baseline_results = OrderedDict()
        for evaluation, baseline in zip(backtests, baselines):
            if not evaluation._strategy.get_short_summary() in backtest_results:
                backtest_results[evaluation._strategy.get_short_summary()] = [evaluation.profit_percent]
                baseline_results[evaluation._strategy.get_short_summary()] = [baseline.profit_percent]
            else:
                backtest_results[evaluation._strategy.get_short_summary()].append(evaluation.profit_percent)
                baseline_results[evaluation._strategy.get_short_summary()].append(baseline.profit_percent)

        return backtest_results, baseline_results


    @staticmethod
    def resurrect_doge(experiment_json, individual_str, function_provider):
        experiment_json = json.loads(experiment_json)
        grammar = Grammar.construct(
            grammar_name=experiment_json['grammar_version'],
            function_provider=function_provider,
            ephemeral_suffix=experiment_json['grammar_version']
        )
        genetic_program = GeneticProgram(
            data_collection=None,
            function_provider=function_provider,
            grammar=grammar,
            fitness_function=experiment_json['fitness_functions'][0],
            tree_depth=experiment_json['tree_depth'],
            order_generator=experiment_json['order_generator']

        )
        return genetic_program.individual_from_string(individual_str), genetic_program



class ExperimentDB:

    def __init__(self):
        self._experiments = {}
        self._num_records = 0
        self._fitness_functions = set()

    def add(self, data, function_provider, grammar_version, fitness_function,
            mating_prob, mutation_prob, population_size, num_generations, premade_individuals,
            order_generator, tree_depth, reseed_params):
        entry = {}
        entry['data'] = data
        entry['function_provider'] = function_provider
        entry['grammar_version'] = grammar_version
        entry['fitness_function'] = fitness_function
        entry['mating_prob'] = mating_prob
        entry['mutation_prob'] = mutation_prob
        entry['population_size'] = population_size
        entry['num_generations'] = num_generations
        entry['premade_individuals'] = premade_individuals
        entry['order_generator'] = order_generator
        entry['tree_depth'] = tree_depth
        entry['reseed_params'] = reseed_params
        entry['experiment_id'] = self._num_records
        name = self.build_experiment_id(**entry)
        self._experiments[name] = entry
        self._num_records += 1
        self._fitness_functions.add(fitness_function.name)

    def get_all(self):
        for k, v in self._experiments.items():
            yield k, v

    def build_experiment_id(self, **kwargs):
        return f"d_{'-'.join(map(str,kwargs['data']))};" \
               f"{kwargs['grammar_version']};" \
               f"{kwargs['fitness_function']};" \
               f"x_{kwargs['mating_prob']};" \
               f"m_{kwargs['mutation_prob']};" \
               f"n_{kwargs['population_size']};" \
               f"gen_{kwargs['num_generations']};" \
               f"td_{kwargs['tree_depth']};" \
               f"{'p' if kwargs['order_generator'] == 'position_based' else 'a'};" \
               f"{'rs' if kwargs['reseed_params']['enabled'] == True else 'nrs'}"

    #          f"provider_{kwargs['function_provider']};" \

    def registered_fitness_functions(self):
        return self._fitness_functions

    def __getitem__(self, key):
        return self._experiments[key]



####################################
#
# Artemis workflow:
#
# 1. Register all experiment variants
# 2. Run experiments (if not already done)
# 3. Browse through experiment records
# Need to run the code that registers the variants before browsing through experiment records, otherwise the browsing
# won't work!
#
####################################

