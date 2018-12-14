import operator
import os
import logging
import numpy as np
import math
from deap import creator, tools, base
from deap.gp import PrimitiveTree
from apps.backtesting.signals import Signal
from apps.backtesting.strategies import SignalStrategy, Strength, TickerStrategy, StrategyDecision
from apps.genetic_algorithms.chart_plotter import *
from apps.genetic_algorithms.custom_deap_algorithms import combined_mutation, eaSimpleCustom, harm
from apps.backtesting.backtester_ticks import TickDrivenBacktester
from apps.backtesting.tick_provider import PriceDataframeTickProvider
from abc import ABC, abstractmethod
from apps.backtesting.order_generator import OrderGenerator
import dill as pickle
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

np.seterr(divide='ignore', invalid='ignore')
# prevent Sharpe ratio NaN warnings

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

HISTORY_SIZE = 200

class Grammar(ABC):

    def __init__(self, function_provider):
        self.function_provider = function_provider

    @property
    def pset(self):
        return self._pset

    @property
    @abstractmethod
    def name(self):
        pass


class GeneticTickerStrategy(TickerStrategy):
    def __init__(self, tree, gp_object):
        self.tree = tree
        self.gp_object = gp_object
        self.i = 0
        self.func = self.gp_object.toolbox.compile(expr=self.tree)

    def process_ticker(self, ticker_data):
        """
        :param price_data: Pandas row with OHLC data and timestamp.
        :param signals: ITF signals co-occurring with price tick.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """

        return self.get_decision(ticker_data)

    def get_decision(self, ticker_data):

        outcome = self.func([ticker_data.timestamp, ticker_data.transaction_currency, ticker_data.counter_currency])

        decision = None
        if outcome == self.gp_object.function_provider.buy:
            signal = Signal("Genetic", 1, None, 3, 3, ticker_data.close_price, 0, ticker_data.timestamp, None,
                            ticker_data.transaction_currency,
                            ticker_data.counter_currency, ticker_data.source, ticker_data.resample_period)
            decision = StrategyDecision(ticker_data.timestamp, ticker_data.transaction_currency,
                                        ticker_data.counter_currency,
                                        self.source, StrategyDecision.BUY, signal)
        elif outcome == self.gp_object.function_provider.sell:
            signal = Signal("Genetic", -1, None, 3, 3, ticker_data.close_price, 0, ticker_data.timestamp, None,
                            ticker_data.transaction_currency,
                            ticker_data.counter_currency, ticker_data.source, ticker_data.resample_period)
            decision = StrategyDecision(ticker_data.timestamp, ticker_data.transaction_currency,
                                        ticker_data.counter_currency,
                                        ticker_data.source, StrategyDecision.SELL, signal)
        elif not outcome == self.gp_object.function_provider.ignore:
            logging.warning("Invalid outcome encountered")

        if decision is None:
            decision = StrategyDecision(ticker_data.timestamp, outcome=StrategyDecision.IGNORE)

        return decision

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))


class GeneticSignalStrategy(SignalStrategy):
    def __init__(self, tree, data, gp_object, history_size=HISTORY_SIZE):
        self.data = data
        self.transaction_currency = data.transaction_currency
        self.counter_currency = data.counter_currency
        self.resample_period = data.resample_period
        self.strength = Strength.any
        self.source = data.source
        self.start_time = data.start_time
        self.end_time = data.end_time
        self.tree = tree
        self.gp_object = gp_object
        self.history_size = history_size
        self.build_from_gp_tree(tree)

    def get_orders(self, signals, start_cash, start_crypto, source, time_delay=0, slippage=0):
        return SignalStrategy.get_orders(self, self.signals, start_cash, start_crypto, source, time_delay, slippage)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def build_from_gp_tree(self, tree):
        self.signals = []
        func = self.gp_object.toolbox.compile(expr=tree)

        outcomes = []
        for i, row in enumerate(self.data.price_data.itertuples()):
            price = row.close_price
            timestamp = row.Index
            if i < self.history_size:
                outcomes.append("skipped")
                continue
            outcome = func([timestamp])

            trend = None
            if outcome == self.gp_object.buy:
                trend = 1
            elif outcome == self.gp_object.sell:
                trend = -1
            elif not outcome == self.gp_object.ignore:
                logging.warning("Invalid outcome encountered")
            if trend != None:
                signal = Signal("Genetic", trend, self.horizon, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                                self.counter_currency, self.source, self.resample_period)
                self.signals.append(signal)

            outcomes.append(outcome.__name__)
        df = self.data.to_dataframe()
        df['outcomes'] = pd.Series(outcomes, index=df.index)
        self.df_data_and_outcomes = df

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))

    def get_dataframe_with_outcomes(self):
        return self.df_data_and_outcomes


class FitnessFunction(ABC):
    @abstractmethod
    def compute(self, individual, evaluation, genetic_program):
        pass

    @staticmethod
    def construct(fitness_function_name):
        for subclass in FitnessFunction.__subclasses__():
            if subclass._name == fitness_function_name:
                return subclass()
        raise Exception(f"Unknown function {fitness_function_name}!")

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name


class FitnessFunctionV1(FitnessFunction):
    _name = "ff_v1"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return evaluation.profit_percent + (max_len - len(individual)) / float(max_len) * 20 \
               + evaluation.num_sells * 5,


class BenchmarkDiffFitness(FitnessFunction):
    _name = "ff_benchmarkdiff"

    def compute(self, individual, evaluation, genetic_program):
        return evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent,


class BenchmarkDiffTrades(FitnessFunction):
    _name = "ff_benchmarkdiff_trades"

    def compute(self, individual, evaluation, genetic_program):
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent)*evaluation.num_profitable_trades,


class BenchmarkLengthControlFitness(FitnessFunction):
    _name = "ff_benchlenctrl"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) + \
               (max_len - len(individual)) / float(max_len) * 20,

class UltimateEpicFunction(FitnessFunction):
    _name = "ff_ultimateepic"

    def compute(self, individual, evaluation, genetic_program):
        try:
            return (math.exp(1)+math.log(1+math.sqrt(evaluation.num_buys * evaluation.num_sells))) ** \
                   (evaluation.profit_percent / (evaluation.benchmark_backtest.profit_percent + 0.000001)),
        except:
            pass


class BenchmarkLengthControlFitnessV2(FitnessFunction):
    _name = "ff_benchlenctrl_v2"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) * \
               (max_len - len(individual)) / float(max_len),


class BenchmarkLengthControlFitnessV3(FitnessFunction):
    _name = "ff_benchlenctrl_v3"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) * \
               (1+0.1*(max_len - len(individual)) / float(max_len)),

class GainsBenchmarkDiffFitness(FitnessFunction):
    _name = "ff_gainsbenchmarkdiff"

    def compute(self, individual, evaluation, genetic_program):
        return 0 if evaluation.num_buy_sell_pairs == 0 \
                 else evaluation.percent_profitable_trades * (
                    evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent),


class GeneticProgram:
    def __init__(self, data_collection, **kwargs):
        self.data_collection = data_collection
        self.function_provider = kwargs.get('function_provider')
        self.grammar = kwargs.get('grammar')
        self.fitness = kwargs.get('fitness_function', FitnessFunctionV1())
        self.tree_depth = kwargs.get('tree_depth', 3)
        self.combined_fitness_operator = kwargs.get('combined_fitness_operator', min)
        self.premade_individuals = kwargs.get('premade_individuals', [])
        self.order_generator = kwargs.get('order_generator', OrderGenerator.ALTERNATING)

        self.reseed_params = kwargs.get('reseed_params', None)
        self.hof_size = kwargs.get('hof_size', 10)
        self._build_toolbox()


    @property
    def pset(self):
        return self.grammar.pset

    def _build_toolbox(self):

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=self.tree_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self.compute_fitness_over_datasets)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", combined_mutation, min_=0, max_=self.tree_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.register("mutate", combined_mutation, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        self.toolbox = toolbox

    def _generate_population(self, population_size, min_fitness=0, num_good_individuals=1, max_iterations=20):
        run_count = 0
        logging.info('Generating population...')
        saved = []
        while True:
            run_count += 1
            population = self.toolbox.population(n=population_size)
            count = 0
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit[0] >= min_fitness:
                    count += 1
                    saved.append(ind)
            # option 1, by rolling the dice we got a total of num_good_individuals
            if count >= num_good_individuals:
                logging.info(f'Initialized a population with {count} individuals'
                      f' of fitness >= {min_fitness} after {run_count} iterations.')
                population = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
                return population
            # option 2, we found some good individuals in the previous runs, some in this run, and we have enough
            elif len(saved) == num_good_individuals:
                population = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
                population[0:len(saved)] = saved
                return population
            # option 3, max runtime exceeded, return what we have
            if run_count > max_iterations:
                population = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
                population[0:len(saved)] = saved
                logging.info(f'Reached {run_count} iterations trying to generate initial population, '
                             f'continuing ({len(saved)} individuals found and inserted)...')
                return population


    def evolve(self, mating_prob, mutation_prob, population_size,
               num_generations, verbose=True, output_folder=None, run_id=None):
        if self.reseed_params is None or not self.reseed_params['enabled']:
            pop = self.toolbox.population(n=population_size)
        else:
            pop = self._generate_population(population_size, min_fitness=self.reseed_params['min_fitness'],
                                            num_good_individuals=self.reseed_params['num_good_individuals'],
                                            max_iterations=self.reseed_params['max_iterations'])

        # insert premade individuals into the population (if any)
        premade = [self.individual_from_string(code) for code in self.premade_individuals]
        pop[-len(premade):] = premade
        if len(premade) > 0:
            logging.info(f'Inserted {len(premade)} individuals into the initial population.')

        hof = tools.HallOfFame(self.hof_size)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log, best = eaSimpleCustom(pop, self.toolbox, mating_prob, mutation_prob, num_generations, stats=mstats,
                                        halloffame=hof, verbose=True, genetic_program=self)

        #pop, log, best = harm(pop, self.toolbox, mating_prob, mutation_prob, num_generations, stats=mstats,
        #                                halloffame=hof, verbose=True, genetic_program=self)

        if verbose:
            logging.info("Winners in each generation: ")
            for i in range(len(best)):
                logging.info(f"  {i}  {best[i]}")
            logging.info("Hall of fame: ")
            for i in range(len(hof)):
                logging.info(f"  {i}  {hof[i]}")
            draw_tree(hof[0])

        if output_folder != None:  # TODO: clean
            hof_name = self.get_hof_filename(mating_prob, mutation_prob, run_id)
            gen_best_name = self.get_gen_best_filename(mating_prob, mutation_prob, run_id)
            out_path_hof = os.path.join(output_folder, hof_name)
            out_path_gen_best = os.path.join(output_folder, gen_best_name)
            if os.path.exists(out_path_hof) and os.path.exists(out_path_gen_best):
                logging.warning(f"{hof_name} already exists, skipping...")
            else:
                pickle.dump(hof, open(out_path_hof, "wb"))
                pickle.dump(best, open(out_path_gen_best, "wb"))

        return hof, best


    #@time_performance
    def compute_fitness(self, individual, data, super_verbose=False):
        evaluation = self.build_evaluation_object(individual, data)

        if evaluation.num_trades > 1 and super_verbose:
            logging.info(str(individual))
            logging.info(evaluation.get_report())
            draw_tree(individual)

        return self.fitness.compute(individual, evaluation, self)
    
    def compute_fitness_over_datasets(self, individual):
        fitnesses = []
        for i, data in enumerate(self.data_collection):
            fitnesses.append(self.compute_fitness(individual, data)[0])

        return self.combined_fitness_operator(fitnesses),

    def build_evaluation_object(self, individual, data, tick_based=True):
        if not tick_based:
            strategy = GeneticSignalStrategy(individual, data, self,
                                             history_size=self.grammar.longest_function_history_size)
            evaluation = strategy.evaluate(data.transaction_currency, data.counter_currency,
                                           data.start_cash, data.start_crypto,
                                           data.start_time, data.end_time, data.source, 60, verbose=False,
                                           order_generator=self.order_generator)

        else:
            strategy = GeneticTickerStrategy(tree=individual,
                                             transaction_currency=data.transaction_currency,
                                             counter_currency=data.counter_currency,
                                             source=data.source,
                                             resample_period=data.resample_period,
                                             gp_object=self)
            tick_provider = PriceDataframeTickProvider(data.price_data)

            # create a new tick based backtester
            evaluation = TickDrivenBacktester(
                tick_provider=tick_provider,
                strategy=strategy,
                transaction_currency=data.transaction_currency,
                counter_currency=data.counter_currency,
                source=data.source,
                resample_period=data.resample_period,
                start_cash=data.start_cash,
                start_crypto=data.start_crypto,
                start_time=data.start_time,
                end_time=data.end_time,
                benchmark_backtest=data.build_buy_and_hold_benchmark(),
                time_delay=0,
                slippage=0,
                verbose=False,
                order_generator=self.order_generator,
                database=data.database
            )
        return evaluation

    def individual_from_string(self, code):
        return creator.Individual(PrimitiveTree.from_string(code, self.grammar.pset))


    @staticmethod
    def get_gen_best_filename(mating_prob, mutation_prob, run_id):
        gen_best_name = "doge_{}_x-{}_m-{}-gen_best.p".format(run_id, mating_prob, mutation_prob)
        return gen_best_name

    @staticmethod
    def get_hof_filename(mating_prob, mutation_prob, run_id):
        hof_name = "doge_{}_x-{}_m-{}-hof.p".format(run_id, mating_prob, mutation_prob)
        return hof_name

    @staticmethod
    def parse_evolution_filename(filename):
        filename = filename.split("_")
        run = int(filename[1])
        mating_prob = float(filename[2].split("-")[1])
        mutation_prob = float(filename[3].split("-")[1])
        return run, mating_prob, mutation_prob

    def load_evolution_file(self, file_path):
        return pickle.load(open(file_path, "rb"))




