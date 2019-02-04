import math


#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random
import logging

from deap import tools, gp
from deap.algorithms import varAnd, varOr
from apps.genetic_algorithms.gp_utils import compress


def eaSimpleCustom(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, genetic_program=None, filter_same_fitnesses_in_hof=True):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """

    # make sure the initial population is compressed
    population = compress_population(population, genetic_program)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    best_in_gens = []

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # compress population
        # print(f'Before compression: {" / ".join(map(str, population))}')
        # population = compress_population(population, genetic_program)
        # print(f'After compression: {" / ".join(map(str, population))}')

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # compress offspring
        offspring = compress_population(offspring, genetic_program)

        hof_fitnesses = set([x.fitness.values[0] for x in halloffame.items])
        filtered_offspring = []

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if not filter_same_fitnesses_in_hof or filter_same_fitnesses_in_hof and fit[0] in hof_fitnesses:
                filtered_offspring.append(ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(filtered_offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        best_in_gens.append(sorted_pop[0])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, best_in_gens

def compress_population(population, genetic_program):
    return [genetic_program.individual_from_string(compress(individual)) for individual in population]

def combined_mutation(individual, expr, pset):
    if random.random() > 0.5:
        return gp.mutInsert(individual, pset)
    else:
        return gp.mutEphemeral(individual, "one")




###### HARM algorithm from Deap, with added compression

######################################
# GP bloat control algorithms        #
######################################

def harm(population, toolbox, cxpb, mutpb, ngen, alpha=0.05, beta = 10, gamma = 0.25, rho = 0.9,
         nbrindsmodel=-1, mincutoff=20, stats=None, halloffame=None, verbose=__debug__, genetic_program=None):
    """Implement bloat control on a GP evolution using HARM-GP, as defined in
    [Gardner2015]. It is implemented in the form of an evolution algorithm
    (similar to :func:`~deap.algorithms.eaSimple`).
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param alpha: The HARM *alpha* parameter.
    :param beta: The HARM *beta* parameter.
    :param gamma: The HARM *gamma* parameter.
    :param rho: The HARM *rho* parameter.
    :param nbrindsmodel: The number of individuals to generate in order to
                            model the natural distribution. -1 is a special
                            value which uses the equation proposed in
                            [Gardner2015] to set the value of this parameter :
                            max(2000, len(population))
    :param mincutoff: The absolute minimum value for the cutoff point. It is
                        used to ensure that HARM does not shrink the population
                        too much at the beginning of the evolution. The default
                        value is usually fine.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. note::
       The recommended values for the HARM-GP parameters are *alpha=0.05*,
       *beta=10*, *gamma=0.25*, *rho=0.9*. However, these parameters can be
       adjusted to perform better on a specific problem (see the relevant
       paper for tuning information). The number of individuals used to
       model the natural distribution and the minimum cutoff point are less
       important, their default value being effective in most cases.
    .. [Gardner2015] M.-A. Gardner, C. Gagne, and M. Parizeau, Controlling
        Code Growth by Dynamically Shaping the Genotype Size Distribution,
        Genetic Programming and Evolvable Machines, 2015,
        DOI 10.1007/s10710-015-9242-8
    """
    def _genpop(n, pickfrom=[], acceptfunc=lambda s: True, producesizes=False):
        # Generate a population of n individuals, using individuals in
        # *pickfrom* if possible, with a *acceptfunc* acceptance function.
        # If *producesizes* is true, also return a list of the produced
        # individuals sizes.
        # This function is used 1) to generate the natural distribution
        # (in this case, pickfrom and acceptfunc should be let at their
        # default values) and 2) to generate the final population, in which
        # case pickfrom should be the natural population previously generated
        # and acceptfunc a function implementing the HARM-GP algorithm.
        producedpop = []
        producedpopsizes = []
        while len(producedpop) < n:
            if len(pickfrom) > 0:
                # If possible, use the already generated
                # individuals (more efficient)
                aspirant = pickfrom.pop()
                if acceptfunc(len(aspirant)):
                    producedpop.append(aspirant)
                    if producesizes:
                        producedpopsizes.append(len(aspirant))
            else:
                opRandom = random.random()
                if opRandom < cxpb:
                    # Crossover
                    aspirant1, aspirant2 = toolbox.mate(*map(toolbox.clone,
                                                             toolbox.select(population, 2)))
                    del aspirant1.fitness.values, aspirant2.fitness.values
                    if acceptfunc(len(aspirant1)):
                        producedpop.append(aspirant1)
                        if producesizes:
                            producedpopsizes.append(len(aspirant1))

                    if len(producedpop) < n and acceptfunc(len(aspirant2)):
                        producedpop.append(aspirant2)
                        if producesizes:
                            producedpopsizes.append(len(aspirant2))
                else:
                    aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                    if opRandom - cxpb < mutpb:
                        # Mutation
                        aspirant = toolbox.mutate(aspirant)[0]
                        del aspirant.fitness.values
                    if acceptfunc(len(aspirant)):
                        producedpop.append(aspirant)
                        if producesizes:
                            producedpopsizes.append(len(aspirant))

        if producesizes:
            return producedpop, producedpopsizes
        else:
            return producedpop

    def halflifefunc(x):
        return x * float(alpha) + beta

    if nbrindsmodel == -1:
        nbrindsmodel = max(2000, len(population))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    best_in_gens = []

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Estimation population natural distribution of sizes
        naturalpop, naturalpopsizes = _genpop(nbrindsmodel, producesizes=True)

        naturalhist = [0] * (max(naturalpopsizes) + 3)
        for indsize in naturalpopsizes:
            # Kernel density estimation application
            naturalhist[indsize] += 0.4
            naturalhist[indsize - 1] += 0.2
            naturalhist[indsize + 1] += 0.2
            naturalhist[indsize + 2] += 0.1
            if indsize - 2 >= 0:
                naturalhist[indsize - 2] += 0.1

        # Normalization
        naturalhist = [val * len(population) / nbrindsmodel for val in naturalhist]

        # Cutoff point selection
        sortednatural = sorted(naturalpop, key=lambda ind: ind.fitness)
        cutoffcandidates = sortednatural[int(len(population) * rho - 1):]
        # Select the cutoff point, with an absolute minimum applied
        # to avoid weird cases in the first generations
        cutoffsize = max(mincutoff, len(min(cutoffcandidates, key=len)))


        # Compute the target distribution
        def targetfunc(x):
            return (gamma * len(population) * math.log(2) /
                    halflifefunc(x)) * math.exp(-math.log(2) *
                                                (x - cutoffsize) / halflifefunc(x))
        targethist = [naturalhist[binidx] if binidx <= cutoffsize else
                      targetfunc(binidx) for binidx in range(len(naturalhist))]

        # Compute the probabilities distribution
        probhist = [t / n if n > 0 else t for n, t in zip(naturalhist, targethist)]

        def probfunc(s):
            return probhist[s] if s < len(probhist) else targetfunc(s)

        def acceptfunc(s):
            return random.random() <= probfunc(s)

        # Generate offspring using the acceptance probabilities
        # previously computed
        offspring = _genpop(len(population), pickfrom=naturalpop,
                            acceptfunc=acceptfunc, producesizes=False)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Compress offspring
        offspring = compress_population(offspring, genetic_program)
        # Replace the current population by the offspring
        population[:] = offspring

        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        best_in_gens.append(sorted_pop[0])

        # Append the current generation statistics to the logbook
        try:
            record = stats.compile(population) if stats else {}
        except:
            pass
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, best_in_gens