import datetime
import os

import scipy.stats

import networkx.utils
from networkx.utils import powerlaw_sequence
import matplotlib.pyplot as plt
import pandas as pd

# from scipy.interpolate import make_interp_spline, BSpline


from mwu import *

curr_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M")
types = ['config', 'erdos', 'sw', 'ba']
types_map = {'config': 'configuration model', 'erdos': 'ER random graph', 'sw': 'small world',
             'ba': 'preferential attachment'}


# import pandas as pd


def sample_thetas(lower, upper, m, d, hack=False):  # fix me
    if not hack:
        return (upper - lower) * np.random.random(size=(m, d)) + lower
    return np.linspace(lower, upper, int(m / d))


def generate_features(lower, upper, num_edges, d, dis_type='uniform'):
    if dis_type == 'uniform':
        return (upper - lower) * np.random.random(size=(num_edges, d)) + lower
    elif dis_type == 'normal':
        sigma = 0.1
        return scipy.stats.truncnorm.rvs(lower / sigma, upper / sigma, loc=0, scale=sigma, size=(num_edges, d))
    elif dis_type == 'hypercube':
        return (upper - lower) * np.random.binomial(1, 0.5, size=(num_edges, d)) + lower
    else:
        raise EnvironmentError


def mwu(functions, iters, k, nu=0.5, num_runs=1):
    # mwu_scores = np.array([mwu_runner.run(num_iters=iters, nu=nu) for _ in xrange(num_runs)])
    # return mwu_scores  # todo maybe mean

    return


def create_master_graph(n, type='erdos'):
    if type == 'config':
        z = networkx.utils.create_degree_sequence(n, powerlaw_sequence)
        G = nx.configuration_model(z)
        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(G.selfloop_edges())  # remove self-loops
    elif type == 'erdos':
        G = nx.erdos_renyi_graph(n, 3. / n)
    elif type == 'sw':
        G = nx.watts_strogatz_graph(n, 5, 1. / n)
    elif type == 'ba':
        G = nx.barabasi_albert_graph(n, 2)
    else:
        raise EnvironmentError
    return G


def create_influence_functions(master_graph, eps_grid, features):
    return [InfluenceFunction(master_graph, eps_grid[j, :], features) for j in xrange(eps_grid.shape[0])]


def get_wei_seed(master_graph, features, k, n, samples_per_function):
    wei_method = InfluenceFunction(master_graph, thetas=0, features=features)
    wei_method.calc_wei_probabilities()
    graphs_max = wei_method.fetch_graphs(samples_per_function, wei='max')
    coverage_dict_max = MWU.coverage_problem_builder(graphs_max)
    seed_max = MWU.coverage_greedy_solver(coverage_dict_max, k, n, samples_per_function)

    graphs_min = wei_method.fetch_graphs(samples_per_function, wei='min')
    coverage_dict_min = MWU.coverage_problem_builder(graphs_min)
    seed_min = MWU.coverage_greedy_solver(coverage_dict_min, k, n, samples_per_function)

    graphs_min = wei_method.fetch_graphs(samples_per_function, wei='min')

    wei_min_influence = np.array([len(MWU.find_reachable_nodes(graph, set(seed_min)))
                                  for graph in graphs_min]).mean()
    wei_max_influence = np.array([len(MWU.find_reachable_nodes(graph, set(seed_min)))
                                  for graph in graphs_max]).mean()
    wei_seed = seed_min
    if wei_max_influence > wei_min_influence:
        wei_seed = seed_max
    return wei_seed


def rand_over_greedies(influence_functions, samples_per_function, k, n):
    # seed_set = []
    # for influence_function in influence_functions:
    #     graphs = influence_function.fetch_graphs(samples_per_function)
    #     coverage_dict = MWU.coverage_problem_builder(graphs)
    #     seed_set.append(MWU.coverage_greedy_solver(coverage_dict, k, n, len(graphs)))
    # return seed_set
    chosen_one = int(np.random.choice(len(influence_functions), 1))
    graphs = influence_functions[chosen_one].fetch_graphs(samples_per_function)
    coverage_dict = MWU.coverage_problem_builder(graphs)
    return MWU.coverage_greedy_solver(coverage_dict, k, n, len(graphs))


def ex1():
    """
    Plot the value of the max-min vs greedy/random/wei and best if possible for the graph as function of k.
    """

    curr_exp_dir = os.path.join('ex1', curr_dir)
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)

    d = 5
    B = 1
    n = 500
    # n = 30

    m = 20
    samples_per_function = 50
    number_of_iters = 10
    # number_of_iters = 1
    times = 4
    # times = 2

    plt.figure(figsize=(10, 10))

    master_graphs = [create_master_graph(n=n, type=type) for type in types]  # V,E
    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)

    X = np.array(range(20, 51, 10))
    # X = np.array(range(1, 7, 3))

    evals_mwu = np.zeros((len(types), len(X), times))
    evals_mwu_last = np.zeros((len(types), len(X), times))
    # evals_rand_greedy_one = np.zeros((len(types), len(X), times))
    evals_rand = np.zeros((len(types), len(X), 2))
    evals_wei = np.zeros((len(types), len(X), times))
    evals_top_k = np.zeros((len(types), len(X), times))
    # all_evals = [evals_mwu, evals_mwu_last, evals_rand_greedy_one, evals_wei, evals_top_k, evals_rand]

    for r, master_graph in enumerate(master_graphs):
        print "working on %s for exp 1:" % types[r]

        features = generate_features(lower=-B, upper=B, num_edges=len(master_graph.edges()), d=d, dis_type='hypercube')
        influence_functions = create_influence_functions(master_graph, eps_grid, features)

        for i, k in enumerate(X):

            mwu_runner = MWU(influence_functions=influence_functions, k=k, samples_per_function=samples_per_function)

            for j in xrange(times):

                mwu_solutions = mwu_runner.run(num_iters=number_of_iters, nu=0.5)

                degrees_map = np.zeros(n)
                for node in xrange(n):
                    degrees_map[node] = len(master_graph.neighbors(node))
                top_k_degrees = np.argsort(-degrees_map)[:k]

                wei_seed = get_wei_seed(master_graph, features, k=k, n=n, samples_per_function=samples_per_function)

                # rand_greedy_seed = rand_over_greedies(influence_functions, samples_per_function, k, n)

                evals_mwu[r][i][j] = mwu_runner.evaluate(mwu_solutions)
                evals_mwu_last[r][i][j] = mwu_runner.evaluate([mwu_solutions[-1]])
                # evals_rand_greedy_one[r][i][j] = mwu_runner.evaluate([rand_greedy_seed])
                evals_wei[r][i][j] = mwu_runner.evaluate([wei_seed])
                evals_top_k[r][i][j] = mwu_runner.evaluate([top_k_degrees])
                assert evals_mwu[r][i][j] >= k
                assert evals_mwu_last[r][i][j] >= k
                # assert evals_rand_greedy_one[r][i][j] < k
                assert evals_wei[r][i][j] >= k
                assert evals_top_k[r][i][j] >= k

            random_solutions = [np.random.choice(n, size=k, replace=False) for _ in xrange(30)]
            random_scores = []

            for rand in random_solutions:
                random_scores.append(mwu_runner.evaluate([rand]))
            random_scores = np.array(random_scores)
            assert all(random_scores >= k)
            evals_rand[r][i][0] = random_scores.mean()
            evals_rand[r][i][1] = random_scores.std()

        # plt.figure(r)

        # for eval in all_evals:
        #     plt.plot(X, eval[r])
        # plt.show()

    X.tofile(curr_exp_dir + "/X.npy")
    evals_mwu.tofile(curr_exp_dir + "/evals_mwu.npy")
    evals_mwu_last.tofile(curr_exp_dir + "/evals_mwu_last.npy")
    # evals_rand_greedy_one.tofile(curr_exp_dir + "/evals_rand_greedy_one.npy")
    evals_wei.tofile(curr_exp_dir + "/evals_wei.npy")
    evals_top_k.tofile(curr_exp_dir + "/evals_top_k.npy")
    for r, type in enumerate(types):

        plt.figure(r)
        plt.tight_layout()

        plt.plot(X, evals_mwu[r].mean(axis=1), '-', label='HIRO')
        plt.fill_between(X, evals_mwu[r].mean(axis=1) - evals_mwu[r].std(axis=1),
                         evals_mwu[r].mean(axis=1) + evals_mwu[r].std(axis=1), alpha=0.5, color='royalblue')

        plt.plot(X, evals_mwu_last[r].mean(axis=1), '-', label='MWU_last')
        plt.fill_between(X, evals_mwu_last[r].mean(axis=1) - evals_mwu_last[r].std(axis=1),
                         evals_mwu_last[r].mean(axis=1) + evals_mwu_last[r].std(axis=1), alpha=0.5, color='moccasin')

        # plt.plot(X, evals_rand_greedy_one[r].mean(axis=1), '-', label='random greedy')
        # plt.fill_between(X, evals_rand_greedy_one[r].mean(axis=1) - evals_rand_greedy_one[r].std(axis=1),
        #                  evals_rand_greedy_one[r].mean(axis=1) + evals_rand_greedy_one[r].std(axis=1), alpha=0.5,color='lightgreen')

        plt.plot(X, evals_wei[r].mean(axis=1) * np.ones_like(X), '-', label='LU greedy method', color='red')
        plt.fill_between(X, evals_wei[r].mean(axis=1) - evals_wei[r].std(axis=1),
                         evals_wei[r].mean(axis=1) + evals_wei[r].std(axis=1), alpha=0.5, color='salmon')

        plt.plot(X, evals_top_k[r].mean(axis=1) * np.ones_like(X), '-', label='top_deg', color='blueviolet')
        plt.fill_between(X, evals_top_k[r].mean(axis=1) - evals_top_k[r].std(axis=1),
                         evals_top_k[r].mean(axis=1) + evals_top_k[r].std(axis=1), alpha=0.5, color='thistle')

        plt.plot(X, evals_rand[r, :, 0], '-', label='random', color='saddlebrown')
        plt.fill_between(X, (evals_rand[r, :, 0] - evals_rand[r, :, 1]),
                         (evals_rand[r, :, 0] + evals_rand[r, :, 1]), alpha=0.5, color='tan')

        plt.ylabel(r"$\mathbf{ \min_i f_i(S)}$", fontweight='bold', fontsize=20)
        plt.xlabel("seed set size", fontweight='bold', fontsize=20)
        # plt.xlabel('\textbf{time} (s)')

        plt.title(types_map[types[r]], fontweight='bold', fontsize=25)
        plt.grid()

        if r == 0: plt.legend(loc=2)

        # plt.subplots_adjust(left=0.55)
        plt.savefig(curr_exp_dir + "/benchmarks-%s.pdf" % types[r], format='pdf', dpi=1000)
        plt.show()

    # ax[r2].plot(X, Y, 'k-', color='darkblue', label='MWU')
    # ax[r2].fill_between(X, np.ones_like(X) * (Y - Y_std), np.ones_like(X) * (Y + Y_std), color='royalblue',
    #                     alpha=0.5)
    # ax[r2].set_xlabel(types_map[types[r]])
    #
    # ax[r2].plot(X, np.ones_like(X) * evals.mean(), 'k-', color='firebrick', label='Random sets avg')
    # ax[r2].fill_between(X, np.ones_like(X) * (evals.mean() - evals.std()),
    #                     np.ones_like(X) * (evals.mean() + evals.std()), color='lightcoral',
    #                     alpha=0.5)
    # # fig.savefig(curr_exp_dir + "/MWU_as_function_of_m.eps", format='eps', dpi=1000)
    # print(Y.max())
    # plt.legend()
    # plt.show()

    # fig.savefig(curr_exp_dir + "/random_sets_histogram.eps", format='eps', dpi=1000)


def ex2():
    """
    Plot the value of the max-min vs greedy/random/wei and best if possible for the graph as function of uncertainty.
    """

    curr_exp_dir = os.path.join('ex2', curr_dir)
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)

    d = 5
    B = 1
    n = 500

    m = 15
    samples_per_function = 50
    times = 5

    plt.figure(figsize=(10, 10))

    master_graphs = [create_master_graph(n=n, type=type) for type in types]  # V,E
    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)

    X = np.array(range(1, 20, 5))

    k_grid = np.array([10, 25, 50])

    evals_mwu = np.zeros((len(types), len(k_grid), len(X), times))

    for r, master_graph in enumerate(master_graphs):
        print "working on %s for exp 2:" % types[r]

        features = generate_features(lower=-B, upper=B, num_edges=len(master_graph.edges()), d=d, dis_type='hypercube')
        influence_functions = create_influence_functions(master_graph, eps_grid, features)

        for k_i, k in enumerate(k_grid):
            mwu_runner = MWU(influence_functions=influence_functions, k=k, samples_per_function=samples_per_function)

            for i, number_of_iters in enumerate(X):
                for j in xrange(times):
                    mwu_solutions = mwu_runner.run(num_iters=number_of_iters, nu=0.5)

                    evals_mwu[r][k_i][i][j] = mwu_runner.evaluate(mwu_solutions)

        # for eval in all_evals:
        #     plt.plot(X, eval[r])
        # plt.show()

    X.tofile(curr_exp_dir + "/X.npy")
    evals_mwu.tofile(curr_exp_dir + "/evals_mwu.npy")

    for r, type in enumerate(types):
        plt.figure(r)
        plt.tight_layout()

        plt.plot(X, evals_mwu[r][0].mean(axis=1), '-', label='HIRO k=7', color='darkblue')
        plt.fill_between(X, evals_mwu[r][0].mean(axis=1) - evals_mwu[r][0].std(axis=1),
                         evals_mwu[r][0].mean(axis=1) + evals_mwu[r][0].std(axis=1), alpha=0.5, color='royalblue')

        plt.plot(X, evals_mwu[r][1].mean(axis=1), '-', label='HIRO k=10', color='firebrick')
        plt.fill_between(X, evals_mwu[r][1].mean(axis=1) - evals_mwu[r][1].std(axis=1),
                         evals_mwu[r][1].mean(axis=1) + evals_mwu[r][1].std(axis=1), alpha=0.5, color='lightcoral')

        plt.plot(X, evals_mwu[r][2].mean(axis=1), '-', label='HIRO k=13', color='darkgreen')
        plt.fill_between(X, evals_mwu[r][2].mean(axis=1) - evals_mwu[r][2].std(axis=1),
                         evals_mwu[r][2].mean(axis=1) + evals_mwu[r][2].std(axis=1), alpha=0.7, color='mediumseagreen')

        plt.ylabel(r"$\mathbf{ \min_i f_i(S)}$", fontweight='bold', fontsize=20)
        plt.xlabel(types_map[types[r]])
        plt.title(types_map[types[r]], fontweight='bold', fontsize=25)
        plt.xlabel("number of iterations", fontweight='bold', fontsize=20)
        plt.grid()

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1 * 0.9, 1.05 * y2))

        if r == 0: plt.legend(loc=2)
        plt.savefig(curr_exp_dir + "/mwu-convergence-%s.pdf" % types[r], format='pdf', dpi=1000)
        plt.show()


def ex3():
    curr_exp_dir = os.path.join('ex3', curr_dir)
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)
    d = 5
    B = 1
    n = 500
    # n = 20

    m = 50
    samples_per_function = 50
    # samples_per_function = 1
    number_of_iters = 5
    # number_of_iters = 1
    times = 5
    # times = 3

    plt.figure(figsize=(10, 10))
    master_graphs = [create_master_graph(n=n, type=type) for type in types]  # V,E
    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)

    X = np.array([1, 10, 20, 30, 40, 50])
    k_grid = np.array([10, 25, 50])

    Y = np.zeros((len(master_graphs), len(k_grid), len(X)))
    Y_std = np.zeros((len(master_graphs), len(k_grid), len(X)))

    for r, master_graph in enumerate(master_graphs):
        print "exp 3, graph %s" % (types[r])

        features = generate_features(lower=-B, upper=B, num_edges=len(master_graph.edges()), d=d, dis_type='hypercube')
        influence_functions = create_influence_functions(master_graph, eps_grid, features)
        for k_i, k in enumerate(k_grid):
            main_runner = MWU(influence_functions=influence_functions, k=k, samples_per_function=samples_per_function)
            # mwu_solutions = main_runner.run(num_iters=number_of_iters, nu=0.5)

            for j, m_prime in enumerate(X):
                # curr_indices = np.random.choice(200, size=X.max(), replace=False)
                # print curr_indices
                print m_prime
                mwu_runner = MWU(influence_functions=[influence_functions[i] for i in range(m_prime)], k=k,
                                 samples_per_function=samples_per_function)

                # times = 1
                inner_evals = np.zeros(times)
                for i in range(times):
                    inner_evals[i] = main_runner.evaluate(mwu_runner.run(num_iters=number_of_iters, nu=0.5))

                Y[r][k_i][j] = inner_evals.mean()
                Y_std[r][k_i][j] = inner_evals.std()

        # a_rand = [np.random.choice(n, size=k, replace=False) for _ in xrange(100)]
        # a_rand = [np.random.choice(n, size=k, replace=False) for _ in xrange(50)]
        # evals.append(np.array([main_runner.evaluate([rset]) for rset in a_rand]))

    # evals = np.array(evals)
    # evals.tofile(curr_exp_dir + "/evals.npy")
    X.tofile(curr_exp_dir + "/X.npy")
    Y.tofile(curr_exp_dir + "/Y.npy")
    Y_std.tofile(curr_exp_dir + "/Y_std.npy")

    for r, type in enumerate(types):
        plt.figure(r)
        plt.tight_layout()
        # plt.rc('text', usetex=True)
        plt.plot(X, Y[r][0], '-', color='darkblue', label='HIRO k=7')
        plt.fill_between(X, Y[r][0] - Y_std[r][0], Y[r][0] + Y_std[r][0], color='royalblue',
                         alpha=0.5)

        plt.plot(X, Y[r][1], '-', color='firebrick', label='HIRO k=10')
        plt.fill_between(X, Y[r][1] - Y_std[r][1], Y[r][1] + Y_std[r][1], color='lightcoral',
                         alpha=0.5)

        plt.plot(X, Y[r][2], '-', color='darkgreen', label='HIRO k=13')
        plt.fill_between(X, Y[r][2] - Y_std[r][2], Y[r][2] + Y_std[r][2], color='mediumseagreen',
                         alpha=0.5)

        # plt.plot(X, np.ones_like(X) * evals[r].mean(), 'k-', color='firebrick', label='Random sets avg')
        # plt.fill_between(X, np.ones_like(X) * (evals[r].mean() - evals[r].std()),
        #                  np.ones_like(X) * (evals[r].mean() + evals[r].std()), color='lightcoral',
        #                  alpha=0.5)

        plt.ylabel(r"$\mathbf{ \min_i f_i(S)}$", fontweight='bold', fontsize=20)
        plt.xlabel("number of functions", fontweight='bold', fontsize=20)
        # plt.xlabel('\textbf{time} (s)')

        plt.title(types_map[types[r]], fontweight='bold', fontsize=25)
        plt.grid()

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1 * 0.95, 1.05 * y2))

        if r == 0: plt.legend(loc=2)

        # plt.subplots_adjust(left=0.55)
        plt.savefig(curr_exp_dir + "/random_sets_histogram-%s.pdf" % types[r], format='pdf', dpi=1000)
        plt.show()


def ex4():
    """
    Plot the value of the max-min vs greedy/random/wei and best if possible for the graph as function of k.
    """

    curr_exp_dir = os.path.join('ex4', curr_dir)
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)

    d = 5
    B = 1
    n = 500
    # n = 50
    k = 10

    m = 15
    samples_per_function = 50
    number_of_iters = 10
    times = 5
    # times = 2

    plt.figure(figsize=(10, 10))

    master_graphs = [create_master_graph(n=n, type=type) for type in types]  # V,E
    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)

    X = (k * np.math.log(100) * np.linspace(0.1, 1, 10)).astype(int)

    evals_mwu = np.zeros((len(types), len(X), times))
    # evals_mwu_last = np.zeros((len(types), times))

    for r, master_graph in enumerate(master_graphs):
        print "working on %s for exp 4:" % types[r]

        features = generate_features(lower=-B, upper=B, num_edges=len(master_graph.edges()), d=d, dis_type='hypercube')
        influence_functions = create_influence_functions(master_graph, eps_grid, features)

        mwu_runner = MWU(influence_functions=influence_functions, k=k, samples_per_function=samples_per_function)

        for j in xrange(times):

            mwu_solutions = mwu_runner.run(num_iters=number_of_iters, nu=0.5)
            union_of_solutions = np.array(list(set.union(*[set(sol) for sol in mwu_solutions])))
            # if r == 0 and j == 0:
            #     print union_of_solutions
            num_all_solutions = len(union_of_solutions)
            print 'len of union=%d, k=%d', (num_all_solutions, k)

            for i, alpha_k in enumerate(X):
                indices = np.random.choice(num_all_solutions, min(max(k, alpha_k), num_all_solutions), replace=False)
                curr_union = union_of_solutions[indices]
                evals_mwu[r][i][j] = mwu_runner.evaluate([curr_union])

    #             # evals_mwu_last[r][j] = mwu_runner.evaluate([mwu_solutions[-1]])

    # plt.figure(r)

    # for eval in all_evals:
    #     plt.plot(X, eval[r])
    # plt.show()

    X.tofile(curr_exp_dir + "/X.npy")
    evals_mwu.tofile(curr_exp_dir + "/evals_mwu.npy")
    # evals_mwu_last.tofile(curr_exp_dir + "/evals_mwu_last.npy")

    for r, type in enumerate(types):

        plt.figure(r)
        plt.tight_layout()
        plt.plot(X, evals_mwu[r].mean(axis=1), '-', label='union score', color='blue')
        plt.fill_between(X, evals_mwu[r].mean(axis=1) - evals_mwu[r].std(axis=1),
                         evals_mwu[r].mean(axis=1) + evals_mwu[r].std(axis=1), alpha=0.5, color='royalblue')

        # plt.plot(X, evals_mwu_last[r].mean()*np.ones_like(X), '-', label='last HIRO')
        # plt.fill_between(X, evals_mwu_last[r].mean()*np.ones_like(X) - evals_mwu_last[r].std(),
        #                  evals_mwu_last[r].mean()*np.ones_like(X) + evals_mwu_last[r].std(), alpha=0.5, color='moccasin')

        plt.ylabel(r"$\mathbf{ \min_i f_i(S)}$", fontweight='bold', fontsize=20)
        plt.xlabel("seed set size", fontweight='bold', fontsize=20)
        # plt.xlabel('\textbf{time} (s)')

        plt.title(types_map[types[r]], fontweight='bold', fontsize=25)
        plt.grid()

        if r == 0: plt.legend(loc=2)

        # plt.subplots_adjust(left=0.55)
        plt.savefig(curr_exp_dir + "/union-%s.pdf" % types[r], format='pdf', dpi=1000)
        plt.show()


def powerset(iterable):
    from itertools import chain, combinations
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def score_for_bf(graph, seed, all_probs):
    scores = np.zeros(all_probs.shape[0])
    for edges_numbers in powerset(range(len(graph.edges()))):
        if len(edges_numbers) == 0:
            continue
        edges = [graph.edges()[i] for i in edges_numbers]

        mask = np.zeros(all_probs.shape[1], dtype=bool)  # np.ones_like(a,dtype=bool)
        mask[np.array(edges_numbers)] = True

        prob = all_probs[:, mask]
        not_prob = 1. - all_probs[:, ~mask]
        g = Graph(len(graph.nodes()), edges)
        reachable = MWU.find_reachable_nodes(g, set(seed))
        scores += np.prod(prob, axis=1) * np.prod(not_prob, axis=1) * len(reachable)
    return scores.min()


def brute_force(influence_functions, k):
    if len(influence_functions[0].nodes) > 20 or k > 3:
        raise RuntimeError
    all_seeds = set(itertools.combinations(influence_functions[0].nodes, k))
    best_score, best_seed = 0, None
    all_probs = np.concatenate([ifunc.probabilities for ifunc in influence_functions
                                ]).reshape(len(influence_functions), -1)
    for seed in all_seeds:
        curr_score = score_for_bf(influence_functions[0].graph, seed, all_probs)
        if curr_score > best_score:
            best_score = curr_score
            best_seed = seed
    return best_seed, best_score


def debug_test():
    ####
    # Create small graph
    # find the best seed in the small graph using brute-force
    # run-mwu and find best seed distribution
    # compare
    ###
    d = 5
    B = 1
    n = 10
    k = 2

    m = 20
    samples_per_function = 50
    number_of_iters = 10
    graph = create_master_graph(n=n, type=types[0])
    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)
    features = generate_features(lower=-B, upper=B, num_edges=len(graph.edges()), d=d, dis_type='hypercube')
    influence_functions = create_influence_functions(graph, eps_grid, features)

    mwu_runner = MWU(influence_functions=influence_functions[:5], k=k, samples_per_function=samples_per_function)
    mwu_solutions = mwu_runner.run(num_iters=number_of_iters, nu=0.5)
    print mwu_solutions
    print "mwu score: ", mwu_runner.evaluate(mwu_solutions)
    print "mwu last score: ", mwu_runner.evaluate(mwu_solutions[-2:-1])
    print "-----------------------"
    seed, score = brute_force(influence_functions, k)
    print "real score: ", score, seed
    print "best score with respect to mwu_runner: ", mwu_runner.evaluate([seed])
    print "-----------------------"

    random_solutions = [np.random.choice(n, size=k, replace=False) for _ in xrange(12)]
    random_scores = []
    for rand in random_solutions:
        random_scores.append(mwu_runner.evaluate([rand]))
    plt.hist(random_scores)
    plt.show()

    print "hey"


def real_data_exp():

    # raw_graph = pd.read_csv('soc-sign-bitcoinalpha.csv', header=None).values[:, 0:2]
    # d = 5
    # B = 1
    # n = 10
    # k = 2
    #
    # m = 20
    # samples_per_function = 50
    # number_of_iters = 10
    #
    #
    # eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)
    # features = generate_features(lower=-B, upper=B, num_edges=len(graph.edges()), d=d, dis_type='hypercube')
    # influence_functions = create_influence_functions(graph, eps_grid, features)

    curr_exp_dir = os.path.join('real_data_exp1', curr_dir)
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)

    raw_graph = pd.read_csv('soc-sign-bitcoinalpha.csv', header=None).values[:, 0:2]
    graph = networkx.Graph()
    graph.add_nodes_from(range(raw_graph.max()))
    graph.add_edges_from(raw_graph)

    d = 5
    B = 1
    times = 2
    samples_per_function = 50
    m = 20

    eps_grid = sample_thetas(lower=-B, upper=B, m=m, d=d)

    X = np.array(range(1, 20, 5))

    k_grid = np.array([10, 25, 50])

    evals_mwu = np.zeros((len(k_grid), len(X), times))


    print "working on for real data"

    features = generate_features(lower=-B, upper=B, num_edges=len(graph.edges()), d=d, dis_type='hypercube')
    influence_functions = create_influence_functions(graph, eps_grid, features)

    for k_i, k in enumerate(k_grid):
        mwu_runner = MWU(influence_functions=influence_functions, k=k, samples_per_function=samples_per_function)

        for i, number_of_iters in enumerate(X):
            for j in xrange(times):
                mwu_solutions = mwu_runner.run(num_iters=number_of_iters, nu=0.5)

                evals_mwu[k_i][i][j] = mwu_runner.evaluate(mwu_solutions)

    # for eval in all_evals:
    #     plt.plot(X, eval[r])
    # plt.show()

    X.tofile(curr_exp_dir + "/X.npy")
    evals_mwu.tofile(curr_exp_dir + "/evals_mwu.npy")

    plt.tight_layout()

    plt.plot(X, evals_mwu[0].mean(axis=1), '-', label='HIRO k=7', color='darkblue')
    plt.fill_between(X, evals_mwu[0].mean(axis=1) - evals_mwu[0].std(axis=1),
                     evals_mwu[0].mean(axis=1) + evals_mwu[0].std(axis=1), alpha=0.5, color='royalblue')

    plt.plot(X, evals_mwu[1].mean(axis=1), '-', label='HIRO k=10', color='firebrick')
    plt.fill_between(X, evals_mwu[1].mean(axis=1) - evals_mwu[1].std(axis=1),
                     evals_mwu[1].mean(axis=1) + evals_mwu[1].std(axis=1), alpha=0.5, color='lightcoral')

    plt.plot(X, evals_mwu[2].mean(axis=1), '-', label='HIRO k=13', color='darkgreen')
    plt.fill_between(X, evals_mwu[2].mean(axis=1) - evals_mwu[2].std(axis=1),
                     evals_mwu[2].mean(axis=1) + evals_mwu[2].std(axis=1), alpha=0.7, color='mediumseagreen')

    plt.ylabel(r"$\mathbf{ \min_i f_i(S)}$", fontweight='bold', fontsize=20)
    plt.xlabel('real data')
    plt.title('real data', fontweight='bold', fontsize=25)
    plt.xlabel("number of iterations", fontweight='bold', fontsize=20)
    plt.grid()

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1 * 0.9, 1.05 * y2))

    plt.legend(loc=2)
    plt.savefig(curr_exp_dir + "/mwu-convergence-real.pdf", format='pdf', dpi=1000)
    plt.show()


if __name__ == '__main__':
    # G = nx.Graph()
    # G = nx.barabasi_albert_graph(100, 2)
    # G = nx.erdos_renyi_graph(100, 0.05)
    # G = nx.watts_strogatz_graph(100, 5, 0.3)  # small world
    # z = nx.utils.create_degree_sequence(100, powerlaw_sequence)
    # G = nx.configuration_model(z)
    # G = nx.Graph(G)  # remove parallel edges
    # G.remove_edges_from(G.selfloop_edges())  # remove self-loops
    # print(G)
    # G.add_nodes_from(xrange(50))
    # graph = nx.generators.random_graphs.barabasi_albert_graph(100, 10)

    # node_pos = np.random.random((100, 2))
    # nx.draw_networkx(G, node_pos, node_color='g')
    # plt.draw()
    # plt.show()

    # print graph.inner_graph
    # l, _ = graph.adjacencyList()
    # print type(l)
    if not os.path.exists(os.path.abspath('ex1')):
        os.makedirs("ex1")
    if not os.path.exists(os.path.abspath('ex2')):
        os.makedirs("ex2")
    if not os.path.exists(os.path.abspath('ex3')):
        os.makedirs("ex3")
    if not os.path.exists(os.path.abspath('ex4')):
        os.makedirs("ex4")
    # rc('text', usetex=True)
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # debug_test()
    real_data_exp()
    print 'hey'
