import numpy as np
import copy
import itertools
import networkx as nx


class InfluenceFunction:

    def __init__(self, graph, thetas, features, probabilities='hack_delete'):
        """
        Each edge has a feature
        :param graph:
        :param features:
        """
        self.graph = graph
        self.nodes = graph.nodes()
        self.edges = graph.edges()
        # todo pandas if code is slow
        self.features = features
        # self.probabilities = 1 / (1 + np.exp(-np.array(thetas).dot(features.T))) / 2.0
        self.probabilities = 1 / (1 + np.exp(-np.array(thetas).dot(features.T)))
        # self.probabilities = probabilities

    def fetch_graph(self, wei=None):
        """
        Produces a subgraph (each edge included with self.probability) from master graph.
        """
        this_edge_list = []
        rands = np.random.random(len(self.edges))
        probs = self.probabilities
        if wei == 'max':
            probs = self.probabilities_max
        elif wei == 'min':
            probs = self.probabilities_min

        for i in xrange(len(self.edges)):
            if rands[i] < probs[i]:
                this_edge_list.append(self.edges[i])

        return Graph(len(self.nodes), this_edge_list)

    def fetch_graphs(self, samples_per_function=50, wei=None):
        """
        Produces a subgraph (each edge included with self.probability) from master graph.
        """
        return [self.fetch_graph(wei) for _ in xrange(samples_per_function)]

    def draw(self):
        node_pos = np.random.random((len(self.nodes), 2))
        nx.draw_networkx(self.graph, node_pos, node_color='g')
        # todo add probabilities/ features option mb for small graphs

    def calc_wei_probabilities(self):
        self.probabilities_min = 1 / (1 + np.exp(np.sum(np.abs(self.features), axis=1)))
        self.probabilities_max = 1 / (1 + np.exp(np.sum(-np.abs(self.features), axis=1)))


class Graph:
    def __init__(self, num_vertices, edge_list):
        self.num_vertices = num_vertices
        self.nodes = list(range(num_vertices))
        self.edge_list = edge_list
        self.adj_list = {}

        for i in xrange(self.num_vertices):
            self.adj_list[i] = []
        for edge in self.edge_list:
            self.adj_list[edge[0]].append(edge[1])
            self.adj_list[edge[1]].append(edge[0])


class MWU:
    def __init__(self, influence_functions, k, samples_per_function):
        self.num_vertices = len(influence_functions[0].nodes)
        self.k = k
        self.influence_functions = influence_functions
        self.num_influence_functions = len(influence_functions)
        # self.num_graphs = len(self.graphs)
        self.dist_over_graphs = None
        self.all_dist_over_graphs = None
        self.samples_per_function = samples_per_function

    def get_min_influence(self, combo):
        return np.array([self.find_reachable_nodes(graph, set(combo)) for graph in self.graphs]).min()

    def compute_max_min_influence(self):
        """
        Brute force finding set of nodes of size self.k that maximizes influence.
        Should only be run when number of possible sets of size self.k is sufficiently small.
        :return:
        """

        influences = np.array([self.get_min_influence(combo) for combo in
                               itertools.combinations(range(self.num_vertices), self.k)])

        best_solution = np.argmax(influences)
        best_influence = influences[best_solution]

        return best_influence, best_solution

    def evaluate(self, solutions):
        all_evaluations = np.zeros((len(solutions), self.num_influence_functions))

        for j in xrange(self.num_influence_functions):
            self.fetch_graphs(samples_per_function=self.samples_per_function, fetch_from=j)
            for i, sol in enumerate(solutions):
                all_evaluations[i, j] = np.array([len(self.find_reachable_nodes(graph, set(sol)))
                                                  for graph in self.graphs]).mean()
        return all_evaluations.mean(axis=0).min()
        # return np.min(all_evaluations.mean(axis=0))

    @staticmethod
    def find_reachable_nodes(graph, start):
        """
        Returns indices of all nodes reachable in graph from given start vertex.
        """
        reached = set()

        if type(start) != set:
            stack = {start}
        else:
            stack = set(list(start))

        while len(stack) > 0:
            curr_node = stack.pop()

            reached = reached.union({curr_node})

            neighbors = set(graph.adj_list[curr_node])
            neighbors = neighbors.difference(reached, stack)
            stack = stack.union(neighbors)
        return reached

    @staticmethod
    def coverage_problem_builder(graphs):
        """
        Formatting of reachable nodes from each node in all subgraphs.
        Formatted structure is exploited by coverage_greedy_solver.
        """
        graph_to_components_map = {i: {} for i in xrange(len(graphs))}

        for i, graph in enumerate(graphs):
            unreachable_nodes = set(graph.nodes)
            while unreachable_nodes:
                node = unreachable_nodes.pop()
                curr_reachable_nodes = MWU.find_reachable_nodes(graph, node)
                unreachable_nodes = unreachable_nodes.difference(curr_reachable_nodes)
                for node in curr_reachable_nodes:
                    graph_to_components_map[i][node] = curr_reachable_nodes

        return graph_to_components_map

    @staticmethod
    def coverage_greedy_solver(orig_coverage_dict, k, num_vertices, num_graphs):
        """
        Runs the (1-1/e) approximation greedy algorithm.
        """

        coverage_dict = orig_coverage_dict
        curr_solution = set()

        # Populate total marginal weight adding each vertex would contribute.
        weight_left = np.zeros(num_vertices)
        for i in xrange(num_graphs):
            for node in xrange(num_vertices):
                weight_left[node] += len(coverage_dict[i][node])

        # Run greedy algorithm.
        for _ in xrange(k):
            if weight_left.max() == 0.0:
                left_nodes = np.array(list(set(np.arange(num_vertices)).difference(curr_solution)))
                curr_solution = set(curr_solution.union(
                                set(left_nodes[np.random.choice(len(left_nodes), k-len(curr_solution),
                                                                replace=False)])))
                break
            selected_node = np.argmax(weight_left)
            curr_solution = curr_solution.union({selected_node})

            weight_left[selected_node] = 0.0
            # Remove elements from remaining nodes in the coverage dictionary.
            for i in xrange(num_graphs):
                if selected_node not in coverage_dict[i]:
                    continue
                weight_diff = len(coverage_dict[i][selected_node])
                for node in list(coverage_dict[i][selected_node]):
                    if node != selected_node:
                        weight_left[node] -= weight_diff
                    coverage_dict[i].pop(node)

        return curr_solution

    def run(self, num_iters, nu, is_printing_dist=False):
        """
        num_iters: how many simulations to run
        nu: parameter for MWU where eta = (log(m) / 2T)^(nu); can be anything if dist_update_type == 'fixed'
        Returns the average bottleneck influence across iterations and the solutions sets from all iterations.
        """
        max_graph_for_greedy = 1000
        # Make sure starting fresh in case previous runs occurred with this class.
        self.dist_over_graphs = np.ones(self.num_influence_functions) / self.num_influence_functions  # 1/m
        self.all_dist_over_graphs = [self.dist_over_graphs]

        solutions = []
        solutions_scores = np.zeros(
            (num_iters, self.num_influence_functions))  # there is a score for each pair of iter, graph
        solutions_scores_sums = np.zeros(self.num_influence_functions)  # the sums of the scores = (sum over iterations)
        # Set up coverage problem.

        for i in xrange(num_iters):
            print 'mwu: iter i = ' + str(i)
            # Obtain new solution.
            # bins = self.fetch_graphs(max_graph_for_greedy)
            self.fetch_graphs(max_graph_for_greedy)
            # bin_mapping = np.cumsum(bins)  # in order to map a sampled graph j to influence function IF use
            #                                         # IF = np.argmin(bin_mapping) <= j)
            graph_to_components_map = self.coverage_problem_builder(
                self.graphs)  # data structure that stores the reachable nodes
            curr_solution = self.coverage_greedy_solver(graph_to_components_map, k=self.k, num_graphs=self.num_graphs,
                                                        num_vertices=self.num_vertices)

            assert len(curr_solution) == self.k
            # curr_solution = self.celf(graph, k)
            solutions.append(list(curr_solution))

            # Update distribution over graphs.
            unnormalized_dist = np.zeros(self.num_influence_functions)

            # for g_index, graph in enumerate(self.graphs):
            #     j = np.argmin(bin_mapping <= g_index)  # index of influence function
            #     bin_size = bins[j]  # number of graphs in the current bin.
            #     assert bin_size != 0
            #     solutions_scores[i, j] += len(self.find_reachable_nodes(graph, curr_solution)) /
            #                               float(bin_size*2*self.k)
            # solutions_scores_sums += solutions_scores[i, :]

            for j in xrange(self.num_influence_functions):
                self.fetch_graphs(samples_per_function=100, fetch_from=j)
                curr_inf = np.array(
                    [len(self.find_reachable_nodes(graph, curr_solution)) for graph in self.graphs]).mean()
                # print curr_inf, self.influence_functions[j].probabilities.mean() * self.num_vertices

                solutions_scores[i, j] += curr_inf / (1.1 * self.k)
                solutions_scores_sums[j] += solutions_scores[i, j]

                unnormalized_dist[j] = np.exp(-1.0 * solutions_scores_sums[j] *
                                              ((np.log(self.num_influence_functions) / (
                                                      2.0 * num_iters)) ** nu))

            self.dist_over_graphs = unnormalized_dist / unnormalized_dist.sum()
            self.all_dist_over_graphs.append(self.dist_over_graphs)
            if i % 100 == 0 and i > 0:
                print 'hey'

            if is_printing_dist:
                print self.dist_over_graphs
                print 'Min: ', solutions_scores[i, :].min()
                print 'Max: ', solutions_scores[i, :].max()

        # Evaluate performance.
        # bottleneck_influence = solutions_scores_sums  # the worst avg performance of any f_i.
        # print bottleneck_influence

        # return solutions, bottleneck_influence
        return solutions

    def fetch_graphs(self, samples_per_function=100, fetch_from=None):

        if fetch_from:
            self.graphs = self.influence_functions[fetch_from].fetch_graphs(samples_per_function)
            self.num_graphs = len(self.graphs)
            return

        choices = np.random.choice(len(self.influence_functions),
                                   size=samples_per_function, replace=True, p=self.dist_over_graphs)
        # this variable will hold the amount of graphs to sample from each function
        choices_bins = np.bincount(choices, minlength=self.num_influence_functions)

        self.graphs = list(itertools.chain.from_iterable([IF.fetch_graphs(choices_bins[i])
                                                          for i, IF in enumerate(self.influence_functions)]))
        self.num_graphs = len(self.graphs)

        return choices_bins  # for later normalization
