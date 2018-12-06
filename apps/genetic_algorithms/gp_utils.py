from dateutil import parser
from deap import gp
from apps.backtesting.utils import datetime_from_timestamp
import datetime

def recompute_tree_graph(nodes, edges):
    children = {node: [] for node in nodes}
    for a, b in edges:
        children[a].append(b)
    return children


def visit_node(graph, node, labels, parent = None):
    children = [visit_node(graph, child, labels, node)
                for child in graph[node]
                if child != parent]
    return [labels[node], children]


def compress(individual):
    nodes, edges, labels = gp.graph(individual)
    from gp_utils import recompute_tree_graph
    graph = recompute_tree_graph(nodes, edges)
    root = visit_node(graph, 0, labels)
    optimized = optimize(root)
    return wrap_children(optimized)


def wrap_children(optimized):
    node, children = optimized
    if len(children) == 0:
        return node
    s = f'{str(node)}('
    for child in children:
        s += str(wrap_children(child))
        s += ','
    s = s[:-1]
    s += ')'
    return s


def optimize(node):
    node_type, children = node
    # optimize all children
    for i in range(len(children)):
        children[i] = optimize(children[i])
    if node_type == "if_then_else":
        cond_type = children[0][0]
        if cond_type is True:
            return children[1]
        if cond_type is False:
            return children[2]
        if str(children[1]) == str(children[2]):
            return children[1]
        # if one of the children contains the same condition
        # for example, if_then_else(rsi_lt_30(ARG0),sell,if_then_else(rsi_lt_30(ARG0),buy,sell))
        child1_node_type, child1 = children[1]
        child2_node_type, child2 = children[2]
        if child1_node_type == "if_then_else" and str(child1[0]) == str(children[0]): # repeated condition
            return child1[1] # if the condition is true, first child gets picked, so we know it's also true
        if child2_node_type  == "if_then_else" and str(child2[0]) == str(children[0]):
            return child2[2]  # if the condition is false, second child gets picked, so we know it's also false

    elif node_type == "or_":
        if children[0][0] is True or children[1][0] is True:
            return [True, []]
        if children[0][0] is False:
            return children[1]
        if children[1][0] is False:
            return children[0]
    elif node_type == "and_":
        if children[0][0] is False or children[1][0] is False:
            return [False, []]
        if children[0][0] is True:
            return children[1]
        if children[1][0] is True:
            return children[0]
    elif node_type == "xor":
        if (children[0][0] is True and children[1][0] is False) or (children[0][0] is False and children[1][0] is True):
            return [True, []]
        if (children[0][0] is True and children[1][0] is True) or (children[0][0] is False and children[1][0] is False):
            return [False, []]
        if children[0][0] is False:
            return children[1]
        if children[1][0] is False:
            return children[0]
    elif node_type == "lt":
        if str(children[0][0]) == str(children[1][0]):
            return [False, []]
        try:
            child1 = float(children[0][0])
            child2 = float(children[1][0])
            return [child1 < child2, []]
        except:
            pass
    elif node_type == "gt":
        if str(children[0][0]) == str(children[1][0]):
            return [False, []]
        try:
            child1 = float(children[0][0])
            child2 = float(children[1][0])
            return [child1 > child2, []]
        except:
            pass

    elif node_type in ("identity_list", "identity_bool", "identity_float"):
        return children[0]

    # can't optimize this node
    # (and children are already optimized)
    return node


class Period:
    def __init__(self, start_time_str, end_time_str, name=None):
        self.start_time_str = start_time_str
        self.start_time = self._get_timestamp(start_time_str)
        self.end_time_str = end_time_str
        self.end_time = self._get_timestamp(end_time_str)
        if name == None:
            name = datetime.datetime.utcfromtimestamp(self.start_time).strftime('%M %d')
        self.name = name

    def _get_timestamp(self, period_str):
        return parser.parse(period_str).timestamp()

    def get_shifted_forward(self, seconds, name=None):
        start_time_str = datetime_from_timestamp(self.start_time + seconds)
        end_time_str = datetime_from_timestamp(self.end_time + seconds)
        return Period(start_time_str, end_time_str)

    def __str__(self):
        return f'{self.start_time_str} - {self.end_time_str}'