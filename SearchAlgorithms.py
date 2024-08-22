from collections import defaultdict
from topspin import TopSpinState
from heuristics import BaseHeuristic
import numpy as np
from queue import PriorityQueue

class Node:
    def __init__(self, state, parent, g, h, w):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + w * h

    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h

        return self.f < other.f


def AStar(start, heuristic_function, T):
    """
    A* Search Algorithm
    :param start: node to start the search
    :param heuristic_function: heuristic function to use
    :param T: maximum number of expansions
    :return: path and number of expansions
    """
    if isinstance(start, list):
        start = TopSpinState(start)

    open_list = PriorityQueue()
    closed_dict = defaultdict(int)
    UB, nuB = np.inf, None
    expansions = 0

    h_start = heuristic_function([start])

    if isinstance(h_start, list):
        h_start = h_start[0]

    n_start = Node(start, None, g=0, h=h_start, w=1)  # Weight is 1 for A*
    open_list.put((n_start.f, n_start))

    while not open_list.empty() and expansions < T:
        _, current_node = open_list.get()
        s, g, p, f = current_node.state, current_node.g, current_node.parent, current_node.f
        expansions += 1

        if s.is_goal():
            UB = g
            nuB = current_node
            break

        for neighbor in s.get_neighbors():
            neighbor_state, neighbor_cost = neighbor
            g_neighbor = g + neighbor_cost
            key_neighbor = neighbor_state
            if key_neighbor not in closed_dict or g_neighbor < closed_dict[key_neighbor]:
                closed_dict[key_neighbor] = g_neighbor
                h_neighbor = heuristic_function([neighbor_state])[0]
                ns = Node(neighbor_state, current_node, g_neighbor, h_neighbor, 1)
                open_list.put((ns.f, ns))

    if nuB is None:
        return None, T

    node = nuB
    path = []
    while node:
        s, g, p, f = node.state, node.g, node.parent, node.f
        s_lst = s.get_state_as_list()
        path.append(s_lst)
        node = p

    return path[::-1], expansions


def WAStar(start, weight, heuristic_function, T):
    """
    Weighted A* Search Algorithm
    :param start: node to start the search
    :param weight: weight for the heuristic function
    :param heuristic_function: heuristic function to use
    :param T: maximum number of expansions
    :return: path and number of expansions
    """
    if isinstance(start, list):
        start = TopSpinState(start)

    open_list = PriorityQueue()
    closed_dict = defaultdict(int)
    UB, nuB = np.inf, None
    expansions = 0

    h_start = heuristic_function([start])

    if isinstance(h_start, list):
        h_start = h_start[0]

    n_start = Node(start, None, g=0, h=h_start, w=weight)
    open_list.put((n_start.f, n_start))

    while not open_list.empty() and expansions < T:
        _, current_node = open_list.get()
        s, g, p, f = current_node.state, current_node.g, current_node.parent, current_node.f
        expansions += 1

        if s.is_goal():
            if UB > g:
                UB, nuB = g, current_node
            continue

        for neighbor in s.get_neighbors():
            neighbor_state, neighbor_cost = neighbor
            g_neighbor = g + neighbor_cost
            key_neighbor = neighbor_state
            if key_neighbor not in closed_dict or g_neighbor < closed_dict[key_neighbor]:
                closed_dict[key_neighbor] = g_neighbor
                h_neighbor = heuristic_function([neighbor_state])[0]
                ns = Node(neighbor_state, current_node, g_neighbor, h_neighbor, weight)
                open_list.put((ns.f, ns))

    if nuB is None:
        return None, T

    node = nuB
    path = []
    while node:
        s, g, p, f = node.state, node.g, node.parent, node.f
        s_lst = s.get_state_as_list()
        path.append(s_lst)
        node = p

    return path[::-1], expansions


def BWAS(start, weight, batch_size, heuristic_function, T):
    """
    Batch Weighted A* Search
    :param start: node to start the search
    :param weight: weight for the heuristic function
    :param batch_size: batch size for the search
    :param heuristic_function: heuristic function to use
    :param T: maximum number of expansions
    :return: path and number of expansions
    """
    if isinstance(start, list):
        start = TopSpinState(start)

    open_list = PriorityQueue()
    closed_dict = defaultdict(int)
    UB, nuB = np.inf, None
    LB = 0
    expansions = 0

    h_start = heuristic_function([start])

    if isinstance(h_start, list):
        h_start = h_start[0]

    n_start = Node(start, None,  g=0, h=h_start, w=weight)
    open_list.put((n_start.f, n_start))

    while not open_list.empty() and expansions < T:
        generated = []
        batch_expansions = 0

        while not open_list.empty() and batch_expansions < batch_size and expansions < T:
            _, current_node = open_list.get()
            s, g, p, f = current_node.state, current_node.g, current_node.parent, current_node.f
            expansions += 1
            batch_expansions += 1
            if not generated:
                LB = max(f, LB)
            if s.is_goal():
                if UB > g:
                    UB, nuB = g, current_node
                continue
            for neighbor in s.get_neighbors():
                neighbor_state, neighbor_cost = neighbor
                g_neighbor = g + neighbor_cost
                # key_neighbor = tuple(neighbor_state.get_state_as_list())
                key_neighbor = neighbor_state
                if key_neighbor not in closed_dict or g_neighbor < closed_dict[key_neighbor]:
                    closed_dict[key_neighbor] = g_neighbor
                    generated.append((neighbor_state, g_neighbor, current_node))
        if LB >= UB:
            break
        states_list = [state for state, _, _ in generated]
        if not states_list:
            continue
        h_list = heuristic_function(states_list)
        n = len(generated)
        for i in range(n):
            s, g, p = generated[i]
            h = h_list[i]
            ns = Node(s, p, g, h, weight)
            open_list.put((ns.f, ns))

    if nuB is None:
        return None, T
    node = nuB
    path = []
    while node:
        s, g, p, f = node.state, node.g, node.parent, node.f
        s_lst = s.get_state_as_list()
        path.append(s_lst)
        node = p

    return path[::-1], expansions

