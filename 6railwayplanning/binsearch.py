#! /bin/env python
import copy
import sys
import time
from collections import deque
from math import ceil

import numpy as np


def prt_err(msg):
    print(msg, file=sys.stderr)


def read_input():
    N, M, C, P = [int(n) for n in input().split()]
    graph = {}
    edges = []
    for _ in range(M):
        u, v, c = [int(n) for n in input().split()]
        edges.append((u, v))
        if u not in graph.keys():
            graph[u] = {v: c}
        else:
            graph[u][v] = c
        if v not in graph.keys():
            graph[v] = {u: c}
        else:
            graph[v][u] = c
    remove = np.zeros(P, "int64")
    for i in range(P):
        remove[i] = int(input())
    return graph, 0, N - 1, edges, remove, C


def linked_path_to_listed_path(t):
    path = []
    while True:
        path.append(t[0])
        t = t[1]
        if t == None:
            break
    path.reverse()
    return path


def find_path(graph, s, t):
    # Perform a breadth-first search to try to find a path from s to t.
    # returns None if a path can't be found, otherwise a path as a list of the nodes traversed
    child_queue = deque()
    par_queue = deque()
    visited = set()

    depth = 0
    curr = None
    par_queue.append((s, None))
    while len(par_queue) != 0:
        curr = par_queue.pop()
        if curr[0] not in visited:
            if curr[0] == t:
                path = linked_path_to_listed_path(curr)
                return path
            elif curr[0] in graph.keys():
                for c in graph[curr[0]]:
                    child_queue.append((c, curr))
            visited.add(curr[0])

        if len(par_queue) == 0:
            depth += 1
            temp = par_queue
            par_queue = child_queue
            child_queue = temp

    return None


def ford_fulkerson(graph, s, t, C):
    total_flow = 0

    # Construct G_f and flows
    G_f = copy.deepcopy(graph)
    flow_graph = {}

    # We don't need to compute the maximum flow, only that it is C or larger
    while total_flow < C:
        path = find_path(G_f, s, t)
        if path == None:
            break

        # Find the smallest room for improvement in the path, and increase the flow through the path with this value.
        smallest_delta = min(G_f[n1][n2] for n1, n2 in zip(path[:-1], path[1:]))

        # Shrink the flow in the residual graph by the smallest delta
        for n1, n2 in zip(path[:-1], path[1:]):
            G_f[n1][n2] -= smallest_delta
            G_f[n2][n1] -= smallest_delta

            if G_f[n1][n2] == 0 or G_f[n2][n1] == 0:
                G_f[n1].pop(n2)
                G_f[n2].pop(n1)

            # Create the flow graph
            if n1 not in flow_graph:
                flow_graph[n1] = set()
            flow_graph[n1].add(n2)
            if n2 not in flow_graph:
                flow_graph[n2] = set()
            flow_graph[n2].add(n1)

        total_flow += smallest_delta

    return total_flow, flow_graph


def compute_edges_and_flow(graph, edges, remove, s, t, C):
    ff_time = 0
    ff_count = 0

    # Lets do a binary search!
    upper_graph = copy.deepcopy(graph)

    lower_graph = copy.deepcopy(graph)
    for i in remove:
        u, v = edges[i]
        lower_graph[u].pop(v)
        lower_graph[v].pop(u)

    lower_flow, _ = ford_fulkerson(lower_graph, s, t, C)
    if lower_flow >= C:
        # If we can remove all edges, just compute the max flow and return
        max_flow, _ = ford_fulkerson(graph, s, t, float("inf"))
        return max_flow, len(remove)

    lower_i = len(remove)
    upper_i = 0
    it = 1
    # Now begin the binary search
    while True:
        it += 1
        mid_i = ceil((lower_i + upper_i) / 2)
        prt_err(f"Iteration {it}: {upper_i} {mid_i} {lower_i}")

        # Check the termination condition
        if mid_i == lower_i:
            # This means that the previous upper flow was the no of edges
            # we were searching for. Compute the maxflow for the upper graph
            # which will be the solution to the problem
            max_flow, _ = ford_fulkerson(upper_graph, s, t, float("inf"))
            return upper_i, max_flow

        # Construct the mid graph
        mid_graph = copy.deepcopy(upper_graph)
        for i in remove[upper_i:mid_i]:
            u, v = edges[i]
            mid_graph[u].pop(v)
            mid_graph[v].pop(u)

        # Compute the mid flow
        mid_flow, _ = ford_fulkerson(mid_graph, s, t, C)
        if mid_flow < C:
            lower_graph = mid_graph
            lower_i = mid_i
        else:
            upper_graph = mid_graph
            upper_i = mid_i


def main():
    tic = time.perf_counter()
    graph, s, t, edges, remove, c = read_input()
    toc = time.perf_counter()
    print(f"Reading time {toc - tic}", file=sys.stderr)

    tic = time.perf_counter()
    amount_of_edges, max_flow = compute_edges_and_flow(graph, edges, remove, s, t, c)
    toc = time.perf_counter()
    print(f"Total processing time {toc - tic}", file=sys.stderr)
    print(f"{amount_of_edges} {max_flow}")


main()
