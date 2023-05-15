#! /bin/env python
import copy
import cProfile
import sys
import time
from collections import deque

import numpy as np


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
    par_queue = deque()
    par_queue.append((s, None))
    child_queue = deque()
    curr = None
    depth = 0
    visited = set()
    while len(par_queue) != 0:
        curr = par_queue.pop()
        if curr[0] not in visited:
            if curr[0] == t:
                path = linked_path_to_listed_path(curr)
                # print("Returning "+str(path),file=sys.stderr)
                return path
            else:
                if curr[0] in graph.keys():
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
        smallest_delta = float("inf")
        for n1, n2 in zip(path[:-1], path[1:]):
            delta = G_f[n1][n2]
            if delta < smallest_delta:
                smallest_delta = delta

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


def compute_edges_and_flow(graph, edges, remove, s, t, c):
    ff_time = 0
    ff_count = 0
    print(f"Number of candidates {len(remove)}", file=sys.stderr)

    # We first compute the maximum flow to initialise the flow graph
    amount_of_edges = 1
    u, v = edges[remove[0]]
    c1 = graph[u].pop(v)
    c2 = graph[v].pop(u)
    tic = time.perf_counter()
    new_flow, flow_graph = ford_fulkerson(graph, s, t, c)
    toc = time.perf_counter()
    print(f"FF time {toc - tic}", file=sys.stderr)

    for i in range(1, len(remove)):
        if i % 10 == 1:
            print(f"Iteration {i}", file=sys.stderr)

        # Now we successively remove the edges
        u, v = edges[remove[i]]
        c1 = graph[u].pop(v)
        c2 = graph[v].pop(u)

        # We only need to recompute the flow if the removed edge
        # was part of the previous flow
        if v in flow_graph and u in flow_graph[v]:
            tic = time.perf_counter()
            new_flow, flow_graph = ford_fulkerson(graph, s, t, c)
            toc = time.perf_counter()
            print(f"FF time {toc - tic}", file=sys.stderr)
            ff_time += toc - tic
            ff_count += 1
            if new_flow < c:
                break
        amount_of_edges += 1

    # Finally, we add in the last removed edge and compute the real max flow
    graph[u][v] = c1
    graph[v][u] = c2
    max_flow, _ = ford_fulkerson(graph, s, t, float("inf"))
    return amount_of_edges, max_flow


def main():
    tic = time.perf_counter()
    graph, s, t, edges, remove, c = read_input()
    org_graph = copy.deepcopy(graph)
    toc = time.perf_counter()
    print(f"Reading time {toc - tic}", file=sys.stderr)

    amount_of_edges, max_flow = compute_edges_and_flow(graph, edges, remove, s, t, c)
    print(f"{amount_of_edges} {max_flow}")


main()
