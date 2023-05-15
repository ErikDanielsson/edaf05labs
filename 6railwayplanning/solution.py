#! /bin/env python
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
    G_f = {}
    # Construct G_f and flows
    smallest_delta = float("inf")
    # print("The graph now looks like: "+str(graph),file=sys.stderr)
    for u in graph.keys():
        for v in graph[u].keys():
            # We add the possible flow increase (the capacity) to both uv and vu in G_f.
            if u not in G_f.keys():
                G_f[u] = {v: graph[u][v]}
            else:
                G_f[u][v] = graph[u][v]
            if v not in G_f.keys():
                G_f[v] = {u: graph[u][v]}
            else:
                G_f[v][u] = graph[u][v]

    while True:
        path = find_path(G_f, s, t)
        # print("Found path "+str(path),file=sys.stderr)
        if path == None:
            break
        # Find the smallest room for improvement in the path, and increase the flow through the path with this value.
        smallest_delta = float("inf")
        for n1, n2 in zip(path[:-1], path[1:]):
            delta = G_f[n1][n2]
            # print("Delta from "+str(n1)+" to "+str(n2)+" is: "+str(delta),file=sys.stderr)
            if delta <= smallest_delta:
                smallest_delta = delta
        # print("The smallest delta was "+str(smallest_delta),file=sys.stderr)
        for n1, n2 in zip(path[:-1], path[1:]):
            G_f[n1][n2] -= smallest_delta
            G_f[n2][n1] -= smallest_delta
            if G_f[n1][n2] == 0 or G_f[n2][n1] == 0:
                G_f[n1].pop(n2)
                G_f[n2].pop(n1)
        total_flow += smallest_delta
    return total_flow


def residual_graph(graph):
    return None


def main():
    tic = time.perf_counter()
    graph, s, t, edges, remove, c = read_input()
    toc = time.perf_counter()
    print(f"Reading time {toc - tic}", file=sys.stderr)
    # print("Graph of capacities",str(graph),file=sys.stderr)
    # print("Edges to remove",str(remove),file=sys.stderr)
    # We want to return the amount of the edge that we can remove, and then
    # edges 0 1 2
    # SUper inefficient solution comes here.
    ff_time = 0
    ff_count = 0
    amount_of_edges = 0
    for index in remove:
        u, v = edges[index]
        graph[u].pop(v)
        graph[v].pop(u)
        # print("Removed edge (" + str(u) + "," + str(v) + ")", file=sys.stderr)
        tic = time.perf_counter()
        new_flow = ford_fulkerson(graph, s, t, c)
        toc = time.perf_counter()
        ff_time += toc - tic
        ff_count += 1
        if new_flow < c:
            break
        amount_of_edges += 1
        max_flow = new_flow
    print(f"Average FF time {ff_time / ff_count}", file=sys.stderr)
    print(str(amount_of_edges) + " " + str(max_flow))


main()
