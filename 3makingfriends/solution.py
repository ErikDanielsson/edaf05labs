#! /bin/env python
import heapq
import sys
import time


def prim(graph, N, graphic=False):
    hepp = [(w, u) for w, u in graph[0]]
    heapq.heapify(hepp)
    T = {0}
    sum = 0
    while len(T) != N:
        # Find the next edge from the queue
        w, v = heapq.heappop(hepp)
        # If the connected node is not in T, add the edge to T and increase sum
        if v not in T:
            for neighbour_weight, neighbour in graph[v]:
                heapq.heappush(hepp, (neighbour_weight, neighbour))
            T.add(v)
            sum += w
    return sum


def main():
    tic1 = time.perf_counter()
    N, M = [int(i) for i in input().split()]
    graph = [None] * N
    for line in sys.stdin:
        u, v, w = [int(i) for i in line.split()]
        # Create the graph adjacency list
        if graph[u - 1] is None:
            graph[u - 1] = [(w, v - 1)]
        else:
            graph[u - 1].append((w, v - 1))
        if graph[v - 1] is None:
            graph[v - 1] = [(w, u - 1)]
        else:
            graph[v - 1].append((w, u - 1))
    toc1 = time.perf_counter()

    tic2 = time.perf_counter()
    s = prim(graph, N, True)
    toc2 = time.perf_counter()

    print(s)
    print(f"Graph creation time: {toc1 - tic1}", file=sys.stderr)
    print(f"Tree creation time: {toc2 - tic2}", file=sys.stderr)


main()
