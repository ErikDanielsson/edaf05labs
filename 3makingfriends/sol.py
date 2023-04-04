#! /bin/env python
import heapq
import sys

import matplotlib.pyplot as plt
import networkx as nx


def prim(graph, N, graphic=False):
    hepp = [(w, 0, u) for w, u in graph[0]]
    heapq.heapify(hepp)
    T = set()
    visited = {0}
    # V = vertices of G
    # E = edges of G
    sum = 0
    if graphic:
        labels = []
        g = nx.Graph()
        fig, ax = plt.subplots()
        for u, neighbours in enumerate(graph):
            for w, v in neighbours:
                g.add_edge(u, v, color="white", weight=w)
                labels.append((u, v, w))
        pos = nx.kamada_kawai_layout(g)

    while len(visited) != N:
        ## (Find first v in Q and u in V - Q s.t. d(v, u) is minimal, by:)
        # Find minimal e = (v, u) in E, check if v in Q and u in V - Q or vice versa.
        # Remove v or u, whichever is in Q, from Q.
        # Add e = (v, u) or e = (u, v) to T
        # Increase the sum to return
        w, u, v = heapq.heappop(hepp)
        if v not in visited:
            for nw, neighbour in graph[v]:
                heapq.heappush(hepp, (nw, v, neighbour))
            visited.add(v)
            sum += w
            T.add((u, v))
            if graphic:
                ax.clear()
                nx.draw_networkx_edges(
                    g,
                    alpha=[
                        1 if (u, v) in T or (v, u) in T else 0.1 for u, v in g.edges
                    ],
                    edge_color=[
                        "red" if (u, v) in T or (v, u) in T else "grey"
                        for u, v in g.edges
                    ],
                    width=[2 if (u, v) in T or (v, u) in T else 1 for u, v in g.edges],
                    ax=ax,
                    pos=pos,
                )

                nx.draw_networkx_nodes(
                    g, pos=pos, node_size=20, node_color=[(0, 0, 0)] * N
                )
                plt.pause(0.2)
    # if graphic:
    #   nx.draw(g, with_labels=True)
    #  plt.show()

    return sum


def main():
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
    # queue.append((w, u, v))
    # heapq.heapify(queue)
    s = prim(graph, N, True)

    print(s)


main()
