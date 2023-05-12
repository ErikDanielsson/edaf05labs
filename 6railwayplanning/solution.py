#! /bin/env python
import sys

import numpy as np


def read_input():
    N, M, C, P = [int(n) for n in input().split()]
    graph = {}
    for _ in range(M):
        u, v, c = [int(n) for n in input().split()]
        graph[(u, v)] = c
        graph[(v, u)] = c
    remove = np.zeros(P, "int64")
    for i in range(P):
        remove[i] = int(input())
    return graph, remove


def main():
    graph, remove = read_input()
    print(graph)
    print(remove)


main()
