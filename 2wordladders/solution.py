#! /bin/env python

import sys
import time
from collections import deque


def main():
    # Read input
    N, Q = [int(i) for i in input().split()]
    words = [input() for _ in range(N)]
    index_map = {w: i for i, w in enumerate(words)}
    queries = [input().split() for _ in range(Q)]
    tic1 = time.perf_counter()
    g = graph(words, index_map)
    tic2 = time.perf_counter()
    s = 0
    for query in queries:
        tic = time.perf_counter()
        print(process_query(query, g, index_map))
        toc = time.perf_counter()
        s += toc - tic

    print(f"Graph construction time: {tic2 - tic1}", file=sys.stderr)
    print(f"Total process time: {s}", file=sys.stderr)


# Construct graph
def graph(words, index_map):
    adj_list = [None] * len(words)
    for w in words:
        w_i = index_map[w]
        last_four = w[-4:]
        adj_list[w_i] = []
        for v in words:
            v_i = index_map[v]
            u = list(v)
            if v_i != w_i:
                for l in last_four:
                    if l in u:
                        u.remove(l)
            if len(u) == 1:
                adj_list[w_i].append(v_i)
    return adj_list


def process_query(query, g, index_map):
    start_i = index_map[query[0]]
    finish_i = index_map[query[1]]
    par_queue = deque()
    par_queue.append(start_i)
    child_queue = deque()
    curr = None
    depth = 0
    visited = set()
    while len(par_queue) != 0:
        curr = par_queue.pop()
        if curr not in visited:
            if curr == finish_i:
                return depth
            else:
                for c in g[curr]:
                    child_queue.append(c)
            visited.add(curr)
        if len(par_queue) == 0:
            depth += 1
            par_queue = child_queue
            child_queue = deque()
    return "Impossible"


main()
