#! /bin/env python

import sys
import time
from queue import SimpleQueue


def main():
    # Read input
    N, Q = [int(i) for i in input().split()]
    words = [input() for _ in range(N)]
    queries = [input().split() for _ in range(Q)]
    tic1 = time.perf_counter()
    g = graph(words)
    tic2 = time.perf_counter()
    s = 0
    for query in queries:
        tic = time.perf_counter()
        print(process_query(query, g))
        toc = time.perf_counter()
        s += toc - tic

    print(f"Graph construction time: {tic2 - tic1}", file=sys.stderr)
    print(f"Total process time: {s}", file=sys.stderr)


# Construct graph
def graph(words):
    adj_list = {}
    for w in words:
        last_four = w[-4:]
        adj_list[w] = []
        for v in words:
            u = list(v)
            if v != w:
                for l in last_four:
                    if l in u:
                        u.remove(l)
            if len(u) == 1:
                adj_list[w].append(v)
    return adj_list


def process_query(query, g):
    start = query[0]
    finish = query[1]
    par_queue = SimpleQueue()
    par_queue.put(start)
    child_queue = SimpleQueue()
    curr = None
    depth = 0
    visited = set()
    while not (par_queue.empty() and par_queue.empty()):
        curr = par_queue.get()
        if curr not in visited:
            if curr == finish:
                return depth
            else:
                for c in g[curr]:
                    child_queue.put(c)
            visited.add(curr)
        if par_queue.empty():
            depth += 1
            par_queue = child_queue
            child_queue = SimpleQueue()
    return "Impossible"


main()
