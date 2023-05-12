#! /bin/env python
import sys

import numpy as np


def read_input():
    chars = {c: i for i, c in enumerate(input().split())}
    k = len(chars)
    costs = np.zeros((k, k))
    for i in range(k):
        costs[i, :] = [int(i) for i in input().split()]
    Q = int(input())
    queries = []
    for _ in range(Q):
        fst, snd = input().split()
        queries.append((fst, snd))
    return chars, costs, queries


def align(fst, snd, chars, costs, delta):
    # Align two string using the Needleman-Wunsch algorithm
    num_fst = [chars[c] for c in fst]
    num_snd = [chars[c] for c in snd]
    n = len(fst)
    m = len(snd)
    opt = np.zeros((n + 1, m + 1))
    opt[:, 0] = [i * delta for i in range(n + 1)]
    opt[0, :] = [i * delta for i in range(m + 1)]
    min_val = float("inf")
    # Compute the opt matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c1 = num_fst[i - 1]
            c2 = num_snd[j - 1]
            alfa = costs[c1, c2]
            match = alfa + opt[i - 1, j - 1]
            insert = delta + opt[i - 1, j]
            delete = delta + opt[i, j - 1]
            opt[i, j] = max(match, insert, delete)

    # Now compute the alignments
    fst_align = []
    snd_align = []
    i = n
    j = m
    while i > 0 or j > 0:
        c1 = num_fst[i - 1]
        c2 = num_snd[j - 1]
        if i > 0 and j > 0 and opt[i, j] == opt[i - 1, j - 1] + costs[c1, c2]:
            fst_align.append(fst[i - 1])
            snd_align.append(snd[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and opt[i, j] == opt[i - 1, j] + delta:
            fst_align.append(fst[i - 1])
            snd_align.append("*")
            i -= 1
        else:
            fst_align.append("*")
            snd_align.append(snd[j - 1])
            j -= 1

    return "".join(reversed(fst_align)), "".join(reversed(snd_align))


def main():
    chars, costs, queries = read_input()
    delta = -4
    alignments = []
    for fst, snd in queries:
        alignments.append(align(fst, snd, chars, costs, delta))
    for fst_align, snd_align in alignments:
        print(f"{fst_align} {snd_align}")


main()
