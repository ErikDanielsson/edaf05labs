#! /bin/env python3
import sys
import time
from queue import SimpleQueue


def main():
    tic1 = time.perf_counter()
    N, lines = read_input()

    comp = [None] * N
    stud = [None] * N

    for line in lines:
        i = line[0]
        if comp[i] is None:
            comp[i] = [None] * N
            # Store company preferences as inverse permutation
            for j in range(N):
                h = line[j + 1]
                comp[i][h] = j
        else:
            stud[i] = SimpleQueue()
            for j in line[1:]:
                stud[i].put(j)
    toc1 = time.perf_counter()

    tic2 = time.perf_counter()
    matching = galeshapley(comp, stud, N)
    toc2 = time.perf_counter()

    for s in matching:
        # Convert to correct format again and print
        print(s + 1)
        pass
    print(f"Read time: {toc1 - tic1}", file=sys.stderr)
    print(f"Match time: {toc2 - tic2}", file=sys.stderr)


def read_input():
    nums = []
    for line in sys.stdin:
        # Use zero index notation
        nums.extend([int(i) - 1 for i in line.split()])
    # Reformat
    N = nums[0] + 1
    lines = []
    for i in range(2 * N):
        start = 1 + (N + 1) * i
        end = 1 + (N + 1) * (i + 1)
        lines.append(nums[start:end])
    return N, lines


def galeshapley(comp, stud, N):
    pairs = [-1] * N
    free_stud = SimpleQueue()
    for i in range(N):
        free_stud.put(i)
    while not free_stud.empty():
        s = free_stud.get()
        c = stud[s].get()
        if pairs[c] == -1:
            # The company does not have a student assigned currently
            pairs[c] = s
        else:
            # The company has a student assigned, check if it is preferred
            old_s = pairs[c]
            # A more preferred student will have a smaller index in the original array.
            if comp[c][old_s] > comp[c][s]:
                pairs[c] = s
                free_stud.put(old_s)
            else:
                free_stud.put(s)
    return pairs


main()
