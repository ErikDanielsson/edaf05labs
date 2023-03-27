#! /bin/env python3
import sys


def main():
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
            stud[i] = line[1:]

    matching = galeshapley(comp, stud, N)

    for s in matching:
        # Convert to correct format again
        print(s + 1)


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
    free_stud = [i for i in range(N)]
    while len(free_stud) > 0:
        s = free_stud.pop(0)
        c = stud[s].pop(0)
        if pairs[c] == -1:
            pairs[c] = s
        else:
            old_s = pairs[c]
            if comp[c][old_s] > comp[c][s]:
                pairs[c] = s
                free_stud.append(old_s)
            else:
                free_stud.append(s)
    return pairs


main()
