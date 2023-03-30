#! /bin/env python
import sys


def prim(graph, N):
    hepp = Heap([(w, u, 0) for w, u in graph[0]])
    T = set()
    # V = vertices of G
    # E = edges of G
    Q = {i for i in range(2, N + 1)}
    sum = 0
    while len(Q) != 0:
        ## (Find first v in Q and u in V - Q s.t. d(v, u) is minimal, by:)
        # Find minimal e = (v, u) in E, check if v in Q and u in V - Q or vice versa.
        # Remove v or u, whichever is in Q, from Q.
        # Add e = (v, u) or e = (u, v) to T
        # Increase the sum to return
        w, u, v = hepp.get()
        for w, neighbour in graph[u]:
            if neighbour in Q:
                hepp.try_insert(w, neighbour, u)
        Q.remove(u)
        sum += w
        T.add((u, v))

    return sum


class Heap:
    def __init__(self, array=None) -> None:
        if array != None:
            self.size = len(array)
            print(f"len arr: {len(array)}")
            self.ind_arr = [None] * len(array)
            for i, (_, u, _) in enumerate(array):
                print(u)
                self.ind_arr[u] = i
            self.heap = array
            self.heap = self.heapify()
        else:
            self.heap = []
            self.ind_arr = []

    def heapify(self):
        n = len(self.array)
        k = n // 2
        while k > 1:
            self.down(k)
            k -= 1

    def get(self):
        node = self.heap[0]
        self.size -= 1
        lw, lu, lv = self.heap[0]
        self.heap[0] = self.heap[self.size]
        self.ind_arr[lu] = 0
        self.down(0)
        return node

    def up(self, i):
        par_ind = (i - 1) // 2

        pw, pu, pv = self.heap[par_ind]
        cw, cu, cv = self.heap[i]
        if i > 0 and pw > cw:
            self.heap[par_ind] = (cw, cu, cv)
            self.heap[i] = (pw, pu, pv)
            self.ind_arr[pu] = i
            self.ind_arr[cu] = par_ind
            self.up(par_ind)

    def down(self, i):
        c1_ind = 2 * i + 1
        c2_ind = 2 * i + 2

        if c1_ind < self.size:
            pw, pu, pv = self.heap[i]
            c1w, c1u, c1v = self.heap[c1_ind]
            if pw > c1w:
                self.heap[i] = (c1w, c1u, c1v)
                self.heap[c1_ind] = (pw, pu, pv)
                self.ind_arr[pu] = c1_ind
                self.ind_arr[c1u] = i
                self.down(c1_ind)
            elif c2_ind < self.size:
                c2w, c2u, c2v = self.heap[c2_ind]
                if pw > c2w:
                    self.heap[i] = (c2w, c2u, c2v)
                    self.heap[c2_ind] = (pw, pu, pv)
                    self.ind_arr[pu] = c2_ind
                    self.ind_arr[c2u] = i
                    self.down(c2_ind)

    def tryinsert(self, w, u, v):
        index = self.ind_arr[u]
        old_w = self.heap[index][0]
        if old_w > w:
            self.heap[index] = (w, u, v)
            self.up(index)


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
    print(graph)
    # queue.append((w, u, v))
    # heapq.heapify(queue)
    print(str(prim(graph, N)), end="")


main()
