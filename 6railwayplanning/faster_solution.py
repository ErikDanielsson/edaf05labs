#! /bin/env python
import sys

import numpy as np
from collections import deque

class node:
    def __init__(self,value,parent=None):
        self.value = value
        self.parent = parent


def read_input():
    N, M, C, P = [int(n) for n in input().split()]
    graph = {}
    edges = []
    for _ in range(M):
        u, v, c = [int(n) for n in input().split()]
        edges.append((u,v))
        if u not in graph.keys():
            graph[u] = {v:c}
        else:
            graph[u][v] = c
        if v not in graph.keys():
            graph[v] = {u:c}
        else:
            graph[v][u] = c
    remove = np.zeros(P, "int64")
    for i in range(P):
        remove[i] = int(input())
    return graph, 0, N-1, edges, remove, C

def linked_path_to_listed_path(t):
    path = []
    while True:
        path.append(t.value)
        t = t.parent
        if t == None:
            break
    path.reverse()
    return path


def find_path(graph, s, t):
    # Perform a breadth-first search to try to find a path from s to t.
    # returns None if a path can't be found, otherwise a path as a list of the nodes traversed
    par_queue = deque()
    par_queue.append(node(s))
    child_queue = deque()
    curr = None
    depth = 0
    visited = set()
    while len(par_queue) != 0:
        curr = par_queue.pop()
        if curr.value not in visited:
            if curr.value == t:
                path = linked_path_to_listed_path(curr)
                #print("Returning "+str(path),file=sys.stderr)
                return path
            else:
                if curr.value in graph.keys():
                    for c in graph[curr.value]:
                        if graph[curr.value][c] > 0:
                            child_queue.append(node(c,parent=curr))
            visited.add(curr.value)
        if len(par_queue) == 0:
            depth += 1
            par_queue = child_queue
            child_queue = deque()
    return None

# Given a residual graph we can find the maximum flow from s to t.
# This version also returns a path which ensures that the flow is greater than C.
def find_max_flow(G_f,s,t,C):
    print("Trying to find max flow of the graph "+str(G_f))
    flow = 0
    safe_path = None
    while True:
        path = find_path(G_f, s, t)
        #print("Found path "+str(path),file=sys.stderr)
        if path == None:
            break
        # Find the smallest room for improvement in the path, and increase the flow through the path with this value.
        smallest_delta = float("inf")
        for i in range(len(path)-1):
            n1 = path[i]
            n2 = path[i+1]
            delta = G_f[n1][n2]
            #print("Delta from "+str(n1)+" to "+str(n2)+" is: "+str(delta),file=sys.stderr)
            if delta <= smallest_delta:
                smallest_delta = delta
        #print("The smallest delta was "+str(smallest_delta),file=sys.stderr)
        for i in range(len(path)-1):
            n1 = path[i]
            n2 = path[i+1]
            G_f[n1][n2] -= smallest_delta
            G_f[n2][n1] -= smallest_delta
            #if G_f[n1][n2] == 0 or G_f[n2][n1] == 0:
            #    G_f[n1].pop(n2)
            #    G_f[n2].pop(n1)
        flow += smallest_delta
        if flow >= C:
            safe_path = path
    print("The flow was: "+str(flow))
    return flow, safe_path

def add_edge(G,u,v, value):
    if u not in G.keys():
        G[u] = {v:value}
    else:
        G[u][v] = value
    if v not in G.keys():
        G[v] = {u:value}
    else:
        G[v][u] = value


def remove_edges(graph,s,t,c,edges):
    total_flow = 0
    good_flow = 0
    amount_removed = 0
    # Construct initial G_f
    G_f = {}
    #print("The graph now looks like: "+str(graph),file=sys.stderr)
    for u in graph.keys():
        for v in graph[u].keys():
            # We add the possible flow increase (the capacity) to both uv and vu in G_f.
            add_edge(G_f,u,v,graph[u][v])
    print("Residual before: "+str(G_f))
    path_to_save = None
    last_good = None
    for u,v in edges:
        print("Trying to remove edge "+str(u)+str(v))
        print("Graph looks like: "+str(G_f))
        uv_val = G_f[u][v]
        vu_val = G_f[v][u]
        assert uv_val == vu_val
        G_f[u][v] = 0
        G_f[v][u] = 0
        # Check if the edge is in a path which guarantees sufficient flow
        in_path = False
        if path_to_save != None:
            for i in range(len(path_to_save)-1):
                n1 = path_to_save[i]
                n2 = path_to_save[i+1]
                if (n1,n2) == (u,v) or (n2,n1) == (u,v):
                    in_path = True
                    break
        # If the edge is not in a good path, we can safely remove it
            if not in_path:
                print("The edge was not in path, remove it immediately, go to next.")
                last_good = G_f.copy()
                print("Last good: "+str(last_good))
                continue
        # Run Ford Fulkerson on the new graph:
        total_flow, path_to_save = find_max_flow(G_f.copy(),s,t,c)
        print("Flow after removing "+str(u)+str(v)+": "+str(total_flow))
        print("Path guaranteeing sufficient flow: "+str(path_to_save))
        if total_flow < c:
            # Add back the last edge:
            add_edge(G_f,u,v,uv_val)
            good_flow, _ = find_max_flow(last_good,s,t,c)
            return amount_removed, good_flow
        amount_removed += 1
        good_flow = total_flow
    return amount_removed, good_flow


def residual_graph(graph):
    return None


def main():
    graph, s, t, edges, remove, c = read_input()
    # Better solution
    edges_to_remove = [edges[i] for i in remove]
    amount_of_edges_removed, max_flow = remove_edges(graph,s,t,c,edges_to_remove)
    print(str(amount_of_edges_removed)+" "+str(max_flow))


main()
