#! /bin/env python
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

is_integer = True


def read_data():
    global is_integer
    _, N = input().split()
    N = int(N)
    points = np.zeros((N, 2))
    min_y = float("inf")
    curr_x = float("inf")
    p0_index = 0
    for i, line in enumerate(sys.stdin):
        coords = line.split(" ")
        x = float.fromhex(coords[0])
        y = float.fromhex(coords[1])
        is_integer = is_integer and x.is_integer() and y.is_integer()
        points[i, :] = [x, y]
        if y < min_y or (y == min_y and x < curr_x):
            min_y = y
            curr_x = x
            p0_index = i
    return N, points, p0_index


def slope(p, q):
    if p[0] == q[0]:
        return np.sign(p[1] - q[1]) * float("inf")
    return (p[1] - q[1]) / (p[0] - q[0])


def compute_slope(p):
    # print(f"Compute slope {p}", file=sys.stderr)
    n = p.shape[0]
    alfa = np.zeros(n)
    for i in range(n):
        alfa[i] = slope(p[i], p[(i + 1) % n])
    return alfa


def cross_product_sign(p1, p2):
    return -np.sign(p1[0] * p2[1] - p1[1] * p2[0])


def left_turn(p1, p2, p3):
    return cross_product_sign(p3 - p1, p2 - p1) > 0


def add(k, fr, to, q, p, n):
    j = fr
    while True:
        q[k] = p[j]
        k += 1
        if j == to:
            return k
        j = (j + 1) % n


def line_segment(p, q, r):
    print(f"LINE SEGMENT: {p}, {q}, {r}")
    u = p - q
    v = r - q
    c = cross_product_sign(u, v)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    return c == 0 and uv > 0 and uv < vv


def include_points(q, p, j, n):
    print("INCLUDE points")
    print(q)
    print(p)
    print(n)
    i = 0
    for k in range(n):
        u = (n + k + j - 1) % n
        v = (n + k + j) % n
        w = (n + k + j + 1) % n
        if not line_segment(q[v, :], q[u, :], q[w, :]):
            print("HAJ")
            p[i, :] = q[v, :]
            i += 1
    return i


def leftmost(p):
    n = p.shape[0]
    i = 0
    for k in range(n):
        if p[k, 0] < p[i, 0]:
            i = k
    if i == 0:
        i = 1
    return i


def dc(p):
    print(f"DC {p}")
    n = p.shape[0]
    if n <= 3:
        return convex_hull_base_case(p)
    n_a = n // 2
    # While
    while p[n_a - 1, 1] == p[n_a, 1]:
        n_a += 1  # This may be a problem if all points have the same y.
    n_b = n - n_a
    q = np.zeros((n, 2))
    a = p[:n_a, :]
    b = p[n_a:n, :]
    n_a, i_l = dc(a)
    n_b, j_l = dc(b)
    print(f"a: {a}")
    print(f"b: {b}")
    alfa = compute_slope(a)
    beta = compute_slope(b)
    n = 0

    # Left and rightmost x
    L_a = a[i_l, 0]
    L_b = b[j_l, 0]
    R_a = a[0, 0]
    R_b = b[0, 0]

    # Four cases
    if L_a < L_b:
        if R_a < R_b:
            # Case 1
            ia_R, ia_L, ja_L, ja_R = case_1(a, n_a, b, n_b, alfa, beta, i_l, j_l)
        else:
            # Case 2
            ia_R, ia_L, ja_L, ja_R = case_2(a, n_a, b, n_b, alfa, beta, i_l, j_l)
    else:
        if R_a < R_b:
            # Case 3
            ia_R, ia_L, ja_L, ja_R = case_3(a, n_a, b, n_b, alfa, beta, i_l, j_l)
        else:
            # Case 4
            ia_R, ia_L, ja_L, ja_R = case_4(a, n_a, b, n_b, alfa, beta, i_l, j_l)
    print("CASE DONE")
    n = add(0, ia_R, ia_L, q, a, n_a)
    print(n)
    n = add(n, ja_R, ja_L, q, b, n_b)
    print(n)

    j = 0
    for k in range(n):
        if q[k, 0] > q[j, 0] or (q[k, 0] == q[j, 0] and q[k, 1] > q[j, 1]):
            j = k
    n = include_points(p, q, j, n)
    l_i = -1
    leftmost = float("inf")
    for i in range(n):
        if p[i, 0] < leftmost:
            l_i = i
            leftmost = p[i, 0]
    return (n, l_i)


def convex_hull_base_case(p):
    print(f"BASECASE {p}")
    # Inga specialfall än mannen
    # De e chill.
    rightmost = np.array([-float("inf"), float("inf")])
    leftmost = float("inf")
    r_i = -1
    l_i = -1
    for i, (x, y) in enumerate(p):
        if x > rightmost[0] or x == rightmost[0] and y > rightmost[1]:
            rightmost[:] = (x, y)
            r_i = i
        if x < leftmost:
            leftmost = x
            l_i = i

    p_r = np.copy(p[r_i, :])
    p_l = np.copy(p[l_i, :])

    if p.shape[0] == 2:
        print(p_r, r_i)
        print(p_l, l_i)
        p[0, :] = p_r
        p[1, :] = p_l
        return 2, 1

    indices = set([0, 1, 2])
    indices.remove(r_i)
    indices.remove(l_i)
    i_third = indices.pop()
    p_third = np.copy(p[i_third, :])
    if left_turn(p_r, p_third, p_l):
        print("LEFT", file=sys.stderr)
        p[0, :] = p_r
        p[1, :] = p_l
        p[2, :] = p_third
        return 3, 1
    else:
        print("RIGHT", file=sys.stderr)
        p[0, :] = p_r
        p[1, :] = p_third
        p[2, :] = p_l
        return 3, 2


def preparata_hong(points):
    ind = np.lexsort((points[:, 0], points[:, 1]))
    temp = [(points[i, 0], points[i, 1]) for i in ind]
    for i, pair in enumerate(temp):
        points[i, :] = pair
    n, i = dc(points)
    hull = points[:n, :]
    return hull


# Case 1
def case_1(a, n_a, b, n_b, alfa, beta, i_l, j_l):
    print("CASE 1", file=sys.stderr)
    i = 0
    j = 0
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        if (alfa[i] > gamma_ij or alfa[i] == -float("inf")) and i < i_l:
            i += 1
        elif (beta[j] > gamma_ij or beta[j] == -float("inf")) and j < j_l:
            j += 1
        else:
            break
    i_r_new = i
    j_r_new = j
    i = i_l
    j = j_l
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        if beta[j] > gamma_ij and j != 0:
            j = (j + 1) % n_b
        elif alfa[i] > gamma_ij and i != 0:
            i = (i + 1) % n_a
        else:
            break
    i_l_new = i
    j_l_new = j
    return i_r_new, i_l_new, j_r_new, j_l_new


# Case 2
def case_2(a, n_a, b, n_b, alfa, beta, i_l, j_l):
    print("CASE 2", file=sys.stderr)
    i = 0
    j = 0
    while True:
        gamma_ij = slope(a[i, :], b[i, :])
        if (alfa[i] > gamma_ij or alfa[i] == -float("inf")) and i < i_l:
            i += 1
        elif (beta[j] > gamma_ij or beta[j] == -float("inf")) and j < j_l:
            j += 1
        else:
            break
    i_r_new = i
    j_r_new = j
    i = i_l
    j = j_l
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        a_k = (n_a + i - 1) % n_a
        b_k = (n_b + j - 1) % n_b
        if np.isfinite(alfa[a_k]) and alfa[a_k] > gamma_ij and i != 0:
            i = a_k
        elif beta[b_k] > gamma_ij and j != 0:
            j = b_k
        else:
            break
    i_l_new = i
    j_l_new = j
    return i_r_new, i_l_new, j_r_new, j_l_new


# Case 3
def case_3(a, n_a, b, n_b, alfa, beta, i_L, j_L):
    print("CASE 3", file=sys.stderr)
    i = 0
    j = 0
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        a_k = (n_a + i - 1) % n_a
        b_k = (n_b + j - 1) % n_b
        if beta[b_k] > gamma_ij and j < j_L:
            j = b_k
        elif alfa[a_k] > gamma_ij and i < i_L:
            i = a_k
        else:
            break
    i_r_new = i
    j_r_new = j
    i = i_L
    j = j_L
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        if beta[j] > gamma_ij and j != 0:
            j = (j + 1) % n_b
        elif alfa[i] > gamma_ij and i != 0:
            i = (i + 1) % n_a
        else:
            break
    i_l_new = i
    j_l_new = j
    return i_r_new, i_l_new, j_r_new, j_l_new


# Case 4
def case_4(a, n_a, b, n_b, alfa, beta, i_l, j_l):
    print("CASE 4", file=sys.stderr)
    i = 0
    j = 0
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        a_k = (n_a + i - 1) % n_a
        b_k = (n_b + j - 1) % n_b
        if beta[b_k] > gamma_ij and j < j_l:
            j = b_k
        elif alfa[a_k] > gamma_ij and i < i_l:
            i = a_k
        else:
            break
    i_r_new = i
    j_r_new = j
    i = i_l
    j = j_l
    while True:
        gamma_ij = slope(a[i, :], b[j, :])
        a_k = (n_a + i - 1) % n_a
        b_k = (n_b + j - 1) % n_b
        if np.isfinite(alfa[a_k]) and alfa[a_k] > gamma_ij and i != 0:
            i = a_k
        elif np.isfinite(beta[b_k]) and beta[b_k] > gamma_ij and j != 0:
            j = b_k
        else:
            break
    i_l_new = i
    j_l_new = j
    return i_r_new, i_l_new, j_r_new, j_l_new


def print_hull(hull):
    print(hull.shape[0])
    if is_integer:
        hull = hull.astype("int")
        for point in hull[:, :]:
            print(f"{point[0]:d} {point[1]:d}")
    else:
        for point in hull[:, :]:
            print(f"{point[0]:.3f} {point[1]:.3f}")


def plot_points(points):
    N = points.shape[0]
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=1)
    if len(points) < 20:
        for i in range(N):
            ax.annotate(f"p{i}", points[i, :])
    return fig, ax


def plot_hull(hull):
    return plt.plot(hull[:, 0], hull[:, 1], marker="o", color="red")


def main(plot=True):
    print("PREPARATA-HONG", file=sys.stderr)
    tic = time.perf_counter()
    N, points, p0_index = read_data()
    toc = time.perf_counter()
    print(f"    Reading time: {toc - tic}", file=sys.stderr)
    tic = time.perf_counter()
    hull = preparata_hong(points)
    toc = time.perf_counter()
    print(f"    Preparata-Hong time: {toc - tic}", file=sys.stderr)
    print_hull(hull)
    if plot:
        plot_points(np.array(points))
        plot_hull(hull)
        plt.show()


main()
