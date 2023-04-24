#! /bin/env python
import sys

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
    return (p[1] - q[1]) / (p[0] - q[0])


def compute_slope(p):
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
        if j == to:
            return k
        j = (j + 1) % n


def line_segment(p_r, p_s, p_t):
    v0 = p_t - p_s
    vc = p_r - p_s
    v0_mag = np.linalg.norm(v0)
    vc_mag = np.linalg.norm(vc)
    return v0 / v0_mag == vc / vc_mag and vc_mag < v0_mag


def include_points(q, p, j, n):
    i = 0
    for k in range(n):
        u = (n + k + j - 1) % n
        v = (n + k + j - 1) % n
        w = (n + k + j - 1) % n
        if not line_segment(q[v, :], q[u, :], q[w, :]):
            p[i, :] = q[v, :]
            i += 1
    return i


def dc(p):
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
            ia_R, ia_L, ja_L, ja_R = case_1(a, n_a, b, n_b, alfa, beta, i_l, j_l)
            # Case 1
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
    n = add(0, ia_R, ia_L, q, a, n_a)
    n = add(n, ja_R, ja_L, q, b, n_b)

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
    # Inga specialfall än mannen
    # De e chill.
    rightmost = -float("inf")
    leftmost = float("inf")
    r_i = -1
    l_i = -1
    for i, (x, _) in enumerate(p):
        if x > rightmost:
            rightmost = x
            r_i = i
        if x < leftmost:
            leftmost = x
            l_i = i

    p_r = p[r_i, :]
    p_l = p[l_i, :]

    if p.shape[0] == 2:
        return np.array([p_r, p_l]), 1

    indices = set([0, 1, 2])
    indices.remove(r_i)
    indices.remove(l_i)
    p_third = indices.pop()
    if left_turn(p_r, p_third, p_l):
        return np.array([p_r, p_l, p_third]), 1
    else:
        return np.array([p_r, p_third, p_l]), 2


def preparata_hong(points):
    points = np.sort(points)
    p, i = dc(points)
    n = len(p)
    print(n)
    print(points)
    return points[:n, :], i


# Case 1
def case_1(a, n_a, b, n_b, alfa, beta, i_l, j_l):
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


def main():
    N, points, p0_index = read_data()
    hull = preparata_hong(points)
    print(hull)


main()
