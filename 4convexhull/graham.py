#! /bin/env python
import cProfile
import functools
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

is_integer = True


def read_data():
    global is_integer
    _, N = input().split()
    N = int(N)
    points = []
    min_y = float("inf")
    curr_x = float("inf")
    p0_index = 0
    for i, line in enumerate(sys.stdin):
        coords = line.split(" ")
        x = float.fromhex(coords[0])
        y = float.fromhex(coords[1])
        is_integer = is_integer and x.is_integer() and y.is_integer()
        points.append((x, y))
        if y < min_y or (y == min_y and x < curr_x):
            min_y = y
            curr_x = x
            p0_index = i
    return N, points, p0_index


def cross_product_sign(p1, p2):
    return -np.sign(p1[0] * p2[1] - p1[1] * p2[0])


def angle_comparator(p1, p2):
    sign = cross_product_sign(p1, p2)
    if sign == 0.0:
        return np.linalg.norm(p1) - np.linalg.norm(p2)
    return sign


def left_turn(p1, p2, p3):
    return cross_product_sign(p3 - p1, p2 - p1) > 0


def graham(N, points, p0_index):
    p0 = np.array(points.pop(p0_index))
    norm_ps = np.array(points) - p0
    norm_ps = np.array(sorted(norm_ps, key=functools.cmp_to_key(angle_comparator)))
    hull = np.zeros((N, 2))
    hull[0, :] = np.zeros(2)
    start_i = 0
    while cross_product_sign(norm_ps[start_i, :], norm_ps[start_i + 1, :]) == 0:
        start_i += 1
    hull[1, :] = norm_ps[start_i, :]
    hull[2, :] = norm_ps[start_i + 1, :]
    hindex = 2
    for p in norm_ps[start_i + 2 :, :]:
        while not left_turn(hull[hindex - 1], hull[hindex], p):
            hindex -= 1
        hindex += 1
        hull[hindex] = p
    points.append(p0)
    hull += p0
    return hull[: hindex + 1, :], points, p0


# Reorders the hull such that the rightmost point is at the beginning of the array.
# Hull then goes clockwise.
def reorder_hull(graham_hull):
    x_max = -float("inf")
    right_index = 0
    m = len(graham_hull)
    graham_hull = graham_hull[::-1, :]
    for i in range(m):
        if graham_hull[i][0] > x_max:
            x_max = graham_hull[i][0]
            right_index = i

    ordered_hull = np.concatenate(
        (graham_hull[right_index:], graham_hull[:right_index])
    )
    return ordered_hull


def main(plot=False):
    print("GRAHAM SCAN", file=sys.stderr)
    tic = time.perf_counter()
    N, points, p0_index = read_data()
    toc = time.perf_counter()
    print(f"    Reading time {toc - tic}", file=sys.stderr)
    tic = time.perf_counter()
    hull, points, p0 = graham(N, points, p0_index)
    toc = time.perf_counter()
    print(f"    Graham time {toc - tic}", file=sys.stderr)
    if plot:
        plot_points(np.array(points))
        plot_hull(hull, p0)
        plt.show()

    # Find rightmost point and reorder
    tic = time.perf_counter()
    hull = reorder_hull(hull)
    toc = time.perf_counter()
    print(f"    Order time {toc - tic}", file=sys.stderr)
    print(len(hull))
    if is_integer:
        hull = hull.astype("int")
        for point in hull[:, :]:
            print(f"{point[0]:d} {point[1]:d}")
    else:
        for point in hull[:, :]:
            print(f"{point[0]:.3f} {point[1]:.3f}")


main()


def plot_points(points):
    N = points.shape[0]
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=1)
    if len(points) < 20:
        for i in range(N):
            ax.annotate(f"p{i}", points[i, :])
    return fig, ax


def plot_hull(hull, p0):
    plt.scatter(p0[0], p0[1])
    return plt.plot(hull[:, 0], hull[:, 1], marker="o", color="red")
