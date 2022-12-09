import numpy as n
import multiprocessing


def nothing_w(x):
    x = x*2


def nothing(n_proc=10, nums=100):
    pool = multiprocessing.Pool(n_proc)
    pool.starmap(nothing_w, [(idx,) for idx in range(nums)])
