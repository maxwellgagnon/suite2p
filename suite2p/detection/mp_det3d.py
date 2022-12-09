import numpy as n
import multiprocessing
from scipy.ndimage import uniform_filter
import cProfile
from ..suite3d import utils as utils3d

def filtframe_shmem_w(in_par, out_par, idxs, size, c1):
    shin, mov_in = utils3d.load_shmem(in_par)
    shout, mov_out = utils3d.load_shmem(out_par)
    for idx in idxs:
        mov_out[idx] = mov_in[idx] - (uniform_filter(mov_in[idx], size=size, mode='constant') / c1)

def filtframe_shmem(mov, size, n_proc, batch_size=50):
    nt, Lz, Ly, Lx = mov.shape
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')
    shmem, shpar, shmov = utils3d.create_shmem_from_arr(mov, copy=True)
    shmem_out, shpar_out, shmov_out = utils3d.create_shmem_from_arr(mov, copy=True)

    batches = [n.arange(idx, min(nt,idx+batch_size)) for idx in n.arange(0,nt, batch_size)]
    pool = multiprocessing.Pool(n_proc)
    pool.starmap(filtframe_shmem_w, [(shpar, shpar_out, b,size,c1) for b in batches])

def filtframe_shmem_2(shmem_in, shmem_out, size, n_proc, batch_size=50):
    nt, Lz, Ly, Lx = shmem_in['shape']
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')

    batches = [n.arange(idx, min(nt,idx+batch_size)) for idx in n.arange(0,nt, batch_size)]
    pool = multiprocessing.Pool(n_proc)
    pool.starmap(filtframe_shmem_w, [(shmem_in, shmem_out, b,size,c1) for b in batches])


def filtframe_w(frame, size, mode, c1):
    # cp = cProfile.Profile()
    # cp.enable()
    minus = (uniform_filter(frame, size=size, mode=mode) / c1)
    ret = frame - minus
    # cp.disable()
    # cp.print_stats(sort='cumtime')
    return ret


def filtframe_mp(mov, size, n_proc):
    cp = cProfile.Profile()
    cp.enable()
    nt, Lz, Ly, Lx = mov.shape
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')
    movt = []
    pool = multiprocessing.Pool(n_proc)
    movt = pool.starmap(
        filtframe_w, [(mov[i], size, 'constant', c1) for i in range(nt)])
    cp.disable()
    cp.print_stats(sort='cumtime')
    return movt


def filtframe(mov, size):
    cp = cProfile.Profile()
    cp.enable()
    nt, Lz, Ly, Lx = mov.shape
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')
    movt = []
    movt = [filtframe_w(mov[i], size, 'constant', c1) for i in range(nt)]
    cp.disable()
    cp.print_stats(sort='cumtime')
    return movt


def worker(globs,idxs, size, c1):
    print(idxs)
    for idx in idxs:
        frame = globs['in'][idx]
        globs['out'][idx] = (uniform_filter(frame, size=size, mode='constant') / c1)
    print('done', idxs)

def filtframe_globals(mov, size, n_procs = 10, batch_size=10):
    manager = multiprocessing.Manager()
    globs = manager.dict()
    nt, Lz, Ly, Lx = mov.shape
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')
    globs['in'] = mov
    globs['out'] = n.zeros_like(mov)
    idxs = n.arange(0, nt, batch_size)
    pool = multiprocessing.Pool(n_procs)
    pool.starmap(worker, [(globs,n.arange(i, min(nt-1,i+batch_size)), size, c1)  for i in idxs])
    return globs['out']


def wrap_filtframe(mov, size):
    cp = cProfile.Profile()
    cp.enable()
    pool = multiprocessing.Pool(2)
    movts = pool.starmap(filtframe, [(mov[:50], size), (mov[50:], size)])
    cp.disable()
    cp.print_stats(sort='cumtime')
    return movts