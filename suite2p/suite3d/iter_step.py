import os
import numpy as n
import copy
from multiprocessing import shared_memory, Pool

from ..io import lbm as lbmio 
from ..registration import register
from . import utils

from ..detection import utils as det_utils
from ..detection import detection3d as det3d

def default_log(string, val): 
    print(string)


def init_batches(tifs, batch_size, max_tifs_to_analyze=None):
    if max_tifs_to_analyze is not None and max_tifs_to_analyze > 0:
        tifs = tifs[:max_tifs_to_analyze]
    n_tifs = len(tifs)
    n_batches = int(n.ceil(n_tifs / batch_size))

    batches = []
    for i in range(n_batches):
        batches.append(tifs[i*batch_size : (i+1) * batch_size])

    return batches

def iter_dataset(tifs, params, dirs, summary, log_cb = default_log,
                    override_input_reg_bins=None):

    ref_img_3d = summary['ref_img_3d']
    crosstalk_coeff = summary['crosstalk_coeff']
    refs_and_masks = summary['refs_and_masks']
    all_ops = summary['all_ops']

    job_iter_dir = dirs['job_iter_dir']
    job_reg_data_dir = dirs['job_iter_dir']
    n_tifs_to_analyze = params['total_tifs_to_analyze']
    tif_batch_size = params['tif_batch_size']
    planes = params['planes']
    notch_filt = params['notch_filt']
    load_from_binary = params['load_from_binary']
    do_subtract_crosstalk = params['subtract_crosstalk']
    mov_dtype = params['dtype']
    temporal_high_pass_width = params['temporal_high_pass_width']
    do_running_sdmov = params['running_sdmov']
    npil_hpf_xy = params['npil_high_pass_xy']
    npil_hpf_z = params['npil_high_pass_z']
    unif_filter_xy = params['detection_unif_filter_xy']
    unif_filter_z = params['detection_unif_filter_z']
    intensity_thresh = params['intensity_threshold']

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    log_cb("Will analyze %d tifs in %d batches" % (len(n.concatenate(batches)), len(batches)),0)
    if load_from_binary:
        log_cb("Loading from saved binaries, will skip registration and crosstalk subtraction", 0)
    
    # init accumulators
    shmem_sum_img, shmem_sum_img_params, sum_img = utils.create_shmem_from_arr(ref_img_3d)
    shmem_mean_img, shmem_mean_img_params, mean_img = utils.create_shmem_from_arr(ref_img_3d.astype(mov_dtype))
    shmem_max_img,shmem_max_img_params, max_img = utils.create_shmem_from_arr(ref_img_3d)
    shmem_vmap2,shmem_vmap2_params, vmap2 = utils.create_shmem_from_arr(ref_img_3d.astype(mov_dtype))
    shmem_sdmov2,shmem_sdmov2_params, sdmov2 = utils.create_shmem_from_arr(ref_img_3d.astype(mov_dtype))
    n_frames_proc = 0
    
    batch_dirs, reg_data_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches)
    if override_input_reg_bins is not None:
        reg_data_paths = override_input_reg_bins
        n_batches = len(reg_data_paths)
        log_cb("Reading files from user-provided location, only %d batches" % n_batches, 0)

    for batch_idx in range(n_batches):
        iter_dir = batch_dirs[batch_idx]
        reg_data_path = reg_data_paths[batch_idx]
        log_cb("Loading Batch %d of %d" % (batch_idx, n_batches), 0)
        if not load_from_binary:
            tifs = batches[batch_idx]
            mov = lbmio.load_and_stitch_tifs(tifs, planes, filt = notch_filt, concat=True, log_cb=log_cb)
            shmem_mov,shmem_mov_params, mov = utils.create_shmem_from_arr(mov, copy=True)
            if do_subtract_crosstalk:
                __ = subtract_crosstalk(shmem_mov_params, crosstalk_coeff, planes = planes, log_cb = log_cb)
            log_cb("Registering Batch %d" % batch_idx, 1)
            all_offsets = register_mov(mov,refs_and_masks, all_ops, log_cb)
            n.save(reg_data_path, mov)
            n.save(os.path.join(iter_dir, 'all_offsets.npy'), all_offsets)
        else:
            log_cb("Reading registered binaries from %s" % reg_data_path)
            mov = n.load(reg_data_path)
            shmem_mov_params = {}

        nz, nt, ny, nx = mov.shape
        n_frames_proc_new = n_frames_proc + nt
        sum_img_batch = mov.sum(axis=1)
        mean_img_batch = sum_img_batch / nt
        max_img_batch = mov.max(axis=1)

        mov3d = mov.swapaxes(0,1).astype(mov_dtype)
        utils.close_and_unlink_shmem(shmem_mov_params)
    
        run_detection(mov3d, vmap2, sdmov2, n_frames_proc, temporal_high_pass_width, do_running_sdmov,
                      npil_hpf_xy, npil_hpf_z, unif_filter_xy, unif_filter_z, intensity_thresh,
                      log_cb = log_cb)

        log_cb("Update accumulators",2)
        sum_img += sum_img_batch
        mean_img = mean_img_batch * (n_frames_proc / n_frames_proc_new) + mean_img_batch * (nt / n_frames_proc_new)
        max_img = n.max([max_img, max_img_batch], axis=0)
        vmap = vmap2 ** .5
        n_frames_proc = n_frames_proc_new

        log_cb("Iteration complete. Saving outputs to %s" % iter_dir,1)
        n.save(os.path.join(iter_dir, 'vmap.npy'), vmap)
        n.save(os.path.join(iter_dir, 'max_img.npy'), max_img)
        n.save(os.path.join(iter_dir, 'mean_img.npy'), mean_img)
        n.save(os.path.join(iter_dir, 'sum_img.npy'), sum_img)


def run_detection(mov3d, vmap2, sdmov2,n_frames_proc, temporal_high_pass_width, do_running_sdmov,
                    npil_hpf_xy, npil_hpf_z, unif_filter_xy, unif_filter_z, intensity_thresh,
                    log_cb = default_log):
    nt, nz, ny, nx = mov3d.shape
    n_frames_proc_new = n_frames_proc + nt
    log_cb("Temporal high-pass filtering",2)
    mov3d = det_utils.temporal_high_pass_filter(mov3d, temporal_high_pass_width, copy=False)

    log_cb("Computing standard deviation of pixels over time",2)
    if do_running_sdmov:
        sdmov2 += det3d.standard_deviation_over_time(mov3d, batch_size=mov3d.shape[0], sqrt=False)
        sdmov = n.maximum(1e-10, sdmov2 / n_frames_proc_new)
    else:
        sdmov = det3d.standard_deviation_over_time(mov3d, batch_size=mov3d.shape[0], sqrt=True)
    
    log_cb("Neuropil subtraction",2)
    mov3d = det3d.neuropil_subtraction(mov3d / sdmov, npil_hpf_xy, npil_hpf_z)
    log_cb("Square convolution",2)
    mov_u0 = det3d.square_convolution_2d(mov3d, filter_size = unif_filter_xy, filter_size_z = unif_filter_z)
    log_cb("Vmap calculation",2)
    vmap2 += det3d.get_vmap3d(mov_u0, intensity_thresh, sqrt=False)

        
def register_mov(mov3d, refs_and_masks, all_ops, log_cb = default_log):
    nz, nt, ny, nx = mov3d.shape
    all_offsets = {'xms' : [],
                   'yms' : [],
                   'cms' : [],
                   'xm1s': [],
                   'ym1s': [],
                   'cm1s': []}
    for plane_idx in range(nz):
        log_cb("Registering plane %d" % plane_idx, 2)
        __, ym, xm, cm, ym1, xm1, cm1 = register.register_frames(
            refAndMasks = refs_and_masks[plane_idx],
            frames = mov3d[plane_idx],
            ops = all_ops[plane_idx])
        all_offsets['xms'].append(xm); all_offsets['yms'].append(ym); all_offsets['cms'].append(cm)
        all_offsets['xm1s'].append(xm1); all_offsets['ym1s'].append(ym1); all_offsets['cm1s'].append(cm1)
    return all_offsets
        



def init_batch_files(job_iter_dir, job_reg_data_dir, n_batches):
    reg_data_paths = []
    batch_dirs = []
    for batch_idx in range(n_batches):
        reg_data_filename = 'reg_data%04d.npy' % batch_idx
        reg_data_path = os.path.join(job_reg_data_dir, reg_data_filename)
        reg_data_paths.append(reg_data_path)

        batch_dir = os.path.join(job_iter_dir, 'batch%04d' % batch_idx)
        os.makedirs(batch_dir, exist_ok=True)
        batch_dirs.append(batch_dir)

    return batch_dirs, reg_data_paths

        

def subtract_crosstalk(shmem_params, coeff = None, planes = None, n_procs=15, log_cb = default_log):

    assert coeff is not None

    if planes is None:
        pairs = [(i, i+15) for i in range(15)]
    else:
        pairs = []
        for plane_idx in planes:
            if plane_idx > 15:
                if plane_idx - 15 in planes:
                    pairs.append((plane_idx-15, plane_idx))
                    log_cb("Subtracting plane %d from %d" % (pairs[-1][0], pairs[-1][1]), 2)
                else:
                    log_cb("Plane %d does not have its pair %d" % (plane_idx, plane_idx-15),0)
    # print(pairs)
    p = Pool(n_procs)
    p.starmap(subtract_crosstalk_worker, [(shmem_params, coeff, pair[0], pair[1]) \
                                                    for pair in pairs])

    return coeff

def subtract_crosstalk_worker(shmem_params, coeff, deep_plane_idx, shallow_plane_idx):
    shmem, mov3d = utils.load_shmem(shmem_params)
    # print(mov3d.shape, shallow_plane_idx, deep_plane_idx)
    mov3d[shallow_plane_idx] = mov3d[shallow_plane_idx] - coeff * mov3d[deep_plane_idx]
    utils.close_shmem(shmem_params)
    
    
