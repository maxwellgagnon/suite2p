import os
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter, gaussian_filter, uniform_filter, percentile_filter
import numpy as n

def default_log(string, *args, **kwargs): 
    print(string)

def detect_cells(patch, vmap, max_iter = 10000, peak_thresh = 2.5, activity_thresh = 2.5, extend_thresh=0.2, 
                    roi_ext_iterations=2, max_ext_iters=20, percentile=0, log=default_log, 
                    recompute_v = False, offset=(0,0,0), savepath=None, debug=False,**kwargs):
    nt, nz, ny, nx = patch.shape
    stats = []

    Th2 = activity_thresh
    vmultiplier = 1 #max(1, nt / magic_number)
    peak_thresh = vmultiplier * peak_thresh
    vmin = vmap.min()
    log("Starting extraction with peak_thresh: %0.3f and Th2: %0.3f" % (peak_thresh, Th2), 1)

    for iter_idx in range(max_iter):
        med, zz, yy, xx, lam, peak_val = find_top_roi3d(vmap, xy_pix_scale = 3)
        if peak_val < peak_thresh:
            log("Iter %04d: peak is too small (%0.3f) - ending extraction" % (iter_idx, peak_val), 2)
            break
        
        tproj = patch[:,zz,yy,xx] @ lam
        threshold = min(Th2, n.percentile(tproj, percentile)) if percentile > 0 else Th2
        log("Cell %d at with peak %.3f, activity_thresh %.3f, max %0.3f" % (iter_idx, peak_val, threshold, tproj.max()), 2)
        active_frames = n.nonzero(tproj > threshold)[0]

        for i in range(roi_ext_iterations):
            log("%d/%d active frames" % (len(active_frames),nt), 3)
            if len(active_frames) == 0:
                log(1,"WARNING: no active frames in roi %d" % iter_idx)
            zz,yy,xx,lam = iter_extend3d(zz,yy,xx,active_frames, patch, extend_thresh=extend_thresh,
                                            max_ext_iters=max_ext_iters, verbose=debug)
            tproj = patch[:,zz,yy,xx] @ lam
            active_frames = n.nonzero(tproj > threshold)[0]
            npix = len(lam)
        
        sub = n.zeros((nt,npix))
        sub[active_frames] = tproj[active_frames, n.newaxis] @ lam[n.newaxis]
        patch[:,zz,yy,xx] -= sub

        if recompute_v:
            mnew = patch[:,zz,yy,xx]
            vmap[zz,yy,xx] = (mnew * n.float32(mnew > threshold)).sum(axis=0) ** 0.5
        else:
            vmap[zz,yy,xx] = vmin
        
        stat = {
            'idx' : iter_idx,
            'coords_patch' : (zz,yy,xx),
            'coords' : (zz+offset[0],yy+offset[1],xx+offset[2]),
            'lam' : lam,
            'med' : med,
            'active_frames' : active_frames
        }
        stats.append(stat)
        log("Added cell %d at %02d, %03d, %03d with peak: %0.3f and %d pixels" % (len(stats), med[0],med[1],med[2], peak_val, npix), 2)
        if savepath is not None and iter_idx % 250 == 0 and iter_idx > 0:
            n.save(savepath,stats)
            log("Saving checkpoint to %s" % savepath)
    log("Found %d cells in %d iterations" % (len(stats), iter_idx+1))
    if savepath is not None:
        log("Saving cells to %s" % savepath)
        n.save(savepath, stats)
    return stats
    



def binned_mean(mov: n.ndarray, bin_size):
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    # from suite2p/binary
    n_frames, nz, ny, nx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, nz, ny, nx).mean(axis=1)

def find_top_roi3d(V1, xy_pix_scale = 3, z_pix_scale = 1, peak_thresh=None):
    zi, yi, xi = n.unravel_index(n.argmax(V1), V1.shape)
    peak_val = V1.max()

    if peak_thresh is not None and peak_val < peak_thresh:
        print("Peak too small")
    #     return

    zz, yy, xx, lam = add_square3d(zi, yi, xi, V1.shape, xy_pix_scale, z_pix_scale)

    med = (zi, yi, xi)
    return med, zz, yy, xx, lam, peak_val

def add_square3d(zi, yi, xi, shape, xy_pix_scale=3, z_pix_scale=1):
    nz, ny, nx = shape

    xs = n.arange(xi - int(xy_pix_scale/2), xi + int(n.ceil(xy_pix_scale/2)))
    ys = n.arange(yi - int(xy_pix_scale/2), yi + int(n.ceil(xy_pix_scale/2)))
    zs = n.arange(zi - int(z_pix_scale/2), zi + int(n.ceil(z_pix_scale/2)))
    zz, yy, xx = [vv.flatten() for vv in n.meshgrid(zs, ys, xs)]

    #check if each coord is within the possible coordinates
    valid_pix = n.all([n.all([vv > -1, vv < nv], axis=0) for vv, nv in zip((zz,yy,xx), (nz,ny,nx))],axis=0)

    zz = zz[valid_pix]
    yy = yy[valid_pix]
    xx = xx[valid_pix]

    mask = n.ones_like(zz)
    mask = mask / n.linalg.norm(mask)
    
    return zz, yy, xx, mask


def iter_extend3d(zz,yy,xx, active_frames, mov, verbose=False, extend_thresh=0.2, max_ext_iters=10,extend_z=True):
    # pr = cProfile.Profile()
    # pr.enable()
    npix = 0
    iter_idx = 0
    mov_act = mov[active_frames].mean(axis=0)
    # lam = n.array([lam0])
    while npix < 10000 and iter_idx < max_ext_iters:
        npix = len(yy)
        zz, yy, xx = extend_roi3d(zz,yy,xx, mov.shape[1:], extend_z=extend_z)
        lam = mov_act[zz, yy, xx]
        incl_pix = lam > max(lam.max() * extend_thresh, 0)
        if incl_pix.sum() == 0:
            if verbose: print("Break - no pixels")
            break
        zz, yy, xx, lam = [vv[incl_pix] for vv in [zz,yy,xx,lam]]
        if verbose: print("Iter %d, %d/%d pix included" % (iter_idx, incl_pix.sum(), len(incl_pix)))
        if not incl_pix.sum() > npix: 
            if verbose: print("Break - no more growing")
            break
        iter_idx += 1
    lam = lam / n.sum(lam**2)**.5
    return zz,yy,xx,lam


def extend_roi3d_iter(zz, yy, xx, shape, n_iters=3):
    for i in range(n_iters):
        zz, yy, xx = extend_roi3d(zz, yy, xx, shape)
    return zz, yy, xx


def extend_roi3d(zz, yy, xx, shape, extend_z=True):
    n_pix = len(zz)
    coords = [[zz[i], yy[i], xx[i]] for i in range(n_pix)]
    for coord_idx in range(n_pix):
        coord = coords[coord_idx]
        for i in range(3):
            if not extend_z and i == 0:
                continue
            for sign in [-1, 1]:
                v = list(coord)
                v[i] = v[i] + sign
                out_of_bounds = False
                for j in range(len(v)):
                    if v[j] < 0 or v[j] >= shape[j]:
                        out_of_bounds = True
                if not out_of_bounds:
                    coords.append(v)

    zz, yy, xx = n.unique(coords, axis=0).T

    return zz, yy, xx


def extend_helper(vv_roi, vv_ring, extend_v, nv, v_max_extension=None):
    if v_max_extension is None:
        v_max_extension = n.inf
    v_min, v_max = vv_ring.min(), vv_ring.max()
    v_absmin = max(0,  vv_roi.min() - v_max_extension,
                   vv_ring.min() - extend_v)
    v_absmax = min(nv, vv_roi.max() + v_max_extension +
                   1, vv_ring.max() + 1 + extend_v)

    # print(v_absmin)
    return n.arange(v_absmin, v_absmax)


def create_cell_pix(stats, shape, lam_percentile=70.0, percentile_filter_shape=(3, 25, 25)):
    nz, ny, nx = shape
    lam_map = n.zeros((nz, ny, nx))
    roi_map = n.zeros((nz, ny, nx))

    for i, stat in enumerate(stats):
        zc, yc, xc = stat['coords']
        lam = stat['lam']
        lam_map[zc, yc, xc] = n.maximum(lam_map[zc, yc, xc], lam)

    if lam_percentile > 0.0:
        filt = percentile_filter(lam_map, percentile=70.0, size=(3, 25, 25))
        cell_pix = ~n.logical_or(lam_map < filt, lam_map == 0)
    else:
        cell_pix = lam_map > 0.0
    return cell_pix

def get_neuropil_mask(stat, cell_pix, min_neuropil_pixels=1000, extend_by=(1, 3, 3), z_max_extension=5,
                      max_np_ext_iters=5, return_coords_only=False, np_ring_iterations=2):
    zz_roi, yy_roi, xx_roi = stat['coords']
    zz_ring, yy_ring, xx_ring = extend_roi3d_iter(
        zz_roi, yy_roi, xx_roi, cell_pix.shape, np_ring_iterations)

    nz, ny, nx = cell_pix.shape

    n_ring = (~cell_pix[zz_ring, yy_ring, xx_ring]).sum()

    zz_np, yy_np, xx_np = zz_ring.copy(), yy_ring.copy(), xx_ring.copy()

    n_np_pix = 0
    iter_idx = 0
    while n_np_pix < min_neuropil_pixels and iter_idx < max_np_ext_iters:
        zs_np = extend_helper(zz_roi, zz_np, extend_by[0], nz, z_max_extension)
        ys_np = extend_helper(yy_roi, yy_np, extend_by[1], ny)
        xs_np = extend_helper(xx_roi, xx_np, extend_by[2], nx)

        zz_np, yy_np, xx_np = n.meshgrid(zs_np, ys_np, xs_np, indexing='ij')
        np_pixs = (~cell_pix[zz_np, yy_np, xx_np])
        n_np_pix = (np_pixs).sum() - n_ring
        # print(n_np_pix)
        # print(zs_np)
        # print(xs_np)

        iter_idx += 1

    if return_coords_only:
        return zz_np, yy_np, xx_np, zz_ring, yy_ring, xx_ring

    else:
        neuropil_mask = n.zeros((nz, ny, nx))
        neuropil_mask[zz_np[np_pixs], yy_np[np_pixs], xx_np[np_pixs]] = True
        neuropil_mask[zz_ring, yy_ring, xx_ring] = False
        pix = n.nonzero(neuropil_mask)

        return pix


def compute_npil_masks(stats, shape, offset = (0,0,0), np_params={}):
    # TODO: parallelize this (EASY)
    cell_pix = create_cell_pix(stats, shape)
    for stat in stats:
        zc, yc, xc = stat['coords']
        npz, npy, npx = get_neuropil_mask(stat, cell_pix, **np_params)
        stat['npcoords'] = (npz, npy, npx)
        stat['npcoords_patch'] = (npz-offset[0], npy-offset[1], npx-offset[2])
    return stats

def extract_activity(mov, stats, batchsize_frames=5000):
    # if you run out of memory, reduce batchsize_frames
    nz,nt,ny,nx = mov.shape
    ns = len(stats)
    F_roi = n.zeros((ns, nt))
    F_neu = n.zeros((ns, nt))

    n_batches = int(n.ceil(nt / batchsize_frames))
    for batch_idx in range(n_batches):
        start = batch_idx * batchsize_frames
        end = min(nt, start + batchsize_frames)
        mov_batch = mov[:,start:end].compute()
        for i in range(ns):
            stat = stats[i]
            zc, yc, xc = stat['coords']
            npzc, npyc, npxc = stat['npcoords']
            lam = stat['lam'] / stat['lam'].sum()
            F_roi[i,start:end] = lam @ mov_batch[zc,:,yc,xc]
            F_neu[i,start:end] = mov_batch[npzc,:,npyc,npxc].mean(axis=0)

    return F_roi, F_neu