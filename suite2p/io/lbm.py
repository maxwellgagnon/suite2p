from inspect import trace
import os
from multiprocessing import shared_memory, Pool
import numpy as n
import time
from skimage import io as skio
from scipy import signal
import tifffile
import imreg_dft as imreg
import json
import psutil
import tracemalloc

def default_log(string, val=None): print(string)
lbm_plane_to_ch = n.array([1,5,6,7,8,9,2,10,11,12,13,14,15,16,17,3,18,19,20,21,22,23,4,24,25,26,27,28,29,30])-1
lbm_ch_to_plane = n.array(n.argsort(lbm_plane_to_ch))

def load_and_stitch_tifs(paths, planes, verbose=True,n_proc=15, mp_args = {}, filt=None, concat=True,
                         convert_plane_ids_to_channel_ids = True, log_cb=default_log, debug=False):
    '''
    Load tifs into memory

    Args:
        paths (list): list of absolutep aths to tiff files
        planes (list): planes to load, 0 being deepest and 30 being shallowest
        verbose (bool, optional): Verbosity. Defaults to True.
        n_proc (int, optional): Number of processors. Defaults to 15.
        mp_args (dict, optional): args to pass to worker. Defaults to {}.
        filt (tuple, optional): parameters of spatiotemporal filter. Defaults to None.
        concat (bool, optional): Concatenate across time. Defaults to True.
        convert_plane_ids_to_channel_ids (bool, optional): Convert the plane_ids to account for scanimage ordering. Set to False to access planes by their scanimage channel ID. Defaults to True.
        log_cb (func, optional): Callback function for logging. Defaults to default_log.
        debug (bool, optional): Debugging mode. Defaults to False.

    Returns:
        _type_: _description_
    '''
    


    if convert_plane_ids_to_channel_ids:
        channels = lbm_plane_to_ch[n.array(planes)]
    if filt is not None:
        filt = get_filter(filt)

    mov_list = []
    for tif_path in paths:
        if verbose: log_cb("Loading %s" % tif_path, 2)
        im, px, py = load_and_stitch_full_tif_mp(tif_path, channels=channels, verbose=False, filt=filt, n_proc=n_proc,debug=debug, **mp_args)
        mov_list.append(im)
    if concat:
        mov = n.concatenate(mov_list,axis=1)
        size = mov.nbytes/(1024*1024*1024)
    else: 
        mov = mov_list
        size = n.sum([mx.nbytes/(1024**3) for mx in mov])
    if verbose: log_cb("Loaded %d files, total %.2f GB" % (len(paths),size),1)
    return mov


def load_and_stitch_full_tif_mp(path, channels, n_proc=10, verbose=True,translations=None, filt = None, debug=False, get_roi_start_pix=False):
    tic = time.time()
    n_ch_overwrite=30
    # TODO imread from tifffile has an overhead of ~20-30 seconds before it actually reads the file?
    tiffile = tifffile.imread(path)
    if len(tiffile.shape) < 4:
        n_t_ch, n1, n2 = tiffile.shape
        tiffile = tiffile.reshape(int(n_t_ch/30.0), 30, n1,n2)
  
    rois = get_meso_rois(path)
    # print("XXXXXX %.2f" % (tiffile.nbytes / 1024**3))
    sh_mem = shared_memory.SharedMemory(create=True, size=tiffile.nbytes)
    sh_tif = n.ndarray(tiffile.shape, dtype=tiffile.dtype, buffer=sh_mem.buf)
    sh_tif[:] = tiffile[:]
    if debug: print("4, %.4f" % (time.time()-tic))
    sh_mem_name = sh_mem.name
    sh_mem_params = (sh_tif.shape, sh_tif.dtype)

    n_t, n_ch_tif,__,__ = sh_tif.shape
    n_ch = len(channels)

    # split and stitch two frames to figure out the output size
    ims_sample= split_rois_from_tif(tiffile[:2], rois, ch_id=0)
    if debug: print("5, %.4f" % (time.time()-tic))
    # sample_out = stitch_rois(ims_sample, rois, return_coords=False,mean_img=False)
    if get_roi_start_pix:
        return stitch_rois_fast(ims_sample, rois, mean_img=False, get_roi_start_pix=True)
    sample_out, px, py = stitch_rois_fast(ims_sample, rois,mean_img=False)
    if debug: print("6, %.4f" % (time.time()-tic))
    __, n_y, n_x = sample_out.shape
    del tiffile

    shape_out = (n_ch,n_t,n_y,n_x)
    size_out = (n_t*n_ch*n_y*n_x) * sh_tif[0,0,0,0].nbytes
    sh_mem_out = shared_memory.SharedMemory(create=True, size=size_out)
    sh_out = n.ndarray(shape_out, dtype=sh_tif.dtype, buffer=sh_mem_out.buf)    
    sh_out_name = sh_mem_out.name
    sh_out_params = (sh_out.shape, sh_out.dtype)
    # print(sh_out_params)
    if debug: print("7, %.4f" % (time.time()-tic))

    if translations is None:
        translations = n.zeros((n_ch,2))

    prep_tic = time.time()
    if verbose: print("    Loaded file into shared memory in %.2f sec" % (prep_tic - tic))

    p = Pool(processes = n_proc)
    output = p.starmap(load_and_stitch_full_tif_worker, 
                      [(idx,ch_id, rois, sh_mem_name, sh_mem_params, sh_out_name, sh_out_params, translations[idx], filt)\
                        for idx,ch_id in enumerate(channels)])
    proc_tic = time.time()
    if verbose: print("    Workers completed in %.2f sec" % (proc_tic - prep_tic))
    
    if debug: print("8, %.4f" % (time.time()-tic))

    # print(output)
    if verbose: print("    Total time: %.2f sec" % (time.time()-tic))
    sh_mem.close()
    sh_mem.unlink()
    p.close()
    p.terminate()

    im_full = n.zeros(sh_out.shape, sh_out.dtype)
    im_full[:] = sh_out[:]
    sh_mem_out.close()
    sh_mem_out.unlink()

    return im_full, px, py

def load_and_stitch_full_tif_worker(idx, ch_id, rois, sh_mem_name, sh_arr_params, sh_out_name, sh_out_params, translation=None, filt=None):
    debug=False
    if debug: print("Loading channel %d" % ch_id)
    tic = time.time()

    sh_mem = shared_memory.SharedMemory(sh_mem_name)
    tiffile = n.ndarray(shape=sh_arr_params[0], dtype=sh_arr_params[1], buffer=sh_mem.buf)

    if filt is not None:
        b,a = filt
        line_mov = tiffile[:,ch_id].mean(axis=-1)
        line_mov_filt = signal.filtfilt(b,a, line_mov)
        line_mov_diff = line_mov - line_mov_filt

        tiffile[:,ch_id] = tiffile[:,ch_id]  - line_mov_diff[:,:,n.newaxis]
        if debug: print('%d filtered in %.2f' % (ch_id, time.time() - tic))
        tic = time.time()

    sh_mem_out = shared_memory.SharedMemory(sh_out_name)
    outputs = n.ndarray(shape=sh_out_params[0], dtype=sh_out_params[1], buffer=sh_mem_out.buf)
    
    prep_time = time.time()
    if debug: print(" %d Loaded in %.2f" % (ch_id, prep_time-tic))
    ims = split_rois_from_tif(tiffile, rois, ch_id = ch_id, return_coords=False)
    split_time = time.time()
    if debug: print(" %d Split in %.2f" % (ch_id, split_time-prep_time))
    outputs[idx], __, __ = stitch_rois_fast(ims, rois, mean_img=False, translation=translation)
    stitch_time = time.time()
    if debug: print(" %d Stitch in %.2f" % (ch_id, stitch_time-split_time))
    if debug: print("Channel %d done in %.2f" % (ch_id, time.time()-tic))

    sh_mem.close() # REMOVE ME IF THERE IS AN ERROR
    sh_mem_out.close() # Remove me if there is an error


    return time.time()-tic


def get_meso_rois(tif_path, max_roi_width_pix=145):
    tf = tifffile.TiffFile(tif_path)
    artists_json = tf.pages[0].tags["Artist"].value

    si_rois = json.loads(artists_json)['RoiGroups']['imagingRoiGroup']['rois']

    rois = []
    warned = False
    for roi in si_rois:
        if type(roi['scanfields']) != list:
            scanfield = roi['scanfields']
        else: 
            scanfield = roi['scanfields'][n.where(n.array(roi['zs'])==0)[0][0]]

    #     print(scanfield)
        roi_dict = {}
        roi_dict['uid'] = scanfield['roiUuid']
        roi_dict['center'] = n.array(scanfield['centerXY'])
        roi_dict['sizeXY'] = n.array(scanfield['sizeXY'])
        roi_dict['pixXY'] = n.array(scanfield['pixelResolutionXY'])
        if roi_dict['pixXY'][0] > max_roi_width_pix and not warned:
            # print("SI ROI pix count in x is %d, which is impossible, setting it to %d" % (roi_dict['pixXY'][0],max_roi_width_pix))
            warned=True
            roi_dict['pixXY'][0] = max_roi_width_pix
    #         print(scanfield)
        rois.append(roi_dict)
    #     print(len(roi['scanfields']))

    roi_pixs = n.array([r['pixXY'] for r in rois])
    return rois


def split_rois_from_tif(im, rois, ch_id = 0, return_coords=False):
    nt, np, ny, nx = im.shape
    n_rois = len(rois)
    ys = n.array([roi['pixXY'][1] for roi in rois])
    n_buff = (ny - ys.sum())/(len(rois)-1)
    if int(n_buff) != n_buff: 
        print("WARNING: Buffer between ROIs is calculated as a non-integer from tiff (%.2f pix)" % n_buff )
    n_buff = int(n_buff)
    
    split_ims = []
    coords = []
    y_start = 0
    for i in range(n_rois):
        ny = ys[i]
        split_im = im[:, ch_id, y_start:y_start+ny]
        coord = n.meshgrid(n.arange(y_start, y_start+ny), n.arange(0,im.shape[-1]))
        split_ims.append(split_im)
        y_start += ny + n_buff
        coords.append(coord)
    if return_coords: return split_ims, coords
    return split_ims


def stitch_rois_fast(ims, rois, mean_img=False, translation = None, get_roi_start_pix=False):
    tic = time.time()

    sizes_pix = n.array([im.shape[1:][::-1] for im in ims])

    centers = n.array([r['center'] for r in rois])
    sizes = n.array([r['sizeXY'] for r in rois])
    corners = centers - sizes/2
    n_rois = len(rois)

    # X is the fast axis along the resonant scanner line direction, Y is orthogonal slow axis
    # For a typical strip, x extent is small and y extent is large

    # maximim and minimum x/y coordinates in SI units (not pixels, also strangely not um)
    xmin, xmax = corners[:,0].min(),(corners[:,0] + sizes[:,0]).max()
    ymin, ymax = corners[:,1].min(),(corners[:,1] + sizes[:,1]).max()

    # calculate pixel sizes (relative to weird SI units)
    pixel_sizes = sizes/sizes_pix
    psize_y = n.mean(pixel_sizes[:,1])
    psize_x = n.mean(pixel_sizes[:,0])
    assert n.product(n.isclose(pixel_sizes[:,1]-psize_y, 0)), "Y pixels not uniform"
    assert n.product(n.isclose(pixel_sizes[:,0]-psize_x, 0)), "X pixels not uniform"

    # SI unit coordinates of each pixel of the full image
    full_xs = n.arange(xmin,xmax, psize_x)
    full_ys = n.arange(ymin,ymax, psize_y)
    # accumulate how many times a value is written into each pixel
#     if accumulate: full_image_acc = n.zeros(full_image.shape[1:])

    # calculate the starting pixel for each of the ROIs when placing them into full image
    roi_start_pix_x = []
    roi_start_pix_y = []
    for roi_idx in range(n_rois):
        x_corner = corners[roi_idx,0]
        y_corner = corners[roi_idx, 1]

        closest_x_idx = n.argmin(n.abs(full_xs-x_corner))
        roi_start_pix_x.append(closest_x_idx)
        closest_x = full_xs[closest_x_idx]
        if not n.isclose(closest_x, x_corner):
            print("ROI %d x does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                  (roi_idx, closest_x, x_corner))

        closest_y_idx = n.argmin(n.abs(full_ys-y_corner))
        roi_start_pix_y.append(closest_y_idx)
        closest_y = full_ys[closest_y_idx]
        if not n.isclose(closest_y, y_corner):
            print("ROI %d y does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                  (roi_idx, closest_y, y_corner))
            
    prep_tic = time.time()

    if get_roi_start_pix:
        return roi_start_pix_y, roi_start_pix_x

    if mean_img:
        n_t = 1
    else:
        n_t = ims[0].shape[0]

    full_image = n.zeros((n_t, len(full_ys), len(full_xs)))

    stitch_rois_fast_helper(full_image, ims, n.array(roi_start_pix_x), n.array(roi_start_pix_y), sizes_pix, mean_img)

    if translation is not None and n.linalg.norm(translation) > 0.01:
        for i in range(full_image.shape[0]):
            full_image[i] = imreg.transform_img(full_image[i],tvec = translation)

    # print(time.time()-prep_tic)
    return full_image, psize_x, psize_y
    
    
# @numba.jit(nopython=True)
def stitch_rois_fast_helper(full_image, ims, roi_start_pix_x, roi_start_pix_y, sizes_pix, mean_img):
    n_rois = len(ims)
    # place each ROI into full image
    for roi_idx in range(n_rois):
        roi_x_start,roi_y_start = roi_start_pix_x[roi_idx], roi_start_pix_y[roi_idx]
        roi_x_end = roi_x_start + sizes_pix[roi_idx][0]
        roi_y_end = roi_y_start + sizes_pix[roi_idx][1]

        if mean_img:
            assert False, 'not implemented'
            # return full_image[0, roi_y_start:roi_y_end, roi_x_start:roi_x_end] = ims[roi_idx].mean(axis=0)
        else:
            full_image[:,roi_y_start:roi_y_end, roi_x_start:roi_x_end] = ims[roi_idx]
    return

def get_filter(filt_params):
    return signal.iirnotch(filt_params['f0'],filt_params['Q'], filt_params['line_freq'] )