import numpy as n
import tensorflow as tf
from tensorflow.keras.models import load_model

def default_log(string, val=1):
    print(string)

def init_deepinterp_model(model_path):
    model = load_model(model_path)
    return model

def denoise_movie(model,frames, batch_size=4, n_before=30, n_after=30,callbacks=None):
    """Run series of frames through the model. First and last 30 frames are skipped. No normalizing done in this function

    Args:
        frames (ndarray): ny, nx, nt
        model (keras.model): result of init_deepinterp_model
    """
    ny, nx, nt = frames.shape
    n_buf = n_before + n_after
    points = n.concatenate([n.arange(n_before), n.arange(n_after)+n_before+1])
    plane_tf = tf.convert_to_tensor(frames)

    @tf.function       
    def get_surrounding_window(sample_idx):
        # ret = tf.stack([plane_tf[:,:,sample_idx + i ] for i in points], axis=-1)
        ret = tf.concat([plane_tf[:,:,sample_idx:sample_idx + n_before],plane_tf[:,:,sample_idx+n_before+1:sample_idx + n_before + n_after + 1]], axis=-1)
        return ret
    frame_dset = tf.data.Dataset.range(nt - n_buf)
    frame_dset = frame_dset.map(get_surrounding_window)
    frame_dset = frame_dset.batch(batch_size)
    yp = model.predict(frame_dset, batch_size=batch_size, callbacks=callbacks)[:,:,:,0]
    return yp
    
def denoise_quadrants(model, plane_in, plane_out, do_norm=True, log= default_log):
    bounds = [((0,512), (0,512)),   ((512,1024), (0,512)),
              ((0,512), (512,1024)),((512,1024), (512,1024))]
    for crop_idx in range(4):
        log("  quadrant %02d" % crop_idx, 2)
        plane = plane_in[:,  bounds[crop_idx][0][0]:bounds[crop_idx][0][1], \
                          bounds[crop_idx][1][0]:bounds[crop_idx][1][1]]
        if do_norm:
            mean = plane.mean()
            std = plane.std()
            plane = (plane - mean) / std
        plane = n.moveaxis(plane, 0, 2)
        yp = denoise_movie(model, plane)
        if do_norm: 
            yp = (yp * std) + mean
        plane_out[:, bounds[crop_idx][0][0]:bounds[crop_idx][0][1], \
                     bounds[crop_idx][1][0]:bounds[crop_idx][1][1]] = yp
def denoise_mov3d(model, mov3d, buffer_full, plane_means, plane_stds, do_norm_per_batch=False, log=default_log):
    mov_out = n.zeros(mov3d.shape)
    nz, nt, ny, nx = mov3d.shape
    n_buffer = buffer_full.shape[1]
    log("Denoising %02d planes with buffer of size %02d" % (nz, n_buffer))
    mov = n.concatenate([buffer_full, mov3d], axis=1)

    assert ny == 1024 and nx == 1024
    if plane_means is None and not do_norm_per_batch:
        plane_means = mov3d.mean(axis=(1,2,3))
    if plane_stds is None and not do_norm_per_batch:
        plane_stds = mov3d.std(axis=(1,2,3))
        
    for plane_idx in range(nz):
        log("Denoising plane %02d of %02d" % (plane_idx, nz), 2)
        plane = mov[plane_idx]
        if not do_norm_per_batch: plane = (plane - plane_means[plane_idx]) / plane_stds[plane_idx]
        denoise_quadrants(model, plane, mov_out[plane_idx], log = log, do_norm=do_norm_per_batch)
        if not do_norm_per_batch: mov_out[plane_idx] = (mov_out[plane_idx] * plane_stds[plane_idx]) + plane_means[plane_idx]
        
    buffer_full[:,:] = mov3d[:, -n_buffer:]
    
    return mov_out