import numpy as n
np = n
from scipy.ndimage import maximum_filter, gaussian_filter, uniform_filter

def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Lz, Ly, Lx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Lz, Ly, Lx).astype(np.float32).mean(axis=1)

def standard_deviation_over_time(mov: np.ndarray, batch_size: int,sqrt=True) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Lz x Ly x Lx
        The statistics for each pixel
    """
    nbins, Lz, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Lz, Ly, Lx), 'float32')
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix+batch_size, :, :], axis=0) ** 2).sum(axis=0))
    if sqrt: 
        sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov

def neuropil_subtraction(mov: np.ndarray, filter_size: int, filter_size_z: int, mode='constant') -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""
    nbinned, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filt_size, mode=mode) / c1)
    return movt

def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """
    Returns a pixel-downsampled movie from 'mov', tapering the edges of 'taper_edge' is True.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to downsample
    taper_edge: bool
        Whether to taper the edges

    Returns
    -------
    filtered_mov:
        The downsampled frames
    """
    n_frames, Lz, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, Lz, int(np.ceil(Ly / 2)), Lx), 'float32')
    movd[:,:, :Ly//2, :] = np.mean([mov[:,:, 0:-1:2, :], mov[:,:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:,:, -1, :] = mov[:,:, -1, :] / 2 if taper_edge else mov[:,:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, Lz,  int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), 'float32')
    mov2[:,:, :, :Lx//2] = np.mean([movd[:,:, :, 0:-1:2], movd[:,:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:,:, :, -1] = movd[:,:, :, -1] / 2 if taper_edge else movd[:,:, :, -1]

    return mov2

def square_convolution_2d(mov: np.ndarray, filter_size: int, filter_size_z: int) -> np.ndarray:
    """Returns movie convolved by uniform kernel with width 'filter_size'."""
    movt = np.zeros_like(mov, dtype=np.float32)
    filt_size = (filter_size_z, filter_size, filter_size)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filter_size, mode='constant')
    return movt
