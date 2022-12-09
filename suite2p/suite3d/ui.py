import os
import shutil
import numpy as n
from matplotlib import pyplot as plt
import napari
from argparse import Namespace


def load_outputs(dir, files = ['stats.npy', 'F.npy', 'Fneu.npy', 'spks.npy', 'vmap.npy', 'im3d.npy'], return_namespace=True):
    outs = {}
    for filename in files:
        key = filename.split('.')[0]
        outs[key] = n.load(os.path.join(dir, filename), allow_pickle=True)

    if return_namespace:
        outs = Namespace(**outs)
    return outs

def normalize_planes(im3d, axes = (1,2), normalize_by='mean'):
    imnorm = im3d - im3d.min(axis=axes, keepdims=True)
    if normalize_by == 'mean':
        imnorm /= imnorm.mean(axis=axes, keepdims=True)
    else:
        assert False
    return imnorm

def make_cell_label_vol(stats, plot_cell_idxs, shape, lam_thresh = 0.5):
    cell_labels = n.zeros(shape, dtype=int)
    n_cells = len(plot_cell_idxs)
    napari_cell_label = n.arange(1,n_cells+1, dtype=int)
    label_to_idx = {}

    for i, cell_idx in enumerate(plot_cell_idxs):
        stat = stats[cell_idx]
        lam = stat['lam'][stat['lam'] > 0]
        if len(lam) < 1: continue
        lam = stat['lam'] / lam.sum()
        npix = len(lam)
        valid_pix = lam > lam_thresh / npix
        zs, ys, xs = [xx[valid_pix] for xx in stat['coords']]

        cell_label = napari_cell_label[i]
        cell_labels[zs, ys, xs] = cell_label
        label_to_idx[cell_label] = cell_idx
    return cell_labels, label_to_idx


def simple_filter_cells(stats, max_w = 30):
    plot_cell_idxs = []
    for cell_idx in range(len(stats)):
        stat = stats[cell_idx]
        lams = stat['lam']
        if n.isnan(lams).sum() > 0:
            continue
        zs, ys, xs = stat['coords']
        if ys.max() - ys.min() > max_w or xs.max() - xs.min() > max_w: 
            # print(cell_idx)
            continue
        plot_cell_idxs.append(cell_idx)
    return n.array(plot_cell_idxs)