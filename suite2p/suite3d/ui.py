import os
import shutil
import numpy as n
from matplotlib import pyplot as plt
from argparse import Namespace
import pyqtgraph as pg

try: 
    import napari
except:
    print("No Napari")

try: from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup
except: pass


def load_outputs(dir, files = ['stats.npy', 'F.npy', 'Fneu.npy', 'spks.npy', 'info.npy', 'iscell_filtered.npy', 'iscell_extracted.npy', 'iscell.npy', 'vmap.npy', 'im3d.npy', 'vmap_patch.npy'], return_namespace=False, additional_outputs = {}, regen_iscell=False):
    outputs = {'dir' : dir}
    for filename in files:
        if filename in os.listdir(dir):
            tag = filename.split('.')[0]
            outputs[tag] = n.load(os.path.join(dir, filename),allow_pickle=True)
            if tag == 'info':
                outputs[tag] = outputs[tag].item()
                outputs['vmap'] = outputs['info']['vmap']
                outputs['fs'] = outputs['info']['all_params']['fs']
                if 'vmap_patch' in outputs['info'].keys():
                    outputs['vmap_patch'] = outputs['info']['vmap_patch']
    if 'iscell' not in outputs.keys() or regen_iscell:
        iscell = n.ones((len(outputs['stats']), 2), dtype=int)
        n.save(os.path.join(dir, 'iscell.npy'), iscell)
        outputs['iscell'] = iscell
    if additional_outputs is not None:
        outputs.update(additional_outputs)
    if 'F' in outputs.keys(): 
        outputs['ts'] = n.arange(outputs['F'].shape[-1]) / outputs['fs']
    return outputs

def normalize_planes(im3d, axes = (1,2), normalize_by='mean'):
    imnorm = im3d - im3d.min(axis=axes, keepdims=True)
    if normalize_by == 'mean':
        imnorm /= imnorm.mean(axis=axes, keepdims=True)
    else:
        assert False
    return imnorm

def make_cell_label_vol(stats, iscell, shape, lam_thresh = 0.5, use_patch_coords = False,labels=None, dtype=int, bg_nan=False, max_lam_only = False):
    cell_labels = n.zeros(shape, dtype=dtype)
    if bg_nan:
        cell_labels[:] = n.nan
    n_cells = iscell.sum()
    label_to_idx = {}

    iscell_idxs = n.where(iscell)[0]

    for i, cell_idx in enumerate(iscell_idxs):
        if cell_idx >= len(stats):
            print("Warning - iscell.npy has more cells than stats.npy")
            break
        stat = stats[cell_idx]
        if stat is None: continue

        lam = stat['lam'][stat['lam'] > 0]
        if len(lam) < 1: continue
        lam = stat['lam'] / lam.sum()
        npix = len(lam)
        if max_lam_only: 
            valid_pix = lam == lam.max()
        else: 
            valid_pix = lam > lam_thresh / npix
        if use_patch_coords: coords = stat['coords_patch']
        else: coords = stat['coords']
        zs, ys, xs = [xx[valid_pix] for xx in coords]
        if labels is None:
            cell_labels[zs, ys, xs] = cell_idx + 1
        else:
            # print(cell_idx, xs, labels[i])
            cell_labels[zs, ys, xs] = labels[i]
            
    return cell_labels


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
        
def update_iscell(iscell, dir):
    iscell_path = os.path.join(dir, 'iscell.npy')
    n.save(iscell_path, iscell)


def create_napari_ui(outputs, lam_thresh=0.3, title='3D Viewer', use_patch_coords=False, scale=(15,4,4), theme='dark', extra_cells=None, extra_cells_names=None, vmap_limits=None,
                     extra_images = None, extra_images_names = None, cell_label_name='cells', vmap_name='corr map', use_filtered_iscell=True):
    if use_patch_coords:
        vmap = outputs['vmap_patch']
    else: 
        vmap = outputs['vmap']
    if use_filtered_iscell and 'iscell_filtered' in outputs.keys():
        iscell = outputs['iscell_filtered']
    else:
        iscell = outputs['iscell']
    if len(iscell.shape) > 1:
        iscell = iscell[:,0]
    cell_labels = make_cell_label_vol(outputs['stats'], iscell, vmap.shape,
                                         lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
    not_cell_labels = make_cell_label_vol(outputs['stats'], 1-iscell, vmap.shape,
                                             lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
    v = napari.view_image(
        vmap, title=title, name=vmap_name, opacity=1.0, scale=scale, contrast_limits=vmap_limits)
    if extra_images is not None:
        for i, extra_image in enumerate(extra_images):
            v.add_image(
                extra_image, name=extra_images_names[i], opacity=1.0, scale=scale)

    if 'im3d' in outputs.keys():
        v.add_image(outputs['im3d'], name='Image', scale=scale)
    cell_layer = v.add_labels(cell_labels, name=cell_label_name, opacity=0.5, scale=scale)

    if extra_cells is not None:
        for i, extra_cell in enumerate(extra_cells):
            extra_cell_labels = make_cell_label_vol(extra_cell, n.ones(len(extra_cell)), vmap.shape,
                                                    lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
            v.add_labels(extra_cell_labels,
                         name=extra_cells_names[i], scale=scale, opacity=0.5)

    not_cell_layer = v.add_labels(
        not_cell_labels, name='not-' +cell_label_name, opacity=0.5, scale=scale)
    
    if 'F' in outputs.keys():
        if outputs['F'].shape[0] != len(iscell):
            assert outputs['F'].shape[0] == iscell.sum()
            trace_idxs = n.cumsum(iscell) - 1
        else:
            trace_idxs = n.arange(len(iscell))

    v.theme = theme
    widg_dict = {}
    widg_dict['plot_widget'] = pg.PlotWidget()
    widg_dict['plot_widget'].addLegend()
    widg_dict['f_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='b', name='F')
    widg_dict['fneu_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='r', name='Npil')
    widg_dict['spks_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='w', name='Deconv')
    widg_dict['dock_widget'] = v.window.add_dock_widget(
        widg_dict['plot_widget'], name='activity', area='bottom')



    def get_traces(cell_idx):
        trace_idx = trace_idxs[cell_idx]
        fx = outputs['F'][trace_idx]
        fn = outputs['Fneu'][trace_idx]
        ss = outputs['spks'][trace_idx]
        return outputs['ts'], fx, fn, ss

    def update_plot(widg_dict, cell_idx):
        ts, fx, fn, ss = get_traces(cell_idx)
        widg_dict['f_line'].setData(ts, fx)
        widg_dict['fneu_line'].setData(ts, fn)
        widg_dict['spks_line'].setData(ts, ss)

    @cell_layer.mouse_drag_callbacks.append
    def on_click(cell_labels, event):
        value = cell_labels.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True)
        print(value)
        if value is not None and value > 0:
            cell_idx = value - 1
            if event.button == 1:
                update_plot(widg_dict, cell_idx)
            # if event.button == 2:
            #     mark_cell(
            #         cell_idx, 0, outputs['iscell'], cell_layer, not_cell_layer)

    @not_cell_layer.mouse_drag_callbacks.append
    def on_click(not_cell_labels, event):
        value = not_cell_labels.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True)
        print('Not cell,', value)
        if value is not None and value > 0:
            cell_idx = value - 1
            if event.button == 1:
                update_plot(widg_dict, cell_idx)
            # if event.button == 2:
            #     mark_cell(
            #         cell_idx, 1, outputs['iscell'], cell_layer, not_cell_layer)

    return v

def make_ui_interactive(v, outputs, filters):
    iscell_manual = outputs['iscell'][:,0]
    stats = outputs['stats']
    shape = outputs['vmap'].shape
    iscell_filt_dir = os.path.join(outputs['dir'], 'iscell_filtered.npy')
    print("Saving filtered cells to %s" % iscell_filt_dir)

    ranges = [filt[1] for filt in filters]
    sliders, values = add_filters(v, filters, outputs)
    for slider in sliders:
     slider.slider.sliderReleased.connect(lambda x=0 : update_cells(v, sliders, 
                ranges, iscell_manual, iscell_filt_dir, stats, shape, values))
    return sliders, values


def update_cells(v, sliders, ranges, iscell_manual, iscell_filt_dir, stats, shape, values):
        iscell = iscell_manual.copy()
        print('Total: %d cells' % iscell.sum())
        for i,slider in enumerate(sliders):
            rng = list(slider.slider.value())
            if rng[0] == ranges[i][0]: rng[0] = values[i].min()
            if rng[1] == ranges[i][1]: rng[1] = values[i].max()
            valid = get_valid_cells(values[i], rng)
            iscell = n.logical_and(iscell, valid)
            
        print("%d cells valid" % iscell.sum())
        v.layers['cells'].data = make_cell_label_vol(stats, iscell, shape,
                            lam_thresh=0.3, use_patch_coords=False)
        v.layers['not-cells'].data = make_cell_label_vol(stats, 1-iscell, shape,
                            lam_thresh=0.3, use_patch_coords=False)
        v.layers['cells'].refresh()
        v.layers['not-cells'].refresh()
        print("Updated layer data")
        n.save(iscell_filt_dir, iscell)
        print("Saved iscell")

def add_slider(v, name, srange=(0,1), callback=None):
    slider = QRangeSliderPopup()
    slider.slider.setRange(*srange)
    slider.slider.setSingleStep(0.1)
    slider.slider.setValue(srange)
    slider.slider._slider.sliderReleased.connect(slider.slider.sliderReleased.emit)
    if callback is not None:
        slider.slider.sliderReleased.connect(callback)
    widget = v.window.add_dock_widget(slider, name=name, 
                                             area='left', add_vertical_stretch=False)
    return widget,slider

def add_filters(v, filters, outputs, callback=None):
    sliders = []
    all_values = []
    for filt in filters:
        values = n.array([filt[3](stat[filt[2]]) for stat in outputs['stats']])
        widget, slider = add_slider(v, filt[0], srange=filt[1], callback=callback)
        sliders.append(slider)
        all_values.append(values)
    return sliders, all_values

def get_valid_cells(prop, limits):
    good_cells = n.logical_and(prop > limits[0], prop < limits[1])
    return good_cells


def mark_cell(cell_idx, mark_as, iscell=None, napari_cell_layer=None, napari_not_cell_layer=None, refresh=True):
    napari_idx = cell_idx + 1
    print("Marking cell %d (napari %d) as %d" %
          (cell_idx, napari_idx, int(mark_as)))
    if mark_as:
        cell_layer_val = napari_idx
        not_cell_layer_val = 0
        coords = napari_not_cell_layer.data == napari_idx
    else:
        cell_layer_val = 0
        not_cell_layer_val = napari_idx
        coords = napari_cell_layer.data == napari_idx

    if napari_cell_layer is not None:
        napari_cell_layer.data[coords] = cell_layer_val
        if refresh:
            napari_cell_layer.refresh()
    if napari_not_cell_layer is not None:
        napari_not_cell_layer.data[coords] = not_cell_layer_val
        if refresh:
            napari_not_cell_layer.refresh()
    if iscell is not None:
        iscell[cell_idx] = int(mark_as)
