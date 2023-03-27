
import tifffile
from cmath import log
import datetime
import os
import numpy as n
from . import init_pass
from . import utils as u3d 
import psutil
from suite2p.io import lbm as lbmio
from multiprocessing import Pool
from suite2p.suite3d.iter_step import register_dataset, fuse_and_save_reg_file, calculate_corrmap
from suite2p.suite3d import extension as ext
from suite2p.extraction import dcnv
from suite2p.detection import svd_utils as svu
from . import ui

class Job:
    def __init__(self, root_dir, job_id, params=None, tifs=None, exist_ok=False, verbosity=10, create=True):
        """Create a Job object that is a wrapper to manage files, current state, log etc.

        Args:
            root_dir (str): Root directory in which job directory will be created
            job_id (str): Unique name for the job directory
            params (dict): Job parameters (see examples)
            tifs (list) : list of full paths to tif files to be used
            exist_ok (bool, optional): If False, will throw error if job_dir exists. Defaults to False.
            verbosity (int, optional): Verbosity level. 0: critical only, 1: info, 2: debug. Defaults to 1.
        """

        self.verbosity = verbosity
        self.job_id = job_id
        if create:
            self.params = params
            self.params['tifs'] = tifs
            self.tifs = tifs
            self.init_job_dir(root_dir, job_id, exist_ok=exist_ok)
        else:
            self.init_job_dir(root_dir, job_id, exist_ok=True, update_params=False)
            self.load_params()
            self.tifs = self.params.get('tifs', [])
            
        self.save_params()


    def log(self, string='', level=1, logfile=True, log_mem_usage=False):
        """Print messages based on current verbosity level

        Args:
            string (str): String to be printed
            level (int, optional): Level equal or below self.verbosity will be printed. Defaults to 1.
        """
        if log_mem_usage:
            
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            total = sm.used + vm.used
            string = "{:<20}".format(string)
            string += ("Total Used: %07.3f GB, Virtual: %07.3f GB, Swap: %07.3f" %
                       ((total/(1024**3), vm.used/(1024**3), sm.used/(1024**3))))
            
        if level <= self.verbosity:
            # print('xxx')
            print(("   " * level) + string)
        if logfile:
            logfile = os.path.join(self.job_dir, 'log.txt')
            self.logfile = logfile
            with open(logfile, 'a+') as f:
                datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = '\n[%s][%02d] ' % (datetime_string, level)
                f.write(header + '   ' * level + string)

    def save_params(self, new_params=None, dir = 'job_dir'):
        """Update saved params in job_dir/params.npy
        """
        params_path = os.path.join(self.dirs[dir], 'params.npy')
        if new_params is not None:
            self.params.update(new_params)
        n.save(params_path, self.params)
        self.log("Updated params file: %s" % params_path)

    def load_params(self, dir = None):
        if dir is None:
            dir = 'job_dir'
        params_path = os.path.join(self.dirs[dir], 'params.npy')
        self.params = n.load(params_path, allow_pickle=True).item()
        self.log("Found and loaded params from %s" % params_path)

    def load_summary(self):
        summary_path = os.path.join(self.dirs['summary'], 'summary.npy')
        summary = n.load(summary_path,  allow_pickle=True).item()
        return summary

    def make_svd_dirs(self, n_blocks=None):
        self.make_new_dir('svd')
        self.make_new_dir('blocks', 'svd', dir_tag='svd_blocks')
        block_dirs = []
        if n_blocks is not None:
            for i in range(n_blocks):
                block_dirs.append(self.make_new_dir('%03d' % i, 'svd_blocks', dir_tag = 'svd_blocks_%03d' % i))
            return block_dirs

    def make_stack_dirs(self, n_stacks):
        stack_dirs = []
        self.make_new_dir('stacks', 'svd', dir_tag='svd_stacks')
        for i in range(n_stacks):
            stack_dirs.append(self.make_new_dir('%03d' % i, 'svd_stacks', dir_tag = 'svd_stacks_%03d' % i))
        return stack_dirs

    def make_extension_dir(self, extension_root, extension_name='ext'):
        extension_dir = os.path.join(extension_root, 's3d-extension-%s' % self.job_id)
        if extension_name in self.dirs.keys():
            self.log("Extension dir %s already exists at %s" % (extension_name, self.dirs[extension_name]))
            return self.dirs[extension_name]
        os.makedirs(extension_dir)
        self.log("Made new extension dir at %s" % extension_dir)
        self.dirs[extension_name] = extension_dir
        self.save_dirs()
        return extension_dir

    def save_dirs(self):
        n.save(os.path.join(self.job_dir, 'dirs.npy'), self.dirs)

    def make_new_dir(self, dir_name, parent_dir_name = None, exist_ok=True, dir_tag = None):
        if parent_dir_name is None:
            parent_dir = self.job_dir
        else: 
            parent_dir = self.dirs[parent_dir_name]
        if dir_tag is None:
            dir_tag = dir_name
        
        dir_path = os.path.join(parent_dir, dir_name)
        os.makedirs(dir_path, exist_ok = exist_ok)
        self.dirs[dir_tag] = dir_path
        n.save(os.path.join(self.job_dir, 'dirs.npy'), self.dirs)
        return dir_path

    def init_job_dir(self, root_dir, job_id, exist_ok=False, update_params=True):
        """Create a job directory and nested dirs

        Args:
            root_dir (str): Root directory to create job_dir in
            job_id (str): Unique name for job
            exist_ok (bool, optional): If False, throws error if job_dir exists. Defaults to False.
        """

        job_dir = os.path.join(root_dir,'s3d-%s' % job_id)
        self.job_dir = job_dir
        if os.path.isdir(job_dir):
            self.log("Job directory %s already exists" % job_dir, 0)
            assert exist_ok, "Manually delete job_dir, or set exist_ok=True"
        else:
            os.makedirs(job_dir, exist_ok=True)

        self.log("Loading job directory for %s in %s" %
                    (job_id, root_dir), 0)
        if 'dirs.npy' in os.listdir(job_dir):
            self.log("Loading dirs ")
            self.dirs = n.load(os.path.join(job_dir, 'dirs.npy'),allow_pickle=True).item()
        else:
            self.dirs = {'job_dir' : job_dir}

        if job_dir not in self.dirs.keys():
            self.dirs['job_dir'] = job_dir

        for dir_name in ['registered_data', 'summary', 'iters']:
            dir_key = dir_name
            if dir_key not in self.dirs.keys():
                new_dir = os.path.join(job_dir, dir_name) 
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir, exist_ok=True)
                    self.log("Created dir %s" % new_dir,2)
                # else:
                    # 
                    # self.log("Found dir %s" % new_dir,2)
                self.dirs[dir_key] = new_dir
                
            else:
                self.log("Found dir %s" % dir_name,2)
        n.save(os.path.join(job_dir, 'dirs.npy'), self.dirs)

    def run_init_pass(self):
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)
    def copy_init_pass(self,summary_old_job):
        n.save(os.path.join(self.dirs['summary'],
               'summary.npy'), summary_old_job)
    
    def register(self, tifs=None, start_batch_idx = 0):
        self.save_params()
        if tifs is None:
            tifs = self.tifs
        register_dataset(tifs, self.params, self.dirs, self.load_summary(), self.log, start_batch_idx = start_batch_idx)

    def calculate_corr_map(self, mov=None, save=True, return_mov_filt=False, mov_sub_parent_dir = None, crop=None, svd_dir=None):
        self.save_params()
        mov_sub_dir = self.make_new_dir('mov_sub', parent_dir_name=mov_sub_parent_dir)
        n.save(os.path.join(mov_sub_dir, 'params.npy'), self.params)
        self.log("Saving mov_sub to %s" % mov_sub_dir)
        if svd_dir is not None:
            self.log("Loading svd-d movie from %s" % svd_dir)
            mov = svu.reconstruct_movie(svd_dir, t_batch_size = self.params['t_batch_size'])
            self.log("Loaded svd movie of shape %s" % str(mov.shape))
        elif mov is None:
            mov = self.get_registered_movie('registered_fused_data', 'fused')
        if crop is not None:
            assert svd_dir is None, 'cant crop with svd - easy fix'
            self.params['detection_crop'] = crop
            self.save_params(dir='mov_sub')
            mov = mov[crop[0][0]:crop[0][1], :, crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
            self.log("Cropped movie to shape: %s" % str(mov.shape))
        return calculate_corrmap(mov, self.params, self.dirs, self.log, return_mov_filt=return_mov_filt, save=save)

    def detect_cells_from_patch(self, patch_idx = 0, zs=(None,None), ys=(None,None), xs=(None,None), ts=(None,None), 
                                vmap=None, mov=None, compute_npil_masks=True,n_proc = 1, vmap_mask = None):
        # print('test')
        self.save_params()
        detection_dir = self.make_new_dir('detection')
        patch_str = 'patch-%04d' % patch_idx
        patch_dir = self.make_new_dir(patch_str, parent_dir_name= 'detection')
        n.save(os.path.join(patch_dir, 'params.npy'), self.params)
        stats_path = os.path.join(patch_dir, 'stats.npy')
        info_path = os.path.join(patch_dir, 'info.npy')
        self.log("Running cell detection on patch %04d at %s, max %d iters" % (patch_idx, patch_dir, self.params['max_iter']))
        self.log("Patch bounds are %s, %s, %s" % (str(zs), str(ys), str(xs)))
        patch_info = {'zs' : zs, 'ys' : ys, 'xs' : xs, 'all_params' : self.params}
        if vmap is None:
            iter_results = self.load_iter_results(-1)
            if 'vmap' in iter_results:
                vmap = iter_results['vmap']
            else: 
                vmap = iter_results['vmap2']**0.5
        else:
            print("Something about offsets should be fixed")
            # vmap = vmap[zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
        if self.params['normalize_vmap']:
            vmap = ui.normalize_planes(vmap)
        if vmap_mask is not None:
            vmap = vmap * vmap_mask
            patch_info['vmap_mask'] = vmap_mask
        patch_info['vmap'] = vmap.copy()
        vmap = vmap[zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
        patch_info['vmap_patch'] = vmap.copy()

        if mov is None:
            mov = self.get_registered_movie('mov_sub', 'mov_sub',axis=0)
            nt, nz,ny,nx = mov.shape
            mov = mov[ts[0]:ts[1], zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
        else:
            __, nz, ny, nx = mov.shape
        if self.params['detection_timebin'] > 1:
            mov = ext.binned_mean(mov, self.params['detection_timebin'])
        try:
            self.log("Loading %.2f GB movie to memory" % (mov.nbytes/1024**3))
            mov = mov.compute()
            self.log("Loaded")
        except:
            self.log("Not a dask array")
        n.save(info_path, patch_info)
        self.log("Saving cell stats and info to %s" % patch_dir)
        # print(zs, ys, xs)
        if n_proc == 1:
            stats = ext.detect_cells(mov, vmap, **self.params, log=self.log, 
                             offset = (zs[0], ys[0], xs[0]), savepath=stats_path)
        else:
            stats = ext.detect_cells_mp(mov, vmap, **self.params, log=self.log,
                                        offset=(zs[0], ys[0], xs[0]), savepath=stats_path, n_proc=n_proc)
        if compute_npil_masks:
            self.log("Computing neuropil masks")
            if n_proc == 1:
                stats = ext.compute_npil_masks(stats, (nz,ny,nx))
            else:
                self.log("Starting MP", 1)
                stats = ext.compute_npil_masks_mp(stats, (nz,ny,nx), n_proc=n_proc)
                self.log("Ended MP", 1)
        n.save(stats_path, stats)
        return stats

    def extract_and_deconvolve(self, patch_idx=0, mov=None, batchsize_frames = 5000, stats = None, offset=None, 
                               n_frames=None, stats_dir=None, iscell = None, ts=None, load_F_from_dir=False):
        
        self.save_params()
        if stats_dir is None:
            stats_dir = self.get_patch_dir(patch_idx)
            stats, info = self.get_detected_cells(patch_idx)
            offset = (info['zs'],info['ys'],info['xs'])
        else:
            if stats is not None:
                if 'stats.npy' not in os.listdir(stats_dir):
                    self.log("Saving provided stats.npy to %s" % stats_dir)
                    n.save(os.path.join(stats_dir, 'stats.npy'), stats)
                else:
                    self.log("WARNING - overwriting with provided stats.npy in %s. Old one is in old_stats.npy" % stats_dir)
                    old_stats = n.load(os.path.join(stats_dir, 'stats.npy'),allow_pickle=True)
                    n.save(os.path.join(stats_dir, 'old_stats.npy'), old_stats)
                    n.save(os.path.join(stats_dir, 'stats.npy'), stats)
            else:
                stats = n.load(os.path.join(stats_dir, 'stats.npy'),allow_pickle=True)
        if mov is None:
            mov = self.get_registered_movie('registered_fused_data','')
        if ts is not None:
            mov = mov[:,ts[0]:ts[1]]

        if iscell is not None:
            if type(iscell) == str:
                if iscell[-4:] != '.npy': iscell += '.npy'
                iscell = n.load(os.path.join(stats_dir, iscell))
            if len(iscell.shape) > 1:
                iscell = iscell[:,-1]
            print(len(stats))
            assert iscell.shape[0] == len(stats)

            valid_stats = [stat for i,stat in enumerate(stats) if iscell[i]]
            save_iscell = os.path.join(stats_dir, 'iscell_extracted.npy')
            self.log("Extracting %d valid cells, and saving cell flags to %s" % (len(valid_stats), save_iscell))
            stats = valid_stats
            n.save(save_iscell, iscell)

        if not load_F_from_dir:
            self.log("Extracting activity")
            F_roi, F_neu = ext.extract_activity(mov, stats, batchsize_frames=batchsize_frames, offset=offset, n_frames=n_frames)
            n.save(os.path.join(stats_dir, 'F.npy'), F_roi)
            n.save(os.path.join(stats_dir, 'Fneu.npy'), F_neu)
        else:
            F_roi = n.load(os.path.join(stats_dir, 'F.npy'))
            F_neu = n.load(os.path.join(stats_dir, 'Fneu.npy'))



        self.log("Deconvolving")
        F_sub = F_roi - F_neu * self.params['npil_coeff']
        F_sub = dcnv.preprocess(F_sub, self.params['dcnv_baseline'], self.params['dcnv_win_baseline'],
                     self.params['dcnv_sig_baseline'], self.params['fs'], self.params['dcnv_prctile_baseline'])
        spks = dcnv.oasis(F_sub, batch_size = self.params['dcnv_batchsize'], tau=self.params['tau'],
                         fs=self.params['fs'])
                         
        self.log("Saving to %s" % stats_dir)
        n.save(os.path.join(stats_dir, 'spks.npy'), spks)
        
        return self.get_traces(stats_dir)

    def get_patch_dir(self, patch_idx = 0):
        patch_str = 'patch-%04d' % patch_idx
        if patch_str in self.dirs.keys():
            return self.dirs[patch_str]
        else:
            patch_dir = self.make_new_dir(patch_str, parent_dir_name= 'detection')
            return self.dirs[patch_str]
    def load_patch_results(self, patch_idx=0):
        patch_dir = self.get_patch_dir(patch_idx)
        stats = n.load(os.path.join(patch_dir, 'stats.npy'), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, 'info.npy'), allow_pickle=True).item()
        try: 
            iscell = n.load(os.path.join(patch_dir, 'iscell.npy'))
        except FileNotFoundError:
            iscell = n.ones((len(stats), 2), dtype=int)
            n.save(os.path.join(patch_dir, 'iscell.npy'))
        return stats, info, iscell

    def combine_patches(self, patch_idxs, combined_name, info_use_idx = 0, save=True,
                        extra_stats_keys = []):
        if save: combined_dir = self.make_new_dir(combined_name, parent_dir_name='detection')
        stats = []
        iscells = []
        keep_stats_keys = ['idx','threshold', 'coords', 'lam','med','peak_val']
        keep_stats_keys += extra_stats_keys

        for patch_idx in patch_idxs:
            stats_patch, info_patch, iscell = self.load_patch_results(patch_idx)
            for stat in stats_patch:
                keep_stat =  {}
                for key in keep_stats_keys:
                    keep_stat[key] = stat[key]
                stats.append(keep_stat)
            iscells.append(iscell)
            if patch_idx == patch_idxs[info_use_idx]: info = info_patch
        iscell = n.concatenate(iscells)
        # stats = n.concatenate(stats)
        self.log("Combined %d patches, %d cells" % (len(patch_idxs), len(stats)))
        if not save: 
            return stats, info, iscell
        else:
            self.log("Saving combined files to %s" % combined_dir)
            n.save(os.path.join(combined_dir, 'stats.npy'), stats)
            self.log("Saved stats", 2)
            n.save(os.path.join(combined_dir, 'iscell.npy'), iscell)
            self.log("Saved iscell", 2)
            n.save(os.path.join(combined_dir, 'info.npy'), info)
            self.log("Saved info (copied from patch) %d" % patch_idxs[info_use_idx], 2)

            return combined_dir



    def get_detected_cells(self, patch_idx = 0):
        patch_dir = self.get_patch_dir(patch_idx)
        stats = n.load(os.path.join(patch_dir, 'stats.npy'), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, 'info.npy'), allow_pickle=True).item()
        return stats, info
    def get_traces(self, patch_idx=0):
        patch_dir = self.get_patch_dir(patch_idx)
        traces = {}
        for filename in ['F.npy', 'Fneu.npy', 'spks.npy']:
            if filename in os.listdir(patch_dir):
                traces[filename[:-4]] = n.load(os.path.join(patch_dir, filename))
        return traces
        

    def get_registered_files(self, key='registered_data', filename_filter='reg_data'):
        all_files = n.os.listdir(self.dirs[key])
        reg_files = [os.path.join(self.dirs[key],x) for x in all_files if x.startswith(filename_filter)]
        return reg_files
    def get_denoised_files(self):
        all_files = n.os.listdir(self.dirs['deepinterp'])
        reg_files = [os.path.join(self.dirs['deepinterp'],x) for x in all_files if x.startswith('dp')]
        return reg_files

    def get_iter_dirs(self):
        iters_dir = self.dirs['iters']
        iter_dirs = [os.path.join(iters_dir, dir) for dir in os.listdir(iters_dir)]
        ret = []
        for dir in iter_dirs:
            if 'vmap.npy' in os.listdir(dir) or 'vmap2.npy' in os.listdir(dir):
                ret.append(dir)
        return ret
    
    def load_iter_results(self, iter_idx):
        iter_dir = self.get_iter_dirs()[iter_idx]
        self.log("Loading from %s" % iter_dir)
        res = {}
        for filename in ['vmap', 'max_img', 'mean_img', 'sum_img', 'vmap2']:
            if filename + '.npy' in os.listdir(iter_dir):
                res[filename] = n.load(os.path.join(iter_dir, filename + '.npy'))
        return res

    def fuse_registered_movie(self, n_shift, n_buf, files=None, save=False, n_proc=4, delete_original=False):
        if files is None:
            files = self.get_registered_files()
        __, xs = lbmio.load_and_stitch_full_tif_mp(
            self.tifs[0], channels=n.arange(1), get_roi_start_pix=True)
        centers = n.sort(xs)[1:]
        shift_xs = n.round(self.load_summary()[
                           'plane_shifts'][:, 1]).astype(int)
        if save:
            reg_fused_dir = self.make_new_dir('registered_fused_data')
        else:
            reg_fused_dir = ''

        # if you get an assertion error here with save=False in _get_more_data, assert left > 0
        # congratulations, you have run into a bug in Python itself! 
        # https://bugs.python.org/issue34563, https://stackoverflow.com/questions/47692566/
        # the files are too big! 
        with Pool(n_proc) as p:
            fused_files = p.starmap(fuse_and_save_reg_file, [(
                file, reg_fused_dir, centers,  shift_xs, n_shift, n_buf, None, None, save, delete_original) for file in files])
        if not save:
            fused_files = n.concatenate(fused_files, axis=1)
        return fused_files

    def get_subtracted_movie(self):
        mov_sub_paths = []
        for d in self.get_iter_dirs():
            if 'mov_sub.npy' in os.listdir(d):
                mov_sub_paths.append(os.path.join(d, 'mov_sub.npy'))

        mov_sub = u3d.npy_to_dask(mov_sub_paths, axis=0)
        return mov_sub

    def get_registered_movie(self, key='registered_data', filename_filter='reg_data', axis=1):
            paths = self.get_registered_files(key, filename_filter)
            mov_reg = u3d.npy_to_dask(paths, axis=axis)
            return mov_reg

    def load_frame_counts(self):
        return n.load(os.path.join(self.dirs['job_dir'],'frames.npy'), allow_pickle=True).item()


    def save_frame_counts(self):
        size_to_frames = {}
        nframes = []
        jobids = []
        for tif in self.tifs:
            jobids.append(int(tif.split('\\')[-2]))
            tifsize = int(os.path.getsize(tif))
            if tifsize in size_to_frames.keys():
                nframes.append(size_to_frames[tifsize])
            else:
                tf = tifffile.TiffFile(tif)
                nf = len(tf.pages) // 30
                nframes.append(nf)
                size_to_frames[tifsize] = nf
                self.log(tif +  ' is %d frames and %d bytes' % (nf, tifsize))

        nframes = n.array(nframes)
        jobids = n.array(jobids)

        tosave = {'nframes' : nframes, 'jobids' : jobids}
        self.frames = tosave
        n.save(os.path.join(self.dirs['job_dir'],'frames.npy'), tosave)

        return nframes, jobids