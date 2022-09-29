import os
import numpy as n
from . import init_pass

class Job:
    def __init__(self, root_dir, job_id, params=None, tifs=None, exist_ok=False, verbosity=1, create=True):
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
        if create:
            self.params = params
            self.params['tifs'] = tifs
            self.tifs = tifs
            self.init_job_dir(root_dir, job_id, exist_ok=exist_ok)
        else:
            self.init_job_dir(root_dir, job_id, exist_ok=True, update_params=False)
            self.load_params()
            self.tifs = self.params.get('tifs', [])
            


    def log(self, string, level=1):
        """Print messages based on current verbosity level

        Args:
            string (str): String to be printed
            level (int, optional): Level equal or below self.verbosity will be printed. Defaults to 1.
        """
        if level <= self.verbosity:
            # print('xxx')
            print(("   " * level) + string)

    def save_params(self, new_params=None):
        """Update saved params in job_dir/params.npy
        """
        params_path = os.path.join(self.dirs['job_dir'], 'params.npy')
        if new_params is not None:
            self.params.update(new_params)
        n.save(params_path, self.params)
        self.log("Updated params file: %s" % params_path, 2)

    def load_params(self):
        params_path = os.path.join(self.dirs['job_dir'], 'params.npy')
        self.params = n.load(params_path, allow_pickle=True).item()
        self.log("Found and loaded params from %s" % params_path)

    def load_summary(self):
        summary_path = os.path.join(self.dirs['job_summary_dir'], 'summary.npy')
        summary = n.load(summary_path,  allow_pickle=True).item()
        return summary

    def init_job_dir(self, root_dir, job_id, exist_ok=False, update_params=True):
        """Create a job directory and nested dirs

        Args:
            root_dir (str): Root directory to create job_dir in
            job_id (str): Unique name for job
            exist_ok (bool, optional): If False, throws error if job_dir exists. Defaults to False.
        """

        job_dir = os.path.join(root_dir,'s3d-%s' % job_id)
        if os.path.isdir(job_dir):
            self.log("Job directory %s already exists" % job_dir, 0)
            assert exist_ok, "Manually delete job_dir, or set exist_ok=True"

        if 'dirs.npy' in os.listdir(job_dir):
            self.log("Loading dirs ")
            self.dirs = n.load(os.path.join(job_dir, 'dirs.npy'),allow_pickle=True).item()
            return

        os.makedirs(job_dir, exist_ok=True)
        job_summary_dir = os.path.join(job_dir,'summary')
        os.makedirs(job_summary_dir, exist_ok=True)
        job_reg_data_dir = os.path.join(job_dir,'registered_data')
        os.makedirs(job_reg_data_dir, exist_ok=True)
        job_iter_dir = os.path.join(job_dir,'registered_data')
        os.makedirs(job_iter_dir, exist_ok=True)

        self.dirs = {}
        self.dirs['job_dir'] = job_dir
        self.dirs['job_summary_dir'] = job_summary_dir
        self.dirs['job_reg_data_dir'] = job_reg_data_dir
        self.dirs['job_iter_dir'] = job_iter_dir
        n.save(os.path.join(job_dir, 'dirs.npy'), self.dirs)

    def run_init_pass(self):
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)