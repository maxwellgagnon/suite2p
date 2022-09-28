import os
import numpy as n
from . import init_pass

class Job:
    def __init__(self, root_dir, job_id, params, tifs, exist_ok=False, verbosity=1):
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
        self.params = params

        self.init_job_dir(root_dir, job_id, exist_ok=exist_ok)
        self.tifs = tifs


    def log(self, string, level=1):
        """Print messages based on current verbosity level

        Args:
            string (str): String to be printed
            level (int, optional): Level equal or below self.verbosity will be printed. Defaults to 1.
        """
        if level <= self.verbosity:
            print(string)

    def save_params(self):
        """Update saved params in job_dir/params.npy
        """
        params_path = os.path.join(self.job_dir, 'params.npy')
        n.save(params_path, self.params)
        self.log("Updated params file: %s" % params_path, 2)

    def init_job_dir(self, root_dir, job_id, exist_ok=False):
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

        os.makedirs(job_dir, exist_ok=True)
        job_summary_dir = os.path.join(job_dir,'summary')
        os.makedirs(job_summary_dir, exist_ok=True)
        job_reg_data_dir = os.path.join(job_dir,'registered_data')
        os.makedirs(job_reg_data_dir, exist_ok=True)

        self.job_dir = job_dir
        self.summary_dir = job_summary_dir
        self.reg_data_dir = job_reg_data_dir
        self.params['job_dir'] = job_dir
        self.params['job_summary_dir'] = job_summary_dir
        self.params['job_reg_data_dir'] = job_reg_data_dir
        self.save_params()

    def run_init_pass(self):
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)