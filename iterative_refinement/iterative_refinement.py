#!/home/mabr3112/anaconda3/bin/python3.9
## ------------------------ Imports ------------------------------------------
import sys
################# PLACE PATH TO YOUR PYTHON ENV HERE ###########################
sys.path += ["/home/mabr3112/anaconda3/lib/python3.9/site-packages/"]

from collections import defaultdict
import shutil
from copy import deepcopy
import re
import itertools
import numpy as np
import pandas as pd
from subprocess import run
import os
import time
from glob import glob
import json
import Bio
from Bio import PDB
from Bio.PDB.PDBIO import PDBIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from parse_multiple_chains import parse_poses
from mpnn_scorecollector import *
import calc_composite_score
import bb_rmsd
import motif_rmsd
import mpnn_tools
import remove_index_layers
import superimposition_tools
import chainbreak_tools
from collections import defaultdict

# import custom modules
import utils.biopython_mutate
import utils.plotting as plots
import utils.biopython_tools
import utils.probs_to_resfile

## ------------------------ If you are here for RiffDiff, check the github riff_diff_protflow repository. This is a much more user-friendly implementation of the mess here ----------------
## ------------------------ Variables // Substitute these with the corresponding paths on your system --------------------------------------------------------------------------------------
__version__ = "0.1"
_mpnn_options = {"sampling_temp": "0.1", "batch_size": "1", "num_seq_per_target": "10"}
_relax_options = {} #"nstruct": "10"}
_relax_flags = {"-beta", "-overwrite"}
_rosetta_options = {}
_rosetta_flags = {"-beta"}
_proteinmpnn_path = "/home/mabr3112/ProteinMPNN/"
_of_path = "/home/mabr3112/OmegaFold/main.py"
_af2_path = "/home/hole11/software/AlphaFold2/localcolabfold/colabfold-conda/bin/"
_rosetta_paths = ["/home/markus/Rosetta/", "/home/florian_wieser/Rosetta_latest/"]
_rfdesign_python = "/home/mabr3112/anaconda3/envs/SE3-nvidia/bin/python3"
_inpaint_path = "/home/mabr3112/RFDesign/inpainting/"
_scripts_path = "/home/mabr3112/projects/iterative_refinement/rosetta_scripts/"
_esm_opts = {"chunk-size": 8, "max-tokens-per-batch": 2}
_esmfold_inference_script = "/home/mabr3112/scripts/esmfold_inference.py"
_trf_relax_script_path = "/home/mabr3112/RFDesign_old_scripts/scripts/trfold_relax.sh"
_python_path = "/home/mabr3112/anaconda3/bin/python3.9"
_hallucination_path = "/home/mabr3112/RFDesign/hallucination/"
_rfdiffusion_inference_script = "/home/mabr3112/RFdiffusion/scripts/run_inference.py"
_rfdiffusion_python_env = "/home/mabr3112/anaconda3/envs/SE3nv/bin/python3"
_ppl_script_path = "/home/mabr3112/projects/iterative_refinement/utils/calc_perplexity.py"
_entropy_script_path = "/home/mabr3112/projects/iterative_refinement/utils/esm_calc_entropy.py"
_protein_generator_model_dir = "/home/mabr3112/protein_generator/"
_protein_generator_path = "/home/mabr3112/protein_generator/slurm_inference.py"

## ------------------------ Slurm Functions ----------------------------------------

def wait_for_job(jobname: str, interval=5) -> str:
    '''
    Waits for a slurm job to be completed, then prints a "completed job" statement and returns jobname.
    <jobname>: name of the slurm job. Job Name can be set in slurm with the flag -J
    <interval>: interval in seconds, how log to wait until checking again if the job still runs.
    '''
    # Check if job is running by capturing the length of the output of squeue command that only returns jobs with <jobname>:
    while len(run(f'squeue -n {jobname} -o "%A"', shell=True, capture_output=True, text=True).stdout.strip().split("\n")) > 1:
        time.sleep(interval)
    print(f"Job {jobname} completed.\n")
    time.sleep(10)
    return jobname

def add_timestamp(x: str) -> str:
    '''
    Adds a unique (in most cases) timestamp to a string using the "time" library.
    Returns string with timestamp added to it.
    '''
    return "_".join([x, f"{str(time.time()).replace('.', '')}"])

def split_list(input_list, element_length):
    '''AAA'''
    result = []
    iterator = iter(input_list)
    while True:
        sublist = list(itertools.islice(iterator, element_length))
        if not sublist:
            break
        result.append(sublist)
    return result

def sbatch_array_jobstarter(cmds: list, sbatch_options: list, jobname="sbatch_array_job", max_array_size=10, wait=True, remove_cmdfile=True, cmdfile_dir="./"):
    '''
    Writes [cmds] into a cmd_file that contains each cmd in a separate line.
    Then starts an sbatch job running down the cmd-file.
    '''
    # check if cmds is smaller than 1000! ## TODO: if yes, split cmds and start split array!
    if len(cmds) > 1000:
        print(f"The commands-list you supplied is longer than 1000 commands. This cluster does not support arrays that are longer than 1000 commands, so your job will be subdivided into multiple arrays.")
        for sublist in split_list(cmds, 1000):
            sbatch_array_jobstarter(cmds=sublist, sbatch_options=sbatch_options, jobname=jobname, max_array_size=max_array_size, wait=wait, remove_cmdfile=remove_cmdfile, cmdfile_dir=cmdfile_dir)
        return None

    # write cmd-file
    jobname = add_timestamp(jobname)
    with open((cmdfile := f"{cmdfile_dir}/{jobname}_cmds"), 'w') as f:
        f.write("\n".join(cmds))

    # write sbatch command and run
    sbatch_cmd = f'sbatch -a 1-{str(len(cmds))}%{str(max_array_size)} -J {jobname} -vvv {" ".join(sbatch_options)} --wrap "eval {chr(92)}`sed -n {chr(92)}${{SLURM_ARRAY_TASK_ID}}p {cmdfile}{chr(92)}`"'
    print(f"\nRunning:\n{sbatch_cmd}")
    run(sbatch_cmd, shell=True, stdout=True, stderr=True, check=True)
    if wait: wait_for_job(jobname)
    if remove_cmdfile: run(f"rm {cmdfile}", shell=True, stdout=True, stderr=True, check=True)
    return None

def sbatch_jobstarter(cmd, sbatch_options: list, jobname="sbatch_job", wait=True):
    '''
    Starts an sbatch job that runs <cmd>.
    Add any <sbatch_options> as a list: ["-e err.log", "-o out.log", "..."]
    '''
    jobname = add_timestamp(jobname)
    sbatch_cmd = f"sbatch -J {jobname} {' '.join(sbatch_options)} -vvv {cmd}"
    run(sbatch_cmd, shell=True, stdout=True, stderr=True, check=True)
    if wait: wait_for_job(jobname)    
    return None

def check_for_files(files_list):
    '''Checks if any file in the list is not present'''
    for in_file in files_list:
        if not os.path.isfile(in_file): raise FileNotFoundError(f"File {in_file} not found!")

# --------------------------------------- Classes -------------------------------------

class Poses():
    '''
    Python class for poses (proteins) either as fastas or as pdbs
    Contains all information about the refinement cycle as attributes.
    
    Has to be initialized with Poses(work_dir, input_data):
        <work_dir>:   Working directory of the cycle. Within this directory, Cycle will create its cycle_dir:
                      work_dir/refinement_cycle_0001 (for n = 1)
        <input_data>: Input data, has to be list with paths to input files. Can be any file that operational functions can work on (currently .fa or .pdb files)
                      
    Attributes:
        <self.poses_df>         pd.DataFrame: This is a pandas DataFrame that contains all information about the current poses.
                                The input_poses are stored in self.poses_df[input_poses]
                                Any DataFrame that is added to self.poses_df should carry in its columns a prefix
                                of where the columns being added are coming from. Example: After running ProteinMPNN
                                for the first time, the ProteinMPNN scores will be prefixed with "mpnn_0001_".
        
        <self.scorefiles>       The Cycle Class stores the location of all scorefiles generated by operational functions
                                in a dictionary. This dictionary is stored as the attribute self.scorefiles
        
        <self.poses>            list: Contains paths to the current poses in a list. Example: ["path_to/pose_1.fa", ..., "path_to/pose_n.fa"]
        
        <self.index_layers>     int: How many index layers (_0001) have been added to the poses starting from <input_data>
    
    '''

    def __init__(self, work_dir: str, input_data: list):
        '''
        Args:
            <n>:          Iteration of the cycle.
            <work_dir>:   Working directory of refinement.
            <input_data>: Input data, has to be list with paths to input files. Can be any file that operational functions can work on.
        '''
        # fast check first, then slower one.
        if all([os.path.isfile(f) for f in input_data]):
            print(f"{len(input_data)} input poses found in input data.\n")
        else:
            check_for_files(input_data)
            raise FileNotFoundError(f"ERROR: One or more files of input data were not found at the specified location. Check input data!")

        # initialize
        self.dir = os.path.abspath(work_dir)
        self.scores_dir = f"{self.dir}/scores/"
        self.plot_dir = f"{self.dir}/plots/"
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.scores_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.poses = [input_data] if type(input_data) == str else [os.path.abspath(p) for p in input_data]
        if not self.poses: raise ValueError(f"ERROR: No poses found with {input_data}")
        self.poses_df = pd.DataFrame({"input_poses": self.poses, 
                                      "poses_description": [x.split("/")[-1].split(".")[0] for x in self.poses],
                                      "poses": self.poses
                                     })
        self.index_layers = 0
        self.scorefile = f"{self.dir}/{self.dir.split('/')[-1]}_scores.json"
        self.scorefiles = dict()
        self.auto_dump_df = True

        # Indeces for operational functions
        self.mpnn_runs = 1
        self.prediction_runs = 1
        self.relax_runs = 1
        self.filter_runs = 1
        self.rosetta_runs = 1
        self.set_poses_number = 1
        self.reindex_number = 1

        # Slurm Resource Settings
        self.max_predict_gpus = 10
        self.max_mpnn_gpus = 5
        self.max_relax_cpus = 320
        self.max_rosetta_cpus = 320
        self.max_inpaint_gpus = 10 
        self.max_diffusion_gpus = 10

        # MPNN attributes

    # ------------------ Poses Initializing functions ----------------------------------------
    def increment_attribute(self, attribute):
        if hasattr(self, attribute):
            setattr(self, attribute, getattr(self, attribute) + 1)
            return getattr(self, attribute) + 1
        else:
            setattr(self, attribute, 1)
            return 1

    # ------------------ Poses Operational functions -----------------------------------------

    def filter_poses_by_score(self, n: float, score_col: str, remove_layers=None, layer_col="poses_description", sep="_", ascending=True, prefix=None, plot=False) -> pd.DataFrame:
        '''
        Filters your current poses by a specified scoreterm down to either a fraction (of all poses) or a total number of poses,
        depending on which value was given with <n>.
        
        Args:
            <n>:                   Any positive number. Number between 0 and 1 are interpreted as a fraction to filter,
                                   Numbers >1 are interpreted as absolute number of poses after filtering.
            <score_col>:           Column name of the column that contains the scores by which the poses should be filtered.
            <plot>:                (bool, str, list) Do you want to plot filtered stats? If True, it will plot filtered scoreterm. 
                                   If str, it will look for the argument in self.poses_df and use that scoreterm for plotting.
                                   If list, it will try to plot all scoreterms in the list.
        
        To filter your DataFrame poses based on parent poses, add arguments: 
            <remove_layers>        how many index layers must be removed to reach parent pose?
            <layer_col>            column name that contains names of the poses, Default="poses_description"
            <sep>                  index layer separator for pose names in layer_col (pose_0001_0003.pdb -> '_')
            
            
        Returns filtered DataFrame. Updates Poses and pose_df.
        '''
        if score_col not in self.poses_df.columns:
            # check if any column in self.poses_df contains <score_col> and if so, return the column that has the hightest number in it (for example mpnn_run_0002)
            if any([x.endswith(score_col) for x in self.poses_df.columns]):
                score_col = sorted([x for x in list(self.poses_df.columns) if x.endswith(score_col)])[-1]
            else:
                raise KeyError(f"ERROR: Scoreterm {score_col} not found in DataFrame. Available Scoreterms: {', '.join(list(self.poses_df.columns))}")

        # store current self.poses_df as prefilter_1_scores.json
        output_name = prefix or f"filter_run_{str(self.filter_runs).zfill(4)}"
        if not os.path.isfile((st := f"{self.scores_dir}/{output_name}.json")): self.poses_df.to_json(st)
        self.increment_attribute("filter_runs")

        # filter df down by n (either fraction if n < 0, or number of filtered poses if n > 1)
        n = determine_filter_n(self.poses_df, n)

        # Filter poses_df down to the number of poses specified with <n>
        orig_len = str(len(self.poses_df))
        filter_df = filter_dataframe(df=self.poses_df, col=score_col, n=n, remove_layers=remove_layers, layer_col=layer_col, sep=sep, ascending=ascending)
        print(f"Filtered poses from {orig_len} to {str(len(filter_df))} structures.")

        # create filter-plots if specified.
        if plot:
            columns = plots.parse_cols_for_plotting(plot, subst=score_col)
            plots.violinplot_multiple_cols_dfs(dfs=[self.poses_df, filter_df], df_names=["Before Filtering", "After Filtering"],
                                               cols=columns, titles=columns, y_labels=columns, out_path=f"{self.plot_dir}/{output_name}.png")

        # update object attributs [poses_df]
        self.poses_df = filter_df
        self.update_poses()

        # if argument self.auto_dump_df is set to True, dump the new poses_df in self.dir
        if self.auto_dump_df: self.poses_df.to_json(self.scorefile)

        return filter_df

    def mpnn_design(self, mpnn_options="", pose_options=None, mpnn_scorefile="mpnn_scores.json", prefix=None,
                    fixed_positions_col=None, tied_positions_col=None, design_chains=None, use_soluble_model=False) -> str:
        '''
        Runs ProteinMPNN on all structures in <poses> with specified <mpnn_options>.
        
        Args:
            <mpnn_options>              Commandline options in string format that shall be passed to ProteinMPNN for every Pose.
            <pose_options>              List of commandline options (string) that shall be passed to ProteinMPNN. List needs to be same length as poses.
            <mpnn_scorefile>            Set a custom scorefile name (default: mpnn_scores.json)
            <prefix>                    Set a prefix for this poses operation. This will also be the name of the working directory for this operation.
            <fixed_positions_col>       Column name in self.poses_df that contains dictionaries with fixed_positions dictionaries for ProteinMPNN.
            <tied_positions_col>        Column name in self.poses_df that contains dictionaries with tied_positions dictionaries for ProteinMPNN.
        
        Increments the self.mpnn_runs and self.index_layers attribute by one.
        Adds mpnn_output_0001 as attribute to self.
        Updates poses attribute to location of *.fa files generated by ProteinMPNN.
        
        Returns: dir
            Returns the directory that stores individual .fa files.
        '''
        # Prepare mpnn_options (make sure that all necessary options are present) and check if poses are .fa files
        mpnn_options = parse_options_string(mpnn_options)
        mpnn_options = prep_options(mpnn_options, _mpnn_options)
        if not all([x.endswith(".pdb") for x in self.poses]): raise TypeError(f"ERROR: your current poses are not .pdb files! They must be .pdb files to run ProteinMPNN!")

        # Setup mpnn_run directory
        mpnn_run = prefix or f"mpnn_run_{str(self.mpnn_runs).zfill(4)}"
        abs_mpnn_dir = f"{self.dir}/{mpnn_run}"
        mpnn_options["out_folder"] = abs_mpnn_dir
        if not os.path.isdir(abs_mpnn_dir): os.makedirs(abs_mpnn_dir, exist_ok=True)

        # Parse pose_options into dictionaries if pose_options is set. Otherwise create empty dictionaries for each pose (no pose_options)
        if pose_options: pose_options = [parse_options_string(options_string) for options_string in pose_options]
        else: pose_options = [{} for pose in self.poses_df["poses_description"]]

        # if fixed_positions option is set, write fixed_positions_jsonl file and add it to mpnn_options:
        if fixed_positions_col:
            fixed_positions_filename = f"{mpnn_options['out_folder']}/fixed_positions.jsonl"
            if not os.path.isfile(fixed_positions_filename):
                # check if fixed_positions in fixed_positions_col are valid
                fixed_positions_list = check_fixed_positions(list(self.poses_df[fixed_positions_col]))
                ffn = write_keysvalues_to_file(keys=list(self.poses_df["poses_description"]),
                                               values=fixed_positions_list,
                                               outfile_name=fixed_positions_filename)
            mpnn_options["fixed_positions_jsonl"] = fixed_positions_filename

        # if tied_positions option is set, write tied_positions_jsonl file and add it to mpnn_options:
        if tied_positions_col:
            mpnn_options["tied_positions_jsonl"] = tied_positions_filename = f"{mpnn_options['out_folder']}/tied_positions.jsonl"
            if not os.path.isfile(tied_positions_filename): write_keysvalues_to_file(keys=list(self.poses_df["poses_description"]), values=list(self.poses_df[tied_positions_col]), outfile_name=tied_positions_filename)

        # if design_chains is set, write design_chains_jsonl file for ProteinMPNN:
        if design_chains:
            it = mpnn_tools.check_design_chains(design_chains, self.poses_df["poses_description"])
            mpnn_options["chain_id_jsonl"] = self.poses_mpnn_jsonl_writer(design_chains, f"{mpnn_options['out_folder']}/design_chains.jsonl", iterate=it)

        # Run ProteinMPNN function that handles ProteinMPNN and check if ProteinMPNN was successful:
        scores = proteinmpnn(self.poses_df["poses"].to_list(), mpnn_options, pose_options=pose_options, max_gpus=self.max_mpnn_gpus, scorefile=mpnn_scorefile, use_soluble_model=use_soluble_model)
        if not glob((mpnn_globstr := f"{mpnn_options['out_folder']}/seqs/*.fa")):
            raise FileNotFoundError(f"No *.fa files found in {mpnn_globstr}. ProteinMPNN did not run properly.")

        # Update attributes: [scorefiles, poses, index_layers, mpnn_runs]
        self.scorefiles[mpnn_run] = f"{abs_mpnn_dir}/{mpnn_scorefile}"
        self.poses = list(scores["location"])
        self.increment_attribute("index_layers")
        self.increment_attribute("mpnn_runs")
        scores = pd.DataFrame(scores).add_prefix((mpnn_prefix := f"{mpnn_run}_"))
        scores = update_df(scores, f"{mpnn_prefix}description", self.poses_df, "poses_description", new_df_col_remove_layer=1)

        # update poses_description column in scores and update poses_df
        scores.loc[:,"poses_description"] = scores.loc[:, f"{mpnn_prefix}description"]
        self.poses_df = scores
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{mpnn_prefix}location"]
        self.poses_df.to_json(self.scorefile)

        return f"{abs_mpnn_dir}/seqs"

    def get_mpnn_probs(self, prefix:str) -> "list[str]":
        '''AAA'''
        # sanity
        if prefix in self.poses_df.columns: raise ValueError(f":prefix: {prefix} already used for mpnn probabilities. Use different prefix!")
        if any([not pose.endswith(".pdb") for pose in self.poses_df["poses"]]): raise RuntimeError(f"There are non-pdb files in your poses DataFrame! Cannot run probabilities on non-pdb files!")

        # check if results are already present
        if not os.path.isdir((probdir := f"{self.dir}/{prefix}_probabilities")): os.makedirs(probdir)
        if len(glob(f"{probdir}/unconditional_probs_only/*.npz")) == len(self.poses_df):
            print(f"Outputs of get_mpnn_probs already found at {probdir}/unconditional_probs_only/. Skipping step.")
            self.poses_df[prefix] = f"{probdir}/unconditional_probs_only/" + self.poses_df["poses_description"] + ".npz"
        else:
            # collect list of files:
            self.poses_df[prefix] = f"{probdir}/unconditional_probs_only/" + self.poses_df["poses_description"] + ".npz"
            self.poses_df[f"{prefix}_files_missing"] = self.poses_df[prefix].apply(lambda filepath: not os.path.isfile(filepath))

            while len(self.poses_df.loc[self.poses_df[f"{prefix}_files_missing"]]) > 0:
                remaining_df = self.poses_df.loc[self.poses_df[f"{prefix}_files_missing"]]
                cmds = [f"{_python_path} {_proteinmpnn_path}/protein_mpnn_run.py --unconditional_probs_only 1 --pdb_path {pose} --out_folder {probdir}" for pose in remaining_df["poses"].to_list()]

                # execute
                sbatch_array_jobstarter(cmds, sbatch_options=[f"--gpus-per-node 1 -c2 -e {probdir}/ProteinMPNN_err.log", f"-o {probdir}/ProteinMPNN_out.log"], jobname="mpnn_probs", max_array_size=self.max_mpnn_gpus, wait=True, cmdfile_dir=probdir, remove_cmdfile=False)

                # check if all outputs are present, if not, rerun MPNN: ##! This is here because some ProteinMPNN runs fail sometimes.
                self.poses_df[f"{prefix}_files_missing"] = self.poses_df[prefix].apply(lambda filepath: not os.path.isfile(filepath))

        # return output
        return self.poses_df[prefix].to_list()

    def write_resfiles_from_mpnn_probs(self, colname: str, probabilities_location_col:str, threshold:float=0.1, motif_threshold:float=0.01, motif_col:str=None, motif_chain:str=None) -> "list[str]":
        '''
        Takes output column of method Poses.get_mpnn_probs() :probabilities_location_col: and writes resfiles. Resfile locations are stored in the Poses.poses_df under column :colname:.
        '''
        # sanity
        if colname in self.poses_df.columns: raise ValueError(f":prefix: {colname} already used. Use different colname!")

        # convert resfiles
        resfile_paths = [utils.probs_to_resfile.write_probs_to_resfile_from_path(row[probabilities_location_col], threshold=threshold, motif_threshold=motif_threshold, motif_list=row[motif_col][motif_chain]) for i, row in self.poses_df.iterrows()]

        # update poses_df and return:
        self.poses_df[colname] = resfile_paths
        return resfile_paths

    def poses_mpnn_jsonl_writer(self, values, json_filename, iterate=False):
        '''
        
        '''
        # Check in self.poses_df for column with name <values> from which to retreive values list.
        if isinstance(values, str):
            values = list(self.poses_df[values])
            iterate = True

        # write jsonl file:
        return mpnn_tools.write_mpnn_jsonl(list(self.poses_df["poses_description"]), values=values, json_filename=json_filename, iterate=iterate)

    def relax_poses(self, relax_options="", pose_options=None, n=10, prefix=None, filter_scoreterm="total_score", remove_bad_decoys=True):
        '''
        Runs (default) 10 relax trajectories of each pose and sets the output with the lowest energy as the new pose.
        This version of relax does not use nstruct, as nstruct without mpirun is limited to one cpu which is inefficient on the cluster!
        
        Args:
            <relax_options>         Commandline options that you want to use to run Rosetta Relax.
                                    Default options are stored in the global variable _relax_options.
            <pose_options>          List of commandline options (sorted by self.poses_df["poses_description"]) 
                                    that can be supplied to the poses individually.
            <n>                     How many relax-trajectories to run. If this option is set, it overwrites 'nstruct' given in <relax_options>!
            <prefix>                Set a custom prefix which will be added to all relax_scores when they are written into poses_df.
                                    By default, it will add relax_poses_0001.
                                    ! Prefix also determines the name of the directory in which relax results will be stored !
            <filter_scoreterm>      The scoreterm that will be used to select the lowest energy pose from all relax trajectories of each pose.
                                    (Default: total_score)
        '''
        # Setup relax_run directory
        if not prefix: prefix = f"relax_run_{str(self.relax_runs).zfill(4)}"
        self.increment_attribute("relax_runs")
        relax_dir = f"{self.dir}/{prefix}"

        # Parse pose_options into dictionaries if pose_options is set. Otherwise create empty dictionaries for each pose (no pose_options)
        if pose_options: pose_options = [parse_rosetta_options_string(options_string) for options_string in pose_options]
        else: pose_options = [({}, []) for pose in self.poses_df["poses_description"]]

        # Prepare relax options
        relax_options, relax_flags = parse_rosetta_options_string(relax_options)
        relax_flags = list(set(relax_flags) | _relax_flags) # combine with set operation and reconvert them into a list.
        relax_options = prep_options(relax_options, _relax_options)

        # Run Rosetta Relax with relax_options and check if run was successful
        relax_output = run_relax(self.poses_df["poses"].to_list(), relax_options, relax_flags, pose_options=[x[0] for x in pose_options], pose_flags=[x[1] for x in pose_options],
                                 n=n, work_dir=relax_dir, scorefile="relax_scores.sc", max_cpus=self.max_relax_cpus)

        # Filter decoys down to the lowest score decoy of each pose
        relax_scores = relax_output["scores"]
        filtered_poses = filter_dataframe(relax_scores, filter_scoreterm, n=1, remove_layers=1, layer_col="description", sep="_", ascending=True)

        if remove_bad_decoys:
            # check if files were already removed:
            if len(glob(f"{relax_dir}/*.pdb")) > len(filtered_poses):
                # Collect DataFrame that contains all but the filtered poses.
                remove_pdbs = relax_scores[~relax_scores["description"].isin(filtered_poses["description"])]

                # remove all .pdb files that are not top decoys from pdb_dir
                print(f"Deleting higher energy relax decoys")
                for pdb in list(remove_pdbs["location"]):
                    try:
                        os.remove(pdb)
                    except FileNotFoundError:
                        print(f"WARNING: Trying to remove file that does not exist anymore: {pdb}")

        # Update poses_df and poses
        self.poses = list(filtered_poses["location"])
        filtered_poses = filtered_poses.add_prefix(prefix+"_")
        self.poses_df = update_df(filtered_poses, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = filtered_poses[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        return relax_output["pdb_dir"]

    def create_relax_decoys(self, relax_options="", pose_options=None, n=10, prefix=None) -> str:
        '''
        Runs (default) 10 relax trajectories of each pose and stores all poses from the relax run.
        This version of relax does not use nstruct, as nstruct without mpirun is limited to one cpu which is inefficient on the cluster!
        
        Args:
            <relax_options>         Commandline options that you want to use to run Rosetta Relax.
                                    Default options are stored in the global variable _relax_options.
            <n>                     How many relax-trajectories to run. If this option is set, it overwrites 'nstruct' given in <relax_options>!
            <prefix>                Set a custom prefix which will be added to all relax_scores when they are written into poses_df.
                                    By default, it will add relax_poses_0001.
                                    ! Prefix also determines the name of the directory in which relax results will be stored !
        '''
        # Setup relax_run directory
        if not prefix: prefix = f"relax_run_{str(self.relax_runs).zfill(4)}"
        self.increment_attribute("relax_runs")
        relax_dir = f"{self.dir}/{prefix}"

        # Parse pose_options into dictionaries if pose_options is set. Otherwise create empty dictionaries for each pose (no pose_options)
        if pose_options: pose_options = [parse_rosetta_options_string(options_string) for options_string in pose_options]
        else: pose_options = [({}, []) for pose in self.poses_df["poses_description"]]

        # Prepare relax options
        relax_options, relax_flags = parse_rosetta_options_string(relax_options)
        relax_flags = list(set(relax_flags) | _relax_flags) # combine with set operation and reconvert them into a list.
        relax_options = prep_options(relax_options, _relax_options)

        # Run Rosetta Relax with relax_options and check if run was successful
        relax_output = run_relax(self.poses_df["poses"].to_list(), relax_options, relax_flags, pose_options=[x[0] for x in pose_options], pose_flags=[x[1] for x in pose_options],
                                 n=n, work_dir=relax_dir, scorefile="relax_scores.sc", max_cpus=self.max_relax_cpus)

        # Update poses_df and poses
        scores = relax_output["scores"]
        self.poses = list(scores["location"])
        scores = scores.add_prefix(prefix+"_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        return relax_output["pdb_dir"]

    def predict_sequences(self, predict_function, options="", pose_options=None, prefix=None, af2_rename_reference_col:str=None) -> str:
        '''
        Predicts sequences given by <poses> with <predict_function>.
        <options> contains all options needed for running <predict_function>.
        <options> should be a string in the style of command-line options that would be passed to the prediction model.
        
        <predict_function>: can be one of [run_AlphaFold2, run_OmegaFold]
        <af2_rename_reference_col>: column in self.poses_df that contains reference pose paths.
        
        Returns directory that contains all predicted .pdbs
        '''
        # sanity checks
        if all([x.endswith(".fa") for x in self.poses_df["poses"].to_list()]):
            pass
        elif all([x.endswith(".pdb") for x in self.poses_df["poses"].to_list()]):
            self.poses_pdb_to_fasta()

        # create prediction dir
        predict_run = prefix or f"predict_run_{str(self.prediction_runs).zfill(4)}"
        abs_predict_dir = f"{self.dir}/{predict_run}"

        # prepare options
        options = parse_options_string(options, sep="--")
        options["output_dir"] = abs_predict_dir

        # run prediction with <predict_function>
        prediction_outputs = predict_function(self.poses_df['poses'].to_list(), options, pose_options=pose_options, max_gpus=self.max_predict_gpus)
        scores = prediction_outputs["scores"] ## lazy programming

        # update attributes:
        self.increment_attribute("prediction_runs")
        scores = scores.add_prefix(predict_run+"_")
        self.poses_df = update_df(scores, f"{predict_run}_description", self.poses_df, "poses_description", new_df_col_remove_layer=0)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{predict_run}_location"]
        self.poses = self.poses_df['poses']
        self.poses_df.to_json(self.scorefile)

        # rename af2 preds if option is set:
        if af2_rename_reference_col: [utils.biopython_tools.rename_pdb_chains_pdbfile(pdb, ref_pdb) for pdb, ref_pdb in zip(self.poses_df['poses'].to_list(), self.poses_df[af2_rename_reference_col].to_list())]

        return prediction_outputs["pdb_dir"]

    def poses_pdb_to_fasta(self, chain_sep=":") -> list[str]:
        '''
        Extracts sequences from self.poses if poses are .pdb files.
        Stores .fa files where poses are stored.
        Args:
            <chain_sep>              Separator with which the chains should be joined if multiple chains are in one object.
        
        returns poses
        '''
        # sanity check
        if any([not x.endswith(".pdb") for x in self.poses]):
            raise TypeError(f"ERROR: Not all poses are .pdb files! Check poses objects.")

        # Start the parser
        pdb_parser = Bio.PDB.PDBParser(QUIET = True)
        ppb = Bio.PDB.PPBuilder()

        # Get the structures
        poses_l = [pdb_parser.get_structure(pose, pose) for pose in self.poses]

        # collect the sequences
        sequences = [chain_sep.join([str(x.get_sequence()) for x in ppb.build_peptides(pose)]) for pose in poses_l]

        # write fasta-files
        renaming_dict = dict()
        for pose, seq in zip(self.poses_df["poses"].to_list(), sequences):
            fasta_name = pose.replace(".pdb", ".fa")
            description = pose.split("/")[-1].split(".")[0]
            with open(fasta_name, 'w') as f:
                f.write(f">{description}\n{seq}")
            renaming_dict[pose] = fasta_name

        # set fasta_files as new poses
        self.poses_df.loc[:, "poses"] = [renaming_dict[pose] for pose in list(self.poses_df["poses"])]
        self.poses = list(self.poses_df["poses"])

        return self.poses

    def rosetta(self, executable: str, options="", pose_options=None, n=10, prefix=None, force_options=None) -> str:
        '''
        Runs Rosetta Executable on poses.
        Commandline arguments can be provided with <options>. <n> overwrites -nstruct commandline option.
        Pose options can be provided as a dictionary {pose: "pose_option"}
        '''
        # Setup relax_run directory
        if not prefix: prefix = f"rosetta_run_{str(self.rosetta_runs).zfill(4)}"
        self.increment_attribute("rosetta_runs")
        rosetta_dir = f"{self.dir}/{prefix}"

        # Parse pose_options into dictionaries if pose_options is set. Otherwise create empty dictionaries for each pose (no pose_options)
        if pose_options: pose_options = [parse_rosetta_options_string(options_string) for options_string in pose_options]
        else: pose_options = [({}, []) for pose in self.poses_df["poses_description"]]

        # Prepare Rosetta options
        rosetta_options, rosetta_flags = parse_rosetta_options_string(options)
        rosetta_flags = list(set(rosetta_flags) | _rosetta_flags) # combine with set operation and reconvert them into a list.
        rosetta_options = prep_options(rosetta_options, _rosetta_options)

        # Run Rosetta Relax with relax_options and check if run was successful
        rosetta_output = run_rosetta(self.poses_df["poses"].to_list(), executable, rosetta_options, rosetta_flags, pose_options=[x[0] for x in pose_options], pose_flags=[x[1] for x in pose_options],
                                    n=n, work_dir=rosetta_dir, scorefile="rosetta_scores.sc", max_cpus=self.max_rosetta_cpus, force_options=force_options)

        # Update poses_df and poses
        scores = rosetta_output["scores"]
        #self.poses = list(scores["location"])
        scores = scores.add_prefix(prefix+"_")
        print(len(self.poses_df))
        print(len(scores))
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses = self.poses_df["poses"].to_list()
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        return rosetta_output["pdb_dir"]

    def inpaint(self, options="", pose_options=None, prefix=None, perres_lddt=False, perres_inpaint_lddt=False, trf_relax=True, calc_chainbreak=True) -> dict:
        '''
        <pose_options> overwrite global <options>
        Args:
            <options>
            <pose_options>
            <prefix>
            <perres_lddt>
            <perres_inpaint_lddt>
        
        Returns:
            Directory where current poses are stored at.
        '''
        # setup directory
        self.increment_attribute("inpainting_runs")
        prefix = prefix or f"inpainting_run_{str(self.inpainting_runs).zfill(4)}"
        inpainting_dir = f"{self.dir}/{prefix}"

        # run inpainting
        inpainting_output = run_inpainting(self.poses_df["poses"].to_list(), work_dir=inpainting_dir, options=options, pose_options=pose_options,
                                           scorefile="inpainting_scores.json", max_gpus=self.max_inpaint_gpus, perres_lddt=perres_lddt, perres_inpaint_lddt=perres_inpaint_lddt)

        # update poses_df and poses
        scores = inpainting_output["scores"]
        self.poses = list(scores["location"])
        scores = scores.add_prefix(prefix+"_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        # run trf_relax if specified:
        if trf_relax:
            trf_dir = trf_relax_dir(inpainting_output["pdb_dir"])

            # calculate rmsds and integrate into poses_df
            rmsds = calc_inpaint_trf_motif_rmsd(inpaint_dir=inpainting_output["pdb_dir"], trf_dir=trf_dir, out_scorefile=f"{inpainting_dir}/trf_motif_ca_rmsd.json")
            rmsds = rmsds.add_prefix(f"{prefix}_")
            self.poses_df = update_df(rmsds, f"{prefix}_description", self.poses_df, "poses_description")

            # update poses_location
            self.new_poses_path(trf_dir)
            inpainting_output["pdb_dir"] = trf_dir
            self.poses_df[f"{prefix}_trf_relax_location"] = self.poses_df["poses"].to_list()

            if calc_chainbreak: self.check_inpaint_chainbreak(prefix)

        return inpainting_output["pdb_dir"]

    def hallucinate(self, options="", pose_options=None, prefix=None, trf_relax=True, calc_chainbreak=False) -> dict:
        '''
        <pose_options> overwrite global <options>
        Args:
            <options>
            <pose_options>
            <prefix>
            <perres_lddt>
            <perres_inpaint_lddt>
        
        Returns:
            Directory where current poses are stored at.
        '''
        # setup directory
        self.increment_attribute("hallucination_runs")
        prefix = prefix or f"hallucination_run_{str(self.hallucination_runs).zfill(4)}"
        hallucination_dir = f"{self.dir}/{prefix}"

        # run inpainting
        hallucination_output = run_hallucination(self.poses_df["poses"].to_list(), work_dir=hallucination_dir, options=options, pose_options=pose_options,
                                           scorefile="hallucination_scores.json", max_gpus=self.max_inpaint_gpus)

        # update poses_df and poses
        scores = hallucination_output["scores"]
        self.poses = list(scores["location"])
        scores = scores.add_prefix(prefix+"_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        # run trf_relax if specified:
        if trf_relax:
            trf_dir = trf_relax_dir(hallucination_output["pdb_dir"])

            # calculate rmsds and integrate into poses_df
            rmsds = calc_inpaint_trf_motif_rmsd(inpaint_dir=hallucination_output["pdb_dir"], trf_dir=trf_dir, out_scorefile=f"{hallucination_dir}/trf_motif_ca_rmsd.json")
            rmsds = rmsds.add_prefix(f"{prefix}_")
            self.poses_df = update_df(rmsds, f"{prefix}_description", self.poses_df, "poses_description")

            # update poses_location
            self.new_poses_path(trf_dir)
            hallucination_output["pdb_dir"] = trf_dir
            self.poses_df[f"{prefix}_trf_relax_location"] = self.poses_df["poses"].to_list()

            if calc_chainbreak: self.check_inpaint_chainbreak(prefix)

        return hallucination_output["pdb_dir"]

    def rfdiffusion(self, options="", pose_options=None, prefix=None, max_gpus:int=10) -> str:
        '''runs RFDiffusion for you on acluster.'''
        # setup directory
        self.increment_attribute("diffusion_runs")
        prefix = prefix or f"diffusion_run_{str(self.diffusion_runs).zfill(4)}"
        rfdiff_dir = f"{self.dir}/{prefix}"

        # run diffusion
        rfdiff_output = run_rfdiffusion(self.poses_df["poses"].to_list(), work_dir=rfdiff_dir, options=options, pose_options=pose_options, scorefile="rfdiffusion_scores.json", max_gpus=max_gpus)

        # update poses_df and poses
        scores = rfdiff_output["scores"]
        scores = scores.add_prefix(prefix + "_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)
        self.poses = list(self.poses_df["poses"])

        return rfdiff_output["pdb_dir"]

    def protein_generator(self, prefix:str, options="", pose_options=None, max_cores:int=10) -> str:
        '''runs protein_generator for you on acluster'''
        # setup
        protein_generator_dir = f"{self.dir}/{prefix}"

        # run diffusion
        protein_generator_output = run_protein_generator(self.poses_df["poses"].to_list(), work_dir=protein_generator_dir, options=options, pose_options=pose_options, scorefile="protein_generator_scores.json", max_cores=max_cores)

        # update poses_df and poses
        scores = protein_generator_output["scores"]
        scores = scores.add_prefix(prefix + "_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)
        self.poses = list(self.poses_df["poses"])

        return protein_generator_output["pdb_dir"]

    def update_motif_res_mapping(self, motif_col: str, inpaint_prefix: str) -> None:
        '''AAA'''
        # check if prefix occurs in DF
        if not col_with_prefix_exists_in_df(self.poses_df, motif_col): raise KeyError(f"Prefix {motif_col} not found in poses_df. Available columns: {', '.join(self.poses_df.columns)}")
        if not col_with_prefix_exists_in_df(self.poses_df, inpaint_prefix): raise KeyError(f"Prefix {inpaint_prefix} not found in poses_df. Available columns: {', '.join(self.poses_df.columns)}")

        self.poses_df[motif_col] = [reassign_motif(motif, ref_pdb_idx, hal_pdb_idx) for motif, ref_pdb_idx, hal_pdb_idx in zip(list(self.poses_df[motif_col]), list(self.poses_df[f"{inpaint_prefix}_con_ref_pdb_idx"]), list(self.poses_df[f"{inpaint_prefix}_con_hal_pdb_idx"]))]
        return None

    def update_res_identities(self, identity_col: str, inpaint_prefix: str) -> None:
        '''AAA'''
        # check if prefix occurs in DF
        if not col_with_prefix_exists_in_df(self.poses_df, identity_col): raise KeyError(f"Prefix {identity_col} not found in poses_df. Available columns: {', '.join(self.poses_df.columns)}")
        if not col_with_prefix_exists_in_df(self.poses_df, inpaint_prefix): raise KeyError(f"Prefix {inpaint_prefix} not found in poses_df. Available columns: {', '.join(self.poses_df.columns)}")

        self.poses_df[identity_col] = [reassign_identity_keys(motif, ref_pdb_idx, hal_pdb_idx) for motif, ref_pdb_idx, hal_pdb_idx in zip(list(self.poses_df[identity_col]), list(self.poses_df[f"{inpaint_prefix}_con_ref_pdb_idx"]), list(self.poses_df[f"{inpaint_prefix}_con_hal_pdb_idx"]))]
        return None

    def check_inpaint_chainbreak(self, inpaint_prefix: str) -> None:
        '''
        '''
        # check if prefix occurs in DF
        if not col_with_prefix_exists_in_df(self.poses_df, inpaint_prefix): raise KeyError(f"Prefix {inpaint_prefix} not found in poses_df. Available columns: {', '.join(self.poses_df.columns)}")

        # add chainbreak value to poses_df
        self.poses_df[f"{inpaint_prefix}_chainbreak"] = [chainbreak_tools.search_chainbreak_in_pdb(relaxed_pose, chainbreak_tools.get_linker_contig(pose.replace(".pdb", ".trb"))) for pose, relaxed_pose in zip(list(self.poses_df[f"{inpaint_prefix}_location"]), list(self.poses_df[f"{inpaint_prefix}_trf_relax_location"]))]

        self.poses_df.to_json(self.scorefile)
        return None

    def thread_sequences(self, sequence_col:str, template_dir:str, remove_layers:int=1, prefix:str=None, scripts_version="rosetta_scripts.default.linuxgccrelease") -> str:
        '''
        Threads generated sequences onto poses found in <template_dir>
        Args:
            <template_dir>
            <remove_layers>
            <prefix>
            
        Returns:
            Directory where threaded poses are stored.
            
        '''
        raise NotImplementedError()
        def threading_read_pose_seq(path):
            with open(path, 'r') as f:
                return "".join([line.strip() for line in f.readlines()[1:]])

        # setup directory and prefix
        self.increment_attribute("threading_runs")
        prefix = prefix or f"threading_run_{str(self.threading_runs).zfill(4)}"
        thread_dir = f"{self.dir}/{prefix}"
        if not os.path.isdir(thread_dir): os.makedirs(thread_dir, exist_ok=True)
        os.makedirs((ros_dir := f"{thread_dir}/raw/"), exist_ok=True)

        # copy templates into temp_thread_dir
        os.makedirs((temp_thread_dir := f"{thread_dir}/temp_thread_dir"), exist_ok=True)
        poses = [x.split("/")[-1] for x in self.poses]
        if remove_layers:
            copylist = [(f"{template_dir}/{'_'.join(pose.split('_')[:-1*remove_layers])}.pdb", f"{temp_thread_dir}/{pose.replace('.fa', '.pdb')}") for pose in poses]
        else:
            copylist = [(f"{template_dir}/{pose.replace('.fa', '.pdb')}", f"{temp_thread_dir}/{pose.replace('.fa', '.pdb')}") for pose in poses]

        for k, v in copylist:
            shutil.copy(k, v)

        # Prepare Rosetta options
        options = ""
        rosetta_options, rosetta_flags = parse_rosetta_options_string(options)
        rosetta_flags = list(set(rosetta_flags) | _rosetta_flags) # combine with set operation and reconvert them into a list.
        rosetta_options = prep_options(rosetta_options, _rosetta_options)

        # parse pose_options (sequences to thread)
        poses = self.poses_df[sequence_col]
        poses_pdb = [f"{temp_thread_dir}/{pose.replace('.fa', '.pdb')}" for pose in poses]
        pose_options = [{"parser:script_vars": f"seq={threading_read_pose_seq(pose)}"} for pose in self.poses]

        # run_rosetta on templates with pose_sequences
        thread_output = dict()
        #thread_output = run_rosetta(poses=poses_pdb, rosetta_executable=scripts_version, options="-parser:protocol {_scripts_path}/thread_repack.xml -beta",
        #                            pose_options=pose_options, n=1, work_dir=ros_dir,
        #                            scorefile="rosetta_scores.sc", max_cpus=self.max_rosetta_cpus)

        # update self.poses and self.poses_df attributes
        scores = thread_output["scores"]
        self.poses = list(scores["location"])
        scores = scores.add_prefix(prefix+"_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=1)
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]
        self.poses_df.to_json(self.scorefile)

        # remove index layer that was added by Rosetta
        self.reindex_poses(out_dir=thread_dir, remove_layers=1, force_reindex=False)
        shutil.rmtree(temp_thread_dir)

        return thread_output["pdb_dir"]

    # ----------------------------------Poses Metrics -----------------------------------------------

    def calc_metric(self, metric_function, metric_prefix: str, metric_args=[], metric_kwargs=None, pose_col="description") -> None:
        '''
        #### This function has very specific, but annoying behavior, needs to be refactored!!!

        Calculates a metric using <metric_function> that is given <metric_kwargs>
        calc_metric will update self.poses_df by merging calc_metric scores with self.poses_df on <metric_prefix>_<pose_col>
        
        Args:
            <metric_function>:   Has to be a function that takes it's arguments in this form: ([path_to_structures], {options})
                                 Additionally it has to return a pd.DataFrame containing metric scores, where <pose_col>
                                 identifies the structures ("description")
            <metric_prefix>:     Prefix that will be added to the scores calculated by metric_function when merged with self.poses_df
        '''
        # calculate metric
        metric_args = self.poses_df["poses"].to_list() + metric_args
        metric_kwargs = metric_kwargs or {}
        scores = metric_function(*metric_args, **metric_kwargs)

        # add metric_prefix to scores and update self.poses_df
        scores = scores.add_prefix(metric_prefix + "_")
        self.poses_df = update_df(scores, f"{metric_prefix}_{pose_col}", self.poses_df, "poses_description", new_df_col_remove_layer=0)
        self.poses_df.to_json(self.scorefile)

        return None

    def calc_metric_from_function(self, metric_function, score_name:str, metric_args=[], metric_kwargs={}) -> None:
        '''AAA'''
        # calculate metric
        self.poses_df[score_name] = [metric_function(pose, *metric_args, **metric_kwargs) for pose in self.poses_df["poses"].to_list()]
        return None

    def calc_bb_rmsd_df(self, ref_pdb, metric_prefix:str, atoms:list=["CA"]):
        '''
        Calculates bb_ca_rmsd of latest poses in self.poses_df to "ref_pdb_col" (which should be e.g. predict_run_0001_location)
        '''
        # define metric name:
        metric_name = f"{metric_prefix}_bb_ca_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # parse ref_pdb option
        ref_pdb_l = parse_pose_options(ref_pdb, self.poses_df)

        # calculate RMSDs
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            print(f"RMSDs found at {scorefile} -> Reading RMSDs directly from file.")
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            if len(self.poses_df) == 0: raise ValueError("ERROR: Length of DataFrame = 0. DataFrame merging failed!")
            rmsds = list(self.poses_df[metric_name])
        else:
            rmsds = [bb_rmsd.superimpose_calc_rmsd(ref_pdb, pose, atoms=atoms) for ref_pdb, pose in zip(ref_pdb_l, self.poses_df["poses"].to_list())]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, f"{metric_prefix}_bb_ca_rmsd"] = rmsds
            self.poses_df.to_json(self.scorefile)

        return rmsds

    def calc_motif_heavy_rmsd_df(self, ref_pdb: str, ref_motif: str, target_motif: str, metric_prefix: str) -> list:
        '''
            Calculates rmsd of current poses to reference pdbs which path can be found in 'frag_col'.
            Args:
                <ref_col>                Column of self.poses_df containing path to reference pdbs
                <ref_motif>              Column of self.poses_df containing dictionary specifying the motif residues: {"chain": [res, ...], ...}
                <target_motif>           Column of self.poses_df containing dictionary specifying the target residues: {"chain": [res, ...], ...} Should be same number of residues as <ref_motif>
                <metric_prefix>          Prefix that should be added to the metric in the poses_df
            Return:
                List of Motif RMSDS calculated for the poses
        '''
        # define metric name:
        metric_name = f"{metric_prefix}_motif_heavy_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # parse options
        ref_pdb_l = parse_pose_options(ref_pdb, self.poses_df)
        ref_motif_l = parse_pose_options(ref_motif, self.poses_df)
        target_motif_l = parse_pose_options(target_motif, self.poses_df)

        # calculate rmsds
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            print(f"Motif Heavy RMSDs found at {scorefile} Reading RMSDs directly from file.")
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            if len(self.poses_df) == 0: raise ValueError("ERROR: Length of DataFrame = 0. DataFrame merging failed!")
            rmsds = list(self.poses_df[metric_name])
        else:
            rmsds = [motif_rmsd.superimpose_calc_motif_rmsd_heavy(ref_pdb, target_pdb, ref_selection=e_ref_motif, target_selection=e_target_motif) for ref_pdb, target_pdb, e_ref_motif, e_target_motif in zip(ref_pdb_l, self.poses_df["poses"].to_list(), ref_motif_l, target_motif_l)]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = rmsds
            # add rmsds to self.poses_df
            self.poses_df.to_json(self.scorefile)
        return rmsds

    def calc_motif_bb_rmsd_df(self, ref_pdb: str, ref_motif: str, target_motif: str, metric_prefix: str, atoms:list[str]=["CA"]) -> list:
        '''
        Calculates rmsd of current poses (of specified <atoms>) of a specified motif to pdbs found in <ref_pdb_dir>.
        Name matching is accomplished by removing index layers (<remove_layer>).
        Args:
            <ref_pdb_dir>            Path to the directory containing the reference pdbs
            <ref_motif>              Dictionary specifying the motif residues: {"chain": [res, ...], ...}
            <target_motif>           Dictionary specifying the target residues: {"chain": [res, ...], ...} Should be same number of residues as <ref_motif> 
            <metric_prefix>          Prefix that should be added to the metric in the poses_df
            <atoms>                  List of atoms for which to calculate RMSD with
            <remove_layers>          How many index layers (_0001) to remove to reach the ref_pdb names in <ref_pdb_dir> from poses
            <layer_separator>        Separator of index layers
        
        Return:
            List of Motif RMSDS calculated for the poses
        '''
        # set metric name
        metric_name = f"{metric_prefix}_motif_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # parse options
        ref_pdb_l = parse_pose_options(ref_pdb, self.poses_df)
        ref_motif_l = parse_pose_options(ref_motif, self.poses_df)
        target_motif_l = parse_pose_options(target_motif, self.poses_df)

        # calculate rmsds
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            if len(self.poses_df) == 0: raise ValueError("Length of DataFrame = 0. DataFrame merging failed!")
            rmsds = self.poses_df[metric_name]
        else:
            rmsds = [motif_rmsd.superimpose_calc_motif_rmsd(ref_pdb, target_pdb, ref_selection=e_ref_motif, target_selection=e_target_motif, atoms=atoms) for ref_pdb, target_pdb, e_ref_motif, e_target_motif in zip(ref_pdb_l, self.poses_df["poses"].to_list(), ref_motif_l, target_motif_l)]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = rmsds

        # add rmsds to self.poses_df
        self.poses_df.to_json(self.scorefile)

        return rmsds

    def calc_bb_rmsd_dir(self, ref_pdb_dir: str, metric_prefix: str, ref_chains=None, pose_chains=None, remove_layers=None, layer_separator="_") -> list:
        '''
        Calculates bb_rmsd of all poses to their matches in ref_pdb_dir after removal of <remove_layers> index layers (_0001).
        '''
        print(f"Calculating RMSDs of poses to reference poses in {ref_pdb_dir}.\nRMSD will be stored as {metric_prefix}_bb_ca_rmsd in the poses_df.")
        # set metric name:
        metric_name = f"{metric_prefix}_bb_ca_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # go from poses to ref_poses by removing layers and adding the directory ref_pdb_dir/
        if remove_layers:
            layers_removed = self.poses_df["poses_description"].str.split(layer_separator).str[:-1*remove_layers].str.join(layer_separator)
            ref_pdbs = [description + ".pdb" for description in layers_removed]
        else:
            ref_pdbs = [x+".pdb" for x in list(self.poses_df["poses_description"])]
        ref_pdbs = ["/".join([ref_pdb_dir, pdb]) for pdb in ref_pdbs]

        # if rmsds are already calculated, read them from rmsd scorefile. Otherwise, calc RMSDs.
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            rmsds = list(self.poses_df[metric_name])
        else:
            rmsds = [bb_rmsd.superimpose_calc_rmsd(ref_pdb, pose, ref_chains=ref_chains, pose_chains=pose_chains, atoms="CA") for ref_pdb, pose in zip(ref_pdbs, list(self.poses_df["poses"]))]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = rmsds

        self.poses_df.to_json(self.scorefile)

        return rmsds

    def calc_motif_heavy_rmsd_dir(self, ref_pdb_dir: str, ref_motif: dict, target_motif: dict, metric_prefix: str, remove_layers:int=1, layer_separator="_") -> list:
        '''
            Calculates rmsd of current poses (of specified <atoms>) of a specified motif to pdbs found in <ref_pdb_dir>.
            Name matching is accomplished by removing index layers (<remove_layer>).
            Args:
                <ref_pdb_dir>            Path to the directory containing the reference pdbs
                <ref_motif>              Dictionary specifying the motif residues: {"chain": [res, ...], ...}
                <target_motif>           Dictionary specifying the target residues: {"chain": [res, ...], ...} Should be same number of residues as <ref_motif> 
                <metric_prefix>          Prefix that should be added to the metric in the poses_df
                <remove_layers>          How many index layers (_0001) to remove to reach the ref_pdb names in <ref_pdb_dir> from poses
                <layer_separator>        Separator of index layers

            Return:
                List of Motif RMSDS calculated for the poses
        '''
        # define metric name:
        metric_name = f"{metric_prefix}_motif_heavy_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # check if ref_motif and target_motif are either list or dictionary and return rmsd_mode:
        if isinstance(ref_motif, dict) and isinstance(target_motif, dict):
            pose_opts = False
        elif isinstance(ref_motif, list) and isinstance(target_motif, list):
            pose_opts = True

         # go from poses to ref_poses by removing layers and adding the directory ref_pdb_dir/
        ref_poses = self.poses_df["poses_description"].str.split(layer_separator).str[:-1*remove_layers].str.join(layer_separator) if remove_layers else self.poses_df["poses_description"]

        # collect list of reference pdbfiles
        ref_pdbs = [description + ".pdb" for description in ref_poses]
        ref_pdb_list = ["/".join([ref_pdb_dir, pdb]) for pdb in ref_pdbs]

        # calculate rmsds
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            print(f"Motif Heavy RMSDs found at {scorefile} Reading RMSDs directly from file.")
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            if len(self.poses_df) == 0: raise ValueError("ERROR: Length of DataFrame = 0. DataFrame merging failed!")
            rmsds = list(self.poses_df[metric_name])
        else:
            if pose_opts:
                rmsds = [motif_rmsd.superimpose_calc_motif_rmsd_heavy(ref_pdb, target_pdb, ref_selection=e_ref_motif, target_selection=e_target_motif) for ref_pdb, target_pdb, e_ref_motif, e_target_motif in zip(ref_pdb_list, list(self.poses_df["poses"]), ref_motif, target_motif)]
            else:
                rmsds = [motif_rmsd.superimpose_calc_motif_rmsd_heavy(ref_pdb, target_pdb, ref_selection=ref_motif, target_selection=target_motif) for ref_pdb, target_pdb in zip(ref_pdb_list, self.poses_df["poses"].to_list())]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = rmsds

        # add rmsds to self.poses_df
        self.poses_df.to_json(self.scorefile)

        return rmsds

    def calc_motif_bb_rmsd_dir(self, ref_pdb_dir: str, ref_motif: dict, target_motif: dict, metric_prefix: str, atoms:list[str]=["CA"], remove_layers:int=1, layer_separator:str="_") -> list:
        '''
        Calculates rmsd of current poses (of specified <atoms>) of a specified motif to pdbs found in <ref_pdb_dir>.
        Name matching is accomplished by removing index layers (<remove_layer>).
        Args:
            <ref_pdb_dir>            Path to the directory containing the reference pdbs
            <ref_motif>              Dictionary specifying the motif residues: {"chain": [res, ...], ...}
            <target_motif>           Dictionary specifying the target residues: {"chain": [res, ...], ...} Should be same number of residues as <ref_motif> 
            <metric_prefix>          Prefix that should be added to the metric in the poses_df
            <atoms>                  List of atoms for which to calculate RMSD with
            <remove_layers>          How many index layers (_0001) to remove to reach the ref_pdb names in <ref_pdb_dir> from poses
            <layer_separator>        Separator of index layers
        
        Return:
            List of Motif RMSDS calculated for the poses
        '''
        # set metric name
        metric_name = f"{metric_prefix}_motif_rmsd"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # check if ref_motif and target_motif are either list or dictionary and return rmsd_mode:
        if type(ref_motif) == dict and type(target_motif) == dict:
            pose_opts = False
        elif type(ref_motif) == list and type(target_motif) == list:
            pose_opts = True
        else:
            raise RuntimeError(f"Parameters ref_motif and target_motif have to be of the same type (either list, or dict).\nType(ref_motif): {type(ref_motif)}\nType(target_motif): {type(target_motif)}")

        # go from poses to ref_poses by removing layers and adding the directory ref_pdb_dir/
        ref_poses = self.poses_df["poses_description"].str.split(layer_separator).str[:-1*remove_layers].str.join(layer_separator) if remove_layers else self.poses_df["poses_description"]

        # collect list of reference pdbfiles
        ref_pdbs = [description + ".pdb" for description in ref_poses]
        ref_pdb_list = ["/".join([ref_pdb_dir, pdb]) for pdb in ref_pdbs]

        # calculate rmsds
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            rmsd_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(rmsd_df, on="poses_description")
            rmsds = self.poses_df[metric_name]
        else:
            if pose_opts:
                rmsds = [motif_rmsd.superimpose_calc_motif_rmsd(ref_pdb, target_pdb, ref_selection=e_ref_motif, target_selection=e_target_motif, atoms=atoms) for ref_pdb, target_pdb, e_ref_motif, e_target_motif in zip(ref_pdb_list, self.poses_df["poses"].to_list(), ref_motif, target_motif)]
            else:
                rmsds = [motif_rmsd.superimpose_calc_motif_rmsd(ref_pdb, target_pdb, ref_selection=ref_motif, target_selection=target_motif, atoms=atoms) for ref_pdb, target_pdb in zip(ref_pdb_list, self.poses_df["poses"].to_list())]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: rmsds}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = rmsds

        # add rmsds to self.poses_df
        self.poses_df.to_json(self.scorefile)

        return rmsds

    # ------------------------- Superimposition tools -------------------------------------------------

    def add_ligand_from_ref(self, ref_col: str, ref_motif, target_motif=None, atoms:list=["CA"], lig_chain="X", prefix:str=None, overwrite:bool=True) -> None:
        '''
        Superimpose a ligand chain from a reference PDB structure onto a motif in a pose PDB structure, and add the ligand chain to the pose.
    
        The motifs are specified as dictionaries where the keys are the IDs of the chains in the PDB structures, and the values are lists of residue numbers. The function will use atoms from the specified residues in the specified chains to perform the superimposition.
    
        The function modifies the poses in-place.
        
        Behavior of Arguments ref_motif and target_motif: 
            if a string is passed, the function will look up self.poses_df[ref_motif] to collect the list of motifs from the poses_df.
            if a list is passed, the function will use the list as is, and it assumes a list of dictionaries (motifs).
            if a dictionary is passed, the function will use this dictionary for every pose in the Poses object.
            if target_motif is not passed, the motif specified in ref_motif will be used.
    
        Args:
            ref_col (str): The name of the column in the `self.poses_df` dataframe containing the reference PDB file paths.
            ref_motif (str, list, or dict): A nested dictionary specifying the chains and residues of the atoms in the reference structure to use for the superimposition.
            target_motif (str, list or dict, optional): A nested dictionary specifying the chains and residues of the atoms in the pose structure to use for the superimposition. If not provided, the `ref_motif` will be used for the pose as well.
            atoms (list, optional): A list of atom names to use for the superimposition. If not provided, all atoms in the specified residues will be used.
            lig_chain (str, optional): The ID of the chain in the reference PDB file to be added to the pose. Defaults to "X".
            prefix (str, optional): The name of the directory in which the new poses should be stored into. Defaults to 'add_chain_0001'

        Returns:
            Path to directory where the new poses are stored.
        '''
        # setup prefix and directory
        prefix = prefix or f"add_ligand_{str(self.increment_attribute('add_ligand')).zfill(4)}"

        # if adding of ligand is already done, skip:
        if os.path.isdir((lig_dir := f"{self.dir}/{prefix}")):
            if glob(f"{lig_dir}/*.pdb"): 
                self.new_poses_path(new_path=lig_dir)
                return lig_dir
            else:
                print(f"WARNING: No files found at {lig_dir}, even though the directory exist. This might hint at a previous buggy run!")

        # create directory
        os.makedirs(lig_dir, exist_ok=True)

        # parse motifs
        pose_motif = target_motif or ref_motif 

        # parse reference pdbs
        ref_l = self.poses_df[ref_col].to_list()

        # parse motif options into list
        ref_motif_l = parse_motif(self.poses_df, ref_motif)
        pose_motif_l = parse_motif(self.poses_df, pose_motif)
        if len(ref_motif_l) != len(pose_motif_l): raise ValueError(f"ERROR: ref_motif_l ({len(ref_motif_l)}) and pose_motif_l ({len(pose_motif_l)}) are not of the same length.\nref_motif_l: {ref_motif_l}\npose_motif_l: {pose_motif_l}.")

        # go through each pose in self.poses_df["poses"]
        for pose, ref, r_motif, p_motif in zip(list(self.poses_df["poses"]), ref_l, ref_motif_l, pose_motif_l):
            # create path to new pose
            new_pose = f"{lig_dir}/{pose.split('/')[-1]}"

            # run superimposition
            superimposition_tools.superimpose_add_chain_by_motif(ref, pose, lig_chain, r_motif, p_motif, new_pose_path=new_pose, superimpose_atoms=atoms, overwrite=overwrite)
        print(f"Added ligand from reference specified in {ref_col} into poses.")

        # add new location of poses to self.poses_df and self.poses
        self.new_poses_path(new_path=lig_dir)

        return lig_dir

    def add_chain_from_ref(self, ref_col, copy_chain, superimpose_chain="A", prefix=None, rename_chain:str=None) -> str:
        '''
        Superimpose a chain from a reference PDB structure onto a pose PDB structure, and add the chain to the pose.
        The function uses the alpha-carbon (CA) atoms of the two chains to perform the superimposition
        The function modifies the poses in-place.

        Args:
            ref_col (str): The name of the column in the `self.poses_df` dataframe containing the reference PDB file paths.
            copy_chain (str): The ID of the chain in the reference PDB file to be added to the pose.
            superimpose_chain (str, optional): The ID of the chain in the pose PDB file to use for the superimposition. Defaults to "A".
            prefix (str, optional): The name of the directory in which the new poses should be stored into. Defaults to 'add_chain_0001'.

        Returns:
            Path to directory where the new poses are stored.

        Examples:
            add_chain_from_ref("ref_poses", "B", superimpose_chain="C", prefix="add_chain_B")
        '''
        # setup prefix and directory
        prefix = prefix or f"add_chain_{str(self.increment_attribute('add_chain')).zfill(4)}"

        # if adding of ligand is already done, skip:
        if os.path.isdir((lig_dir := f"{self.dir}/{prefix}")):
            if glob(f"{lig_dir}/*.pdb"): 
                self.new_poses_path(new_path=lig_dir)
                return None
            else:
                print(f"WARNING: No files found at {lig_dir}, even though the directory exist. This might hint at a previous buggy run!")

        # create directory
        os.makedirs(lig_dir, exist_ok=True)

        # iterate over poses and add chain
        for pose, ref in zip(list(self.poses_df["poses"]), self.poses_df[ref_col]):
            new_pose = f"{lig_dir}/{pose.split('/')[-1]}"
            if superimpose_chain: superimposition_tools.superimpose_add_chain(ref, pose, copy_chain, superimpose_chain=superimpose_chain, new_pose_path=new_pose)
            else: superimposition_tools.add_chain(pose, ref, copy_chain, new_pose_path=new_pose, rename_chain=rename_chain)
        print(f"Added chain {copy_chain} from reference specified in {ref_col} into poses.")

        # add new location of poses to self.poses_df and self.poses
        self.new_poses_path(new_path=lig_dir)

        return lig_dir

    def remove_chain_from_poses(self, remove_chain:str, prefix:str=None) -> str:
        '''
        Superimpose a chain from a reference PDB structure onto a pose PDB structure, and add the chain to the pose.
        The function uses the alpha-carbon (CA) atoms of the two chains to perform the superimposition
        The function modifies the poses in-place.

        Args:
            ref_col (str): The name of the column in the `self.poses_df` dataframe containing the reference PDB file paths.
            copy_chain (str): The ID of the chain in the reference PDB file to be added to the pose.
            superimpose_chain (str, optional): The ID of the chain in the pose PDB file to use for the superimposition. Defaults to "A".
            prefix (str, optional): The name of the directory in which the new poses should be stored into. Defaults to 'add_chain_0001'.

        Returns:
            Path to directory where poses are stored.

        Examples:
            add_chain_from_ref("ref_poses", "B", superimpose_chain="C", prefix="add_chain_B")
        '''
        # setup prefix and directory
        prefix = prefix or f"remove_chain_{str(self.increment_attribute('remove_chain')).zfill(4)}"

        # if removing of chain is already done, skip:
        if os.path.isdir((pdb_dir := f"{self.dir}/{prefix}")):
            if glob(f"{pdb_dir}/*.pdb"): 
                self.new_poses_path(new_path=pdb_dir)
                self.poses_df[f"{prefix}_location"] = self.poses_df["poses"]
                print(f"Moved poses to {prefix}_location")
                return None
            else:
                print(f"WARNING: No files found at {pdb_dir}, even though the directory exist. This might hint at a previous buggy run!")

        # create directory
        os.makedirs(pdb_dir, exist_ok=True)

        # iterate over poses and remove chain
        for pose in list(self.poses_df["poses"]):
            new_pose = f"{pdb_dir}/{pose.split('/')[-1]}"
            superimposition_tools.remove_chain(pose, remove_chain, new_pose_path=new_pose)
        print(f"Removed chain {remove_chain} from poses.")

        # add new location of poses to self.poses_df and self.poses
        self.new_poses_path(new_path=pdb_dir)
        self.poses_df[f"{prefix}_location"] = self.poses_df["poses"]
        print(f"Moved poses to {prefix}_location")

        return pdb_dir


    # ------------------------- Poses Misc Functions --------------------------------------------------

    def calc_composite_score(self, name: str, scoreterms, weights):
        '''
        Calculates a composite score named <name> from <scoreterms> weighted by <weights>.
        Adds new scoreterm as attribute to self.poses_df.
        <name> has to be a unique name that does not exists yet in poses_df!
        If <name> already exists in DataFrame, then calc_composite_score skips the calculation!
        
        Args:
            <name>: Name of the composite_score that you want to calculate and add to your DataFrame.
            <scoreterms>: Scoreterms from which to calculate the composite_score.
            <weights>: Weights to give to <scoreterms>. Weights has to be same length as <scoreterms>!
        
        Returns composite score
        '''
        #TODO: implement scoreterm lookup!

        # check if <name> exists in poses_df.columns
        if name in self.poses_df.columns: return print(f"WARNING: Scoreterm {name} already exists in poses DataFrame. Skipping Calculation.")

        # calculate composite score
        configs = "\n".join([f"{score}: {weight}" for score, weight in zip(scoreterms, weights)])
        print(f"\nCalculating composite score {name} with settings:\n{configs}\n")
        new_df = calc_composite_score.calc_composite_score(self.poses_df, name, scoreterms, weights)

        # update self.poses_df
        self.poses_df = new_df
        self.poses_df.to_json(self.scorefile)

        return None

    def update_poses(self) -> None:
        '''
        Filters the self.poses object down to the poses present in self.poses_df["poses_description"].
        '''
        len_before = len(self.poses_df["poses"])
        self.poses = self.poses_df["poses"].to_list()
        print(f"Reducing number of poses from {str(len_before)} to {str(len(self.poses))}.\n")
        return None

    def dump_poses(self, out_path: str) -> None:
        '''
        Stores poses at <path>, returns <path>.
        '''
        # check if directory exists:
        if not os.path.isdir(out_path): os.makedirs(out_path, exist_ok=True)

        # copy poses into out_path
        print(f"Copying {len(self.poses)} poses into {out_path}")
        for pose in self.poses_df["poses"].to_list():
            shutil.copy(pose, out_path + "/")

        return out_path

    def set_poses(self, poses: list[str], scores=None, prefix=None, description_col="description", remove_layers=0, sep="_") -> list[str]:
        '''
        Sets a list of <poses> as the new poses of the object. Updates the poses in the DataFrame and fills up previous scores.
        Also adds scores if needed.
        
        Args:
            <poses>               List with paths to where the poses are stored.
            <scores>              Optional: pd.DataFrame containing the scores 
            <prefix>              Optional: Prefix that will be added to every score of the added pd.DataFrame (<scores>). Default: 'add_0001_'
            <description_col>     Column that contains the 'index' by which to merge scores with poses_df
            <remove_layers>       Number of index layers to remove to reach the names of 'poses_description' in self.poses_df
        
        Returns:
            List of Poses.
        '''
        # if scores is not given, create pd.DataFrame with 'new_description' column
        scores = scores or pd.DataFrame({description_col: [pose.split(sep)[-1].split(".")[0] for pose in poses]})
        scores.loc[:, "location"] = poses
        prefix = prefix or f"add_{str(self.set_poses_number).zfill(4)}"

        # update poses_df based on scores
        scores = scores.add_prefix(prefix + "_")
        self.poses_df = update_df(scores, f"{prefix}_{description_col}", self.poses_df, "poses_description", new_df_col_remove_layer=remove_layers)

        # update essential columns in poses_df
        self.poses_df.loc[:, "poses"] = self.poses_df[f"{prefix}_location"]
        self.poses_df.loc[:, "poses_description"] = self.poses_df[f"{prefix}_description"]

        # if argument self.auto_dump_df is set to True, dump the new poses_df in self.dir
        if self.auto_dump_df: self.poses_df.to_json(self.scorefile)

        # set self.poses and return poses
        self.poses = list(self.poses_df[f"{prefix}_location"])

        return self.poses

    def new_poses_path(self, new_path: str) -> None:
        '''
        !!!!!!! WIP !!!!!!!!
        
        Sets a new location for the current poses: resets self.poses and self.poses_df["poses"]]. Everything else remains unchanged.
        This function is intended to be used after small changes were made to the poses that did not require renaming.
        
        '''
        old_path_l = list(self.poses_df["poses"].str.split("/").str[:-1].str.join("/").unique())

        # sanity checks
        if len(old_path_l) > 1: raise ValueError(f"ERROR: Your poses are stored at multiple, different locations. Method new_poses_path() is only to be used with one unique location for all poses.")
        if not os.path.isdir(os.path.abspath(new_path)): raise FileNotFoundError(f"ERROR: Directory {new_path} does not exist! Are you sure you specified the correct path?")

        # update poses
        self.poses_df.loc[:, "poses"] = [x.replace(old_path_l[0], new_path) for x in list(self.poses_df["poses"])]
        self.poses = list(self.poses_df["poses"])

        return print(f"relocated poses to {new_path}")

    def reindex_poses(self, out_dir:str=None, remove_layers:int=None, force_reindex:bool=True, keep_layers=False) -> str:
        '''
        Removes <remove_layers> from poses and reindexes them.
        '''
        remove_layers = remove_layers or self.index_layers

        # set up out_dir:
        out_dir = out_dir or f"reindex_{str(self.reindex_number).zfill(4)}"
        out_dir = self.dir + "/" + out_dir
        if not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

        # reindex and copy poses into new location:
        renaming_dict = remove_index_layers.remove_index_layers(self.poses_df["poses"].to_list(), output_dir=out_dir, n_layers=remove_layers, force_reindex=force_reindex, keep_layers=keep_layers)
        renaming_dict = {k.split("/")[-1].split(".")[0]: v for k, v in renaming_dict.items()}

        # add 'reindexed_description' column to self.poses_df according to the renaming_dict:
        old_descriptions = deepcopy(list(self.poses_df["poses_description"]))
        self.poses_df.loc[:, "poses_description"] = [renaming_dict[location].split("/")[-1].split(".")[0] for location in old_descriptions]
        self.poses_df.loc[:, "poses"] = [renaming_dict[location] for location in old_descriptions]

        # set reindexed poses as new poses:
        self.poses = list(self.poses_df["poses"])

        # if argument self.auto_dump_df is set to True, dump the new poses_df in self.dir
        if self.auto_dump_df:
            self.poses_df.to_json(self.scorefile)

        return out_dir

    def calc_protparams(self, prefix=None) -> None:
        '''
        Calculates protparam scores for poses and stores them in poses_df.
        '''
        # set the prefix
        prefix = prefix or f"params_{str(self.increment_attribute('protparams')).zfill(4)}"

        # Start the parser
        pdb_parser = Bio.PDB.PDBParser(QUIET = True)
        ppb = Bio.PDB.PPBuilder()

        # if poses are pdb-files:
        if all([x.endswith(".pdb") for x in self.poses_df["poses"].to_list()]):  
            # Get the structures
            poses_l = [pdb_parser.get_structure(pose, pose) for pose in self.poses_df["poses"].to_list()]

            # collect the sequences
            sequences = ["".join([str(x.get_sequence()) for x in ppb.build_peptides(pose)]) for pose in poses_l]

        elif all([x.endswith(".fa") for x in self.poses]):
            sequences = [read_fasta_sequence(pose) for pose in self.poses]

        else:
            raise TypeError(f"ERROR: one or more of your poses are of invalid type (Only .fa and .pdb are allowed.)")

        # collect metrics for sequences
        analyses = [ProteinAnalysis(sequence) for sequence in sequences]
        metrics_dict = {"pI": [x.isoelectric_point() for x in analyses],
                        "extinction_coefficient": [x.molar_extinction_coefficient() for x in analyses],
                        "instability_index": [x.instability_index() for x in analyses],
                        "aromaticity": [x.aromaticity() for x in analyses],
                        "molecular_weight": [x.molecular_weight() for x in analyses],
                       }

        # add metrics to self.poses_df
        for metric in metrics_dict:
            self.poses_df.loc[:, f"{prefix}_{metric}"] = metrics_dict[metric]

        # if argument self.auto_dump_df is set to True, dump the new poses_df in self.dir
        if self.auto_dump_df:
            self.poses_df.to_json(self.scorefile)

        return None

    def store_sequences(self, prefix:str=None) -> None:
        '''
        Method to store sequences of the current poses in self.poses_df
        '''
        # set the column name for the poses_df:
        col = prefix or f"sequence_{str(self.increment_attribute('seqs')).zfill(4)}"

        # Start the parser
        pdb_parser = Bio.PDB.PDBParser(QUIET = True)
        ppb = Bio.PDB.PPBuilder()

        # gather sequences from poses into a list
        if all([x.endswith(".pdb") for x in list(self.poses_df["poses"])]):  
            # if poses are pdb-files, get the structures and collect the sequences:
            poses_l = [pdb_parser.get_structure(pose, pose) for pose in list(self.poses_df["poses"])]
            sequences = ["".join([str(x.get_sequence()) for x in ppb.build_peptides(pose)]) for pose in poses_l]

        elif all([x.endswith(".fa") for x in list(self.poses_df["poses"])]):
            # if poses are .fa-files, just read fasta-files:
            sequences = [read_fasta_sequence(pose) for pose in list(self.poses_df["poses"])]

        else:
            raise TypeError(f"ERROR: one or more of your poses are of invalid type (Only .fa and .pdb are allowed.)")

        # set new scoreterm in poses_df:
        self.poses_df.loc[:, col] = sequences
        self.poses_df.to_json(self.scorefile)

        return None

    def biopython_mutate(self, col: str, mutation_dict:dict=None):
        '''
        requires dict with mutations listed like this:
        {"A10": "ALA", "B12": "CYS", "C3": "PHE"}
        Dict can be read out from poses_df (set the col parameter)
        '''
        # argument processing
        if col: mutations_dictlist = self.poses_df[col].to_list()
        elif mutation_dict: mutations_dictlist = [mutation_dict for pose in self.poses_df["poses"].to_list()]

        # check if poses are pdb-files:
        if any([not x.endswith(".pdb") for x in self.poses_df["poses"].to_list()]): raise TypeError(f"Poses must be .pdb files to run biopython_mutate()\nPoses:{', '.join(self.poses_df['poses'].to_list())}")

        for pdb, mutation_dict in zip(self.poses_df["poses"].to_list(), mutations_dictlist):
            pose_path = utils.biopython_mutate.mutate_pdb(pdb, mutation_dict=mutation_dict, out_pdb_path=pdb, biopython_model_number=0)
        return None

    def add_site_score(self, cat_res_sc_rmsd, pose_motif, perresidue_plddt,  metric_prefix: str) -> list:
        '''
            Calculates rmsd of current poses to reference pdbs which path can be found in 'frag_col'.
            Args:
                <cat_res_sc_rmsd>        List of residue (sidechain) rmsds for each pose (should be the same residues as specified in pose_motif_list) OR column name in poses_df that contains rmsds
                <pose_motif>             List containing one dictionary specifying the motif residues on the pose: {"chain": [res, ...], ...} per pose OR column name in poses_df that contains motif residues
                <metric_prefix>          Prefix that should be added to the metric in the poses_df
                <perresidue_plddt>       Nested list of list of perresidue plddt scores for each pose OR column name in poses_df that contains perresidue_plddt lists
                #TODO: convert perresidue plddt to {'chain': [list of perresidue plddts]}, otherwise this might lead to confusing results if there is more than one chain
            Return:
                List of Motif RMSDS calculated for the poses
        '''
        # define metric name:
        metric_name = f"{metric_prefix}_site_score"
        if metric_name in self.poses_df.columns:
            print(f"WARNING: Metric {metric_name} already found in poses_df! Skipping calculation! (Check if you set the same prefix twice!!)")
            return list(self.poses_df[metric_name])

        # parse options
        cat_res_sc_rmsd_list = parse_pose_options(cat_res_sc_rmsd, self.poses_df)
        pose_motif_list = parse_pose_options(pose_motif, self.poses_df)
        perresidue_plddt_list = parse_pose_options(perresidue_plddt, self.poses_df)

        # calculate sitescore
        if os.path.isfile((scorefile := f"{self.scores_dir}/{metric_name}_scores.json")):
            print(f"Site score found at {scorefile} Reading scores directly from file.")
            site_score_df = pd.read_json(scorefile)
            self.poses_df = self.poses_df.merge(site_score_df, on="poses_description")
            if len(self.poses_df) == 0: raise ValueError("ERROR: Length of DataFrame = 0. DataFrame merging failed!")
            site_score = list(self.poses_df[metric_name])
        else:
            # select residues of the first chain defined in the motif:
            cat_res_pos = [list(resdict.values())[0] for resdict in pose_motif_list]

            # calculate average plddt of residues defined in motif:
            cat_res_av_plddt = []
            for resposlist, plddtlist in zip(cat_res_pos, perresidue_plddt_list):
                plddts = [plddtlist[resnum - 1] for resnum in resposlist]
                cat_res_av_plddt.append(sum(plddts) / len(plddts))
            site_score = [utils.metrics.calc_site_score(sc_rmsd, av_residue_plddt) for sc_rmsd, av_residue_plddt in zip(cat_res_sc_rmsd_list, cat_res_av_plddt)]
            pd.DataFrame({"poses_description": list(self.poses_df["poses_description"]), metric_name: site_score}).to_json(scorefile)
            self.poses_df.loc[:, metric_name] = site_score
        # add scores to self.poses_df
        self.poses_df.to_json(self.scorefile)
        return site_score

    def add_LINK_to_poses(self, covalent_bonds_column: str, prefix: str, overwrite_lig_num:int=None) -> None:
        '''Adrian's function. Adds link record to .pdb files for Rosetta.
        WARNING: overwrites .pdb files. If LINK is already present, but donefile was deleted, the LINKs will be overwritten!'''

        # check if poses are pdb files, otherwise raise error
        if not all([x.endswith(".pdb") for x in self.poses_df["poses"]]): raise ValueError("One or multiple poses are not in .pdb format! add_LINK_to_poses requires .pdb formatted poses!") 

        # setup prefix and check if link is present
        link_name = f"{prefix}_LINKS"
        if os.path.isfile((donefile := f"{self.dir}/{prefix}_links_done.txt")):
            print(f"WARNING: LINK {link_name} already found in poses_df! Skipping!")
            return None

        # parse covalent bonds
        covalent_bonds_superlist = [row[covalent_bonds_column].split(",") for _, row in self.poses_df.iterrows()]
        link_list = []
        for pose, covalent_bonds in zip(self.poses_df['poses'].to_list(), covalent_bonds_superlist):
            if any([":" in bond for bond in covalent_bonds]): 
                # parse LINK record from covalent bond
                links = [parse_link_from_covalent_bond(covalent_bond, overwrite_lig_num=overwrite_lig_num) for covalent_bond in covalent_bonds]

                # write LINK record bonds into .pdb files
                _ = prefix_string_to_file(pose, "\n".join(links) + "\n")
                link_list.append(links)
            else:
                link_list.append(None)

        # write donefile:
        with open(donefile, 'w') as f: f.write("done")

        # save changes
        self.poses_df.loc[:, link_name] = link_list
        self.poses_df.to_json(self.scorefile)
        return None

    def calc_esm2_pseudo_perplexity(self, options:str="", prefix:str=None, gpu=False, max_cores:int=1000) -> None:
        '''Calculates ESM pseudo-perplexity (in-house implementation of pseudo-perplexity described in ESMFold paper SI) for entire sequence.'''
        def parse_options(options_str: str, prefix: str) -> dict:
            ''''''
            opts_d = parse_options_string(options_str, sep="--")
            opts_d["output_dir"] = self.dir + "/" + prefix
            return opts_d

        # setup prefix
        prefix = prefix or f"esm_ppl_{str(self.increment_attribute('esm_ppl')).zfill(4)}"
        max_cores = max_cores or 10 if gpu else 312

        # parse options
        options = parse_options(options, prefix)

        # make sure poses are .fa files
        if not all([pose.endswith(".fa") for pose in self.poses_df["poses"].to_list()]): raise RuntimeError(f"All Poses must be .fa files!")

        # run perplexity calculation:
        scores = calc_esm_perplexity(self.poses_df["poses"].to_list(), options=options, prefix=prefix, gpu=gpu, max_cores=max_cores)

        # merge with DF
        scores = scores.add_prefix(prefix + "_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=0)
        self.poses_df.to_json(self.scorefile)

        # return
        return None

    def esm2_sequence_entropy(self, options:str="", prefix:str=None, max_cores:int=1000, gpu=False, keep_probs=False) -> None:
        '''Calculates ESM sequence etnropy (in-house implementation of entropy calculation with ESMFold). Not optimized yet.'''
        def parse_options(options_str: str, prefix: str) -> dict:
            ''''''
            opts_d = parse_options_string(options_str, sep="--")
            opts_d["output_dir"] = self.dir + "/" + prefix + "/"
            return opts_d

        # setup prefix
        prefix = prefix or f"esm_entropy_{str(self.increment_attribute('esm_ppl')).zfill(4)}"
        max_cores = max_cores or 10 if gpu else 312

        # parse options
        options = parse_options(options, prefix)

        # make sure poses are .fa files
        if all([pose.endswith(".pdb") for pose in self.poses_df["poses"].to_list()]): self.poses_pdb_to_fasta(chain_sep=":")
        elif not all([pose.endswith(".fa") for pose in self.poses_df["poses"].to_list()]): raise RuntimeError(f"All Poses must be .fa files!")

        # run perplexity calculation:
        scores = esm_entropy(self.poses_df["poses"].to_list(), options=options, prefix=prefix, gpu=False, max_cores=max_cores, keep_probs=keep_probs)

        # merge with DF
        scores = scores.add_prefix(prefix + "_")
        self.poses_df = update_df(scores, f"{prefix}_description", self.poses_df, "poses_description", new_df_col_remove_layer=0)
        self.poses_df.to_json(self.scorefile)

        # return
        return None

    def create_motif_from_residue_scores(self, prefix:str, score_col: str, abs_cutoff:float=None, percentile_cutoff:float=25, keep_below=True) -> None:
        '''
        !!! Only works for single-chain poses !!!
        !!! score_col must be a column in poses_df that contains perresidue-lists of a score. (e.g. entropies, energies, or pLDDTs)
        !!! abs_cutoff overwrites percentile_cutoff !!!
        Creates a motif that selects the positions that are below (default) or above a specified cutoff in a column carrying residue-scores.
        '''
        # check for col
        if col_with_prefix_exists_in_df(df=self.poses_df, prefix=prefix): raise KeyError(f"Column with name {prefix} already exists in poses_df. pick different prefix.")

        # define cutoff
        scores = self.poses_df[score_col].to_list()
        if abs_cutoff: cutoff_l = [abs_cutoff for x in scores]
        else: cutoff_l = [np.percentile(value_l, percentile_cutoff) for value_l in scores]

        # add a column to poses_df that contains a motif with residue indeces for residues where 'value' is above or below 'cutoff'
        if keep_below: self.poses_df[prefix] = [{"A": (np.where(np.array(value_l) < cutoff)[0] + 1).tolist()} for cutoff, value_l in zip(cutoff_l, scores)]
        else: self.poses_df[prefix] = [{"A": (np.where(np.array(value_l) > cutoff)[0] + 1).tolist()} for cutoff, value_l in zip(cutoff_l, scores)]

        return None

    def create_pymol_perresidue_coloring(self, prefix:str, transitions:list[float], color1:list[float], color2:list[float], color3:list[float]=None) -> None:
        ''' '''
        raise NotImplementedError()

    def create_pymol_motif_coloring(self, prefix:str, motif_col:str, poses_col:str="poses", color_motif:list[float]=[1,0.8,0], color_bg:list[float]=[0.5,0.5,0.5]) -> None:
        '''Creates pymol coloring schemes for .pdb files of poses in 'poses_col'. 
        'poses_col' needs to be a column in poses_df that contains locations of .pdb files
        '''

        return None

    def all_against_all_sequence_similarity(self, sequence_column: str, prefix:str) -> None:
        '''Calculates maximum all-against-all sequence similarity in numpy for all sequences in the sequence column. All sequences MUST be of the same length!'''
        if prefix in self.poses_df.columns: raise KeyError(f"Column {prefix} already used in poses_df please choose different prefix!")
        self.poses_df[prefix] = utils.metrics.all_against_all_sequence_similarity(self.poses_df[sequence_column].to_list())

# ------------------ Protein Generator ---------------------------------------------
def run_protein_generator(poses: list[str], work_dir:str, options:str=None, pose_options=None, scorefile:str=None, max_cores:int=10) -> str:
    '''Runs protein_generator for you on acluster'''
    # setup workdir and scorefile
    work_dir = os.path.abspath(work_dir)
    if not os.path.isdir((pdb_dir := f"{work_dir}/output_pdbs/")): os.makedirs(pdb_dir, exist_ok=True)

    # Look for output-file in pdb-dir. If output is present and correct, then skip protein_generator.
    pg_out_dict = {"pdb_dir": pdb_dir}
    if os.path.isfile((scorefilepath := f"{work_dir}/{scorefile}")):
        return {"pdb_dir": pdb_dir, "scores": pd.read_json(scorefilepath)}

    # parse_options and pose_options:
    if pose_options:
        # safety check (pose_options must have the same length as poses)
        if len(poses) != len(pose_options): raise ValueError(f"Arguments <poses> and <pose_options> for RFDiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
    else:
        # make sure empty lists are passed for pose_options
        pose_options = ["" for x in poses]

    # write protein generator cmds:
    cmds = [write_pg_cmd(pose, output_dir=pg_out_dict["pdb_dir"], options=options, pose_options=pose_options) for pose, pose_opts in zip(poses, pose_options)]

    # setup sbatch run
    sbatch_options = [f"--gpus-per-node 1 -c1 -e {work_dir}/protein_generator_err.log -o {work_dir}/protein_generator_out.log"]
    sbatch_array_jobstarter(
        cmds=cmds,
        sbatch_options=sbatch_options,
        jobname="protein_generator",
        max_array_size=max_cores,
        wait=True,
        remove_cmdfile=False,
        cmdfile_dir=work_dir
    )

    # collect scores and return outputs
    pg_out_dict['scores'] = collect_protein_generator_scores(scores_dir=pdb_dir, scorefile=scorefilepath)
    pg_out_dict['scores'].to_json(scorefilepath)

    return pg_out_dict

def write_pg_cmd(pose_path:str, options:str, pose_options:str, output_dir:str, path_to_executable:str=_protein_generator_path) -> str:
    '''Writes command to run protein_generator'''
    # parse description:
    desc = pose_path.split("/")[-1].split(".")[0]

    # parse_options:
    opts, flags = parse_pg_opts(options, pose_options)
    opts = ' '.join([f"--{key} {value}" for key, value in opts.items()])
    flags = " --".join(flags)

    return f"{path_to_executable} --out {output_dir}/{desc} {opts} {flags}"

def parse_pg_opts(options:str, pose_opts:str) -> str:
    '''Parses options to run protein_generator'''
    def tmp_parse_options_flags(options_str: str) -> tuple[str, str]:
        '''parses split options '''
        if not options_str: return {}, []
        # split along separator
        firstsplit = [x.strip() for x in options.split("--") if x]

        # parse into options and flags:
        opts = dict()
        flags = list()
        for item in firstsplit:
            if len((x := item.split())) > 1:
                opts[x[0]] = " ".join(x[1:])
            else:
                flags.append(x[0])

        return opts, flags

    # parse into options and flags:
    opts, flags = tmp_parse_options_flags(options)
    pose_opts, pose_flags = tmp_parse_options_flags(pose_opts)

    # merge options and pose_options (pose_opts overwrite opts), same for flags
    opts.update(pose_opts)
    flags = list(set(flags) | set(pose_flags))

    return opts, flags

def collect_protein_generator_scores(scores_dir: str, scorefile: str) -> pd.DataFrame:
    '''collects scores from protein_generator output'''
    # read .pdb files
    pl = glob(f"{scores_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"No .pdb files were found in the output directory of protein_generator {scores_dir}. protein_generator might have crashed (check output log), or path might be wrong!")

    # parse .trb-files into DataFrames
    df = pd.concat([parse_protein_generator_trbfile(p.replace(".pdb", ".trb")) for p in pl], axis=0).reset_index(drop=True)

    # write scorefile
    df.to_json(scorefile)

    return df

def parse_protein_generator_trbfile(trbfile: str) -> pd.DataFrame:
    '''Reads protein_generator output .trb file and parses the scores into a pandas DataFrame.'''
    trb = np.load(trbfile, allow_pickle=True)

    # expand collected data if needed:
    data_dict = {
        "description": trbfile.split("/")[-1].replace(".trb", ""),
        "location": trbfile.replace("trb", "pdb"),
        "lddt": [sum(trb["lddt"]) / len(trb["lddt"])],
        "perres_lddt": [trb["lddt"]],
        "sequence": trb["args"]["sequence"],
        "contigs": trb["args"]["contigs"],
        #"ref_idx": trb["args"]["ref_idx"],
        #"hal_idx": trb["args"]["hal_idx"],
        "inpaint_str": [trb["inpaint_str"].numpy().tolist()],
        "inpaint_seq": [trb["inpaint_seq"].numpy().tolist()]
    }
    return pd.DataFrame(data_dict)

# ------------------ ESM-perplexity ------------------------------------------------
def write_ppl_cmd(input_fasta:str, output_dir:str, options:dict, script_path=_ppl_script_path) -> str:
    ''' '''
    options["input_fasta"] = input_fasta
    options["output_json"] = f"{output_dir}/{input_fasta.split('/')[-1].replace('.fa', '')}_scores.json"
    opts_str = " --" + " --".join([f"{key}={value}" for key, value in options.items()])
    return f"{script_path} {opts_str}"

def write_entropy_cmd(input_fasta:str, output_dir:str, options:dict, script_path=_entropy_script_path) -> str:
    ''' '''
    options["input_fasta"] = input_fasta
    options["output_dir"] = output_dir
    opts_str = " --" + " --".join([f"{key}={value}" for key, value in options.items()])
    return f"{script_path} {opts_str}"

def esm_entropy(fasta_files: list[str], options:dict, prefix:str, gpu:bool=False, max_cores:int=None, keep_probs:bool=False) -> pd.DataFrame:
    '''wraps around calculation of entropy using esm_calc_entropy for list of .fa files'''
    # setup directories
    os.makedirs((e_work_dir := options.pop("output_dir")), exist_ok=True)
    os.makedirs((e_out_dir := f"{e_work_dir}/esm_output/"), exist_ok=True)
    os.makedirs((e_input_dir := f"{e_work_dir}/esm_input/"), exist_ok=True)

    # check if outputs already present
    if os.path.isfile((scorefile := f"{e_work_dir}/{prefix}_scores.json")):
        print(f"\nOutputs of entropy-calculation run {e_work_dir} already found at {scorefile}\nSkipping prediction step.")
        return pd.read_json(scorefile)

    # split poses into <max_cores> sublists and write them into fasta files to be used as input to calc_perplexity.py
    splitnum = len(fasta_files) if len(fasta_files) <= max_cores else max_cores
    poses_split = [list(x) for x in np.array_split(fasta_files, int(splitnum))]
    pose_fastas = [mergefastas(poses, f"{e_input_dir}/entropy_fasta_{str(i+1).zfill(4)}.fa", replace_colon=True) for i, poses in enumerate(poses_split)]

    # write commands for fastafiles and run
    esm_cmds = [write_entropy_cmd(fasta, output_dir=e_out_dir, options=options) for fasta in pose_fastas]
    sbatch_opts = [f"-e {e_work_dir}/esm_entropy_err.log -o {e_work_dir}/esm_entropy_out.log"]
    if gpu: sbatch_opts.append("--gpus-per-node 1")
    sbatch_array_jobstarter(esm_cmds, sbatch_opts, jobname="entropy_esm", max_array_size=max_cores, wait=True, remove_cmdfile=False, cmdfile_dir=e_work_dir)

    # collect scores into one df
    scoreterms = ["description", "perresidue_entropy", "sequence_entropy"]
    if keep_probs: scoreterms.append("perresidue_probabilities")
    df = pd.concat([pd.read_json(f"{e_out_dir}/{fasta.split('/')[-1].replace('.fa', '')}_scores.json")[scoreterms] for fasta in pose_fastas], ignore_index=True)
    df.to_json(scorefile)

    # return output
    return df

def calc_esm_perplexity(fasta_files: list[str], options:dict, prefix:str, gpu:bool=False, max_cores:int=None) -> pd.DataFrame:
    '''wraps around calculation of perplexity for list of .fa files'''
    # setup directories
    os.makedirs((ppl_work_dir := options.pop("output_dir")), exist_ok=True)
    os.makedirs((ppl_out_dir := f"{ppl_work_dir}/esm_output"), exist_ok=True)
    os.makedirs((ppl_input_dir := f"{ppl_work_dir}/esm_input/"), exist_ok=True)

    # check if outputs already present
    if os.path.isfile((scorefile := f"{ppl_work_dir}/{prefix}_scores.json")):
        print(f"\nOutputs of ppl-calculation run {ppl_work_dir} already found at {scorefile}\nSkipping prediction step.")
        return pd.read_json(scorefile)

    # split poses into <max_cores> sublists and write them into fasta files to be used as input to calc_perplexity.py
    splitnum = len(fasta_files) if len(fasta_files) <= max_cores else max_cores
    poses_split = [list(x) for x in np.array_split(fasta_files, int(splitnum))]
    pose_fastas = [mergefastas(poses, f"{ppl_input_dir}/ppl_fasta_{str(i+1).zfill(4)}.fa", replace_colon=True) for i, poses in enumerate(poses_split)]

    # write commands for fastafiles and run
    esm_cmds = [write_ppl_cmd(fasta, output_dir=ppl_out_dir, options=options) for fasta in pose_fastas]
    sbatch_opts = [f"-e {ppl_work_dir}/esm_perplexity_err.log -o {ppl_work_dir}/esm_perplexity_out.log"]
    if gpu: sbatch_opts.append("--gpus-per-node 1")
    sbatch_array_jobstarter(esm_cmds, sbatch_opts, jobname="pseudo_perplexity_esm", max_array_size=max_cores, wait=True, remove_cmdfile=False, cmdfile_dir=ppl_work_dir)

    # collect scores into one df
    df = pd.concat([pd.read_json(f"{ppl_out_dir}/{fasta.split('/')[-1].replace('.fa', '')}_scores.json") for fasta in pose_fastas], ignore_index=True)
    df.to_json(scorefile)

    # return output
    return df

# ------------------ Rosetta-specific ----------------------------------------------
def prefix_string_to_file(file_path: str, prefix: str, save_path:str=None) -> str:
    '''Adds something to the beginning of a file.'''
    with open(file_path, 'r') as f: file_str = f.read()
    with open(save_path or file_path, 'w') as f: f.write(prefix + file_str)
    return save_path

def parse_link_from_covalent_bond(covalent_bond: str, overwrite_lig_num:int=None) -> str:
    '''parses covalent bond into Rosetta formated link record for PDB headers.'''
    res_data, lig_data = covalent_bond.split(':')
    res_data = res_data.split('_')
    lig_data = lig_data.split('_')
    res_atom = res_data[2]
    res_id = res_data[1]
    res_chain = res_data[0][-1]
    res_num = res_data[0][:-1]
    lig_atom = lig_data[2]
    lig_id = lig_data[1]
    lig_chain = lig_data[0][-1]
    lig_num = overwrite_lig_num or lig_data[0][:-1]
    return f"LINK         {res_atom:<3} {res_id:<3} {res_chain:>1}{res_num:>4}                {lig_atom:<3}  {lig_id:>3} {lig_chain:>1}{lig_num:>4}                  0.00"

# ------------------ Operational functions -----------------------------------------

def prep_options(dict_a: dict, dict_b: dict) -> dict:
    '''
    dict_b is overwritten by dict_a
    '''
    dict_b.update(dict_a)
    return dict_b

## ------------------ Inpainting ---------------------------------------------------
def run_inpainting(poses: list, work_dir: str, options="", pose_options=None, scorefile="inpainting_scores.json", max_gpus=5, perres_lddt=False, perres_inpaint_lddt=False) -> dict:
    '''
    Important: options specified in <pose_options> overwrite global options specified in <options>!
    '''
    # setup workdir:
    work_dir = os.path.abspath(work_dir)
    pdb_dir = f"{work_dir}/output_pdbs/"
    if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

    # setup pdb_dir and look for output file. Skip inpainting, if output is present:
    return_dict = {"pdb_dir": (pdb_dir := f"{work_dir}/output_pdbs/")}
    if os.path.isfile((inpaint_scorefile := f"{work_dir}/{scorefile}")):
        return {"pdb_dir": pdb_dir, "scores": pd.read_json(inpaint_scorefile)}

    # parse options, flags, pose_options and pose_flags
    options, flags = parse_options_flags(options, sep='--')
    if pose_options:
        #sanity check:
        if len(poses) != len(pose_options): raise ValueError(f"ERROR: Arguments <poses> and <pose_option> of the function run_inpainting() must be of the same length!\nlen(poses) = {len(poses)}\nlen(pose_options) = {len(pose_options)}")
        pose_options, pose_flags = zip(*[parse_options_flags(opt_str, sep="--") for opt_str in pose_options])
    else: 
        pose_options, pose_flags = [{} for x in poses], [[] for x in poses]

    # write inpainting commands for each pose in poses:
    cmds = [write_inpainting_cmd(pose, options=dict(options, **pose_opts), flags=list(set(flags) | set(pose_flgs)), out_dir=pdb_dir) for pose, pose_opts, pose_flgs in zip(poses, pose_options, pose_flags)]

    # execute inpainting commands in a jobarray on maximum <max_gpus>
    sbatch_options = [f"--gpus-per-node 1 -c1 -e {work_dir}/inpainting_err.log -o {work_dir}/inpainting_out.log"]
    sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="inpaint_poses", 
                                  max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=work_dir)

    # collect inpainting scores and return outputs.
    return_dict["scores"] = collect_inpainting_scores(inpaint_dir=pdb_dir, scorefile=inpaint_scorefile, rename_pdbs=True, perres_lddt=perres_lddt, perres_inpaint_lddt=perres_inpaint_lddt)
    return_dict["scores"].to_json(inpaint_scorefile)

    return return_dict

def collect_inpainting_scores(inpaint_dir: str, scorefile="inpainting_scores.json", rename_pdbs=True, perres_lddt=False, perres_inpaint_lddt=False) -> pd.DataFrame:
    '''
    Collects scores from .trb and .npz files of inpainting and renames (if <rename_pdbs> is set) output .pdbs from output_1.pdb to output_0001.pdb
    Args:
        <inpaint_dir>                   Output directory of your inpainting run.
        <scorefile>                     Name of the scorefile you want to write
        <rename_pdbs>                   (True)  bool: If set, renames .pdb files in dataframe and inpaint_dir to standard indeces (_0001 instead of _1, etc.)
        <perres_lddt>                   (False) bool: If set, perresidue lddts will be collected as lists into the output DataFrame
        <perres_inapint_lddt>           (False) bool: If set, perresidue inpainting lddts will be collected as lists into the output DataFrame
        
    Returns:
        Pandas DataFrame containing all scores and pdb-file locations (+names)
    '''
    # collect scores from .files into one pandas DataFrame
    pl = glob(f"{inpaint_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"ERROR: No .pdb files were found in the inpainting output direcotry {inpaint_dir}. Inpainting might have crashed (check inpainting error-log), or the path might be wrong!")

    # collect inpaint scores into DataFrame by parsing each .trbfile for which a .pdb file exists
    df = pd.concat([parse_inpainting_trbfile(p.replace(".pdb", ".trb"), perres_lddt=perres_lddt, perres_inpaint_lddt=perres_inpaint_lddt) for p in pl])

    # if option <rename_pdbs> is set, update indeces to standard indexing (using zfill)
    if rename_pdbs:
        # rename description to standard indexing (_0001 instead of _1, etc...)
        df.loc[:, "new_description"] = ["_".join(desc.split("_")[:-1]) + "_" + str(int(desc.split("_")[-1]) + 1).zfill(4) for desc in df["description"]]
        df.loc[:, "new_loc"] = [loc.replace(old_desc, new_desc) for loc, old_desc, new_desc in zip(list(df["location"]), list(df["description"]), list(df["new_description"]))]

        # rename all inpainting outputfiles according to new indeces:
        _empty = [[os.rename(f, f.replace(old_desc, new_desc)) for f in glob(f"{inpaint_dir}/{old_desc}.*")] for old_desc, new_desc in zip(list(df["description"]), list(df["new_description"]))]
        #_empty = [os.rename(old, new) for old, new in zip(df["location"], df["new_loc"])]

        # Collect information of path to .pdb files into DataFrame under 'location' column
        df = df.drop(columns=["location"]).rename(columns={"new_loc": "location"})
        df = df.drop(columns=["description"]).rename(columns={"new_description": "description"})

    return df.reset_index(drop=True)

def parse_inpainting_trbfile(trbfile: str, perres_lddt=False, perres_inpaint_lddt=False) -> pd.DataFrame:
    '''
    Reads scores from inpainting trb-file
    Args:
        <trbfile>                        Path to inpainting .trb file.
        <perres_lddt>                    bool: If set, perresidue lddts will be collected as lists into the output DataFrame
        <perres_inpaint_lddt>            bool: If set, perresidue inpainting lddts will be collected as lists into the output DataFrame
    '''
    # read trbfile:
    if trbfile.endswith(".trb"): data_dict = np.load(trbfile, allow_pickle=True)
    else: raise ValueError(f"ERROR: only .trb-files can be passed into parse_inpainting_trbfile. <trbfile>: {trbfile}")

    # instantiate scoresdict and start collecting:
    sd = dict()
    sd["lddt"] = np.mean(data_dict["lddt"])
    sd["inpaint_lddt"] = np.mean(data_dict["inpaint_lddt"])
    sd["template"] = data_dict["settings"]["pdb"]
    sd["con_ref_pdb_idx"] = [data_dict["con_ref_pdb_idx"]]
    sd["con_hal_pdb_idx"] = [data_dict["con_hal_pdb_idx"]]
    sd["res_mapping"] = [[(x[1], y[1]) for x, y in zip(data_dict["con_ref_pdb_idx"], data_dict["con_hal_pdb_idx"])]]
    sd["location"] = trbfile.replace(".trb", ".pdb")
    sd["description"] = trbfile.split("/")[-1].replace(".trb", "")

    if perres_lddt: sd["perresidue_lddt"] = [data_dict["lddt"]]
    if perres_inpaint_lddt: sd["perresidue_inpaint_lddt"] = [data_dict["inpaint_lddt"]]

    return pd.DataFrame(sd)

def write_inpainting_cmd(pose: str, options: dict, flags: list, out_dir: str) -> str:
    '''
    Writes an inpainting command that can be run on acluster on a slurm script.
    Args:
        <pose>
        <options>
        <flags>
    Returns:
        cmd that can be run on the cluster
    '''
    if options == None: options = {}
    if flags == None: flags = []

    # parse pose name as output into options:
    options["out"] = out_dir + pose.split("/")[-1].replace(".pdb", "")
    if not "dump_all" in flags: flags.append("dump_all")

    options_string = " ".join([f"--{k}={v}" for k, v in options.items()])
    flags_string = " ".join(["--"+flag for flag in flags])

    return f"{_rfdesign_python} {_inpaint_path}/inpaint.py --pdb {pose} {options_string} {flags_string}"

def trf_relax_dir(path_to_dir: str, max_array_size=512) -> str:
    '''
    runs trf_relax script on inpaints.
    '''
    # sanity check
    if not os.path.isdir(path_to_dir): raise FileNotFoundError

    # check if outputs are already present:
    if os.path.isdir((trf_dir := f"{path_to_dir}/trf_relax")):
        if len(glob(f"{path_to_dir}/*.pdb")) == len(glob(f"{trf_dir}/*.pdb")):
            ### IMPORTANT: Here, I use the number of .pdb files in the inpainting output directory and the trf_relax directory as an indirect metric.
            print(f"Inpaints at {path_to_dir} are already relaxed at {trf_dir}. Skipping trf_relax.sh")
            return trf_dir

    # create list of commands to run trf_relax
    cmdfile = f"{path_to_dir}/trf_relax_commands"
    run(f"{_trf_relax_script_path} {path_to_dir} {cmdfile}", stdout=True, stderr=True, check=True, shell=True)

    with open(cmdfile, 'r') as f:
        cmds = [x.strip() for x in f.readlines() if x]

    # sbatch commandslist
    sbatch_options = [f"-e {path_to_dir}/trf_relax.err -o {path_to_dir}/trf_relax.out"]
    sbatch_array_jobstarter(cmds, sbatch_options, jobname="trf_relax", max_array_size=max_array_size, remove_cmdfile=False, cmdfile_dir=path_to_dir)

    return trf_dir

def run_hallucination(poses: list, work_dir: str, options="", pose_options=None, scorefile="inpainting_scores.json", max_gpus=5, perres_lddt=False, perres_inpaint_lddt=False) -> dict:
    '''
    Important: options specified in <pose_options> overwrite global options specified in <options>!
    '''
    # setup workdir:
    work_dir = os.path.abspath(work_dir)
    pdb_dir = f"{work_dir}/output_pdbs/"
    if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

    # setup pdb_dir and look for output file. Skip inpainting, if output is present:
    return_dict = {"pdb_dir": (pdb_dir := f"{work_dir}/output_pdbs/")}
    if os.path.isfile((hallucination_scorefile := f"{work_dir}/{scorefile}")):
        return {"pdb_dir": pdb_dir, "scores": pd.read_json(hallucination_scorefile)}

    # parse options, flags, pose_options and pose_flags
    options, flags = parse_options_flags(options, sep='--')
    if pose_options:
        #sanity check:
        if len(poses) != len(pose_options): raise ValueError(f"ERROR: Arguments <poses> and <pose_option> of the function run_inpainting() must be of the same length!\nlen(poses) = {len(poses)}\nlen(pose_options) = {len(pose_options)}")
        pose_options, pose_flags = zip(*[parse_options_flags(opt_str, sep="--") for opt_str in pose_options])
    else: 
        pose_options, pose_flags = [{} for x in poses], [[] for x in poses]

    # write inpainting commands for each pose in poses:
    cmds = [write_hallucination_cmd(pose, options=dict(options, **pose_opts), flags=list(set(flags) | set(pose_flgs)), out_dir=pdb_dir) for pose, pose_opts, pose_flgs in zip(poses, pose_options, pose_flags)]

    # execute inpainting commands in a jobarray on maximum <max_gpus>
    sbatch_options = [f"--gpus-per-node 1 -c1 -e {work_dir}/hallucination_err.log -o {work_dir}/hallucination_out.log"]
    sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="hallucinate_poses", 
                                  max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=work_dir)

    # collect inpainting scores and return outputs.
    return_dict["scores"] = collect_hallucination_scores(hallucination_dir=pdb_dir, scorefile=hallucination_scorefile, rename_pdbs=True)
    return_dict["scores"].to_json(hallucination_scorefile)

    return return_dict

def write_hallucination_cmd(pose: str, options: dict, flags: list, out_dir: str) -> str:
    '''
    Writes an hallucination command that can be run on acluster on a slurm script.
    Args:
        <pose>
        <options>
        <flags>
    Returns:
        cmd that can be run on the cluster
    '''
    if options is None: options = {}
    if flags is None: flags = []

    # parse pose name as output into options:
    options["out"] = out_dir + pose.split("/")[-1].replace(".pdb", "")

    options_string = " ".join([f"--{k}={v}" for k, v in options.items()])
    flags_string = " ".join(["--"+flag for flag in flags])

    return f"{_rfdesign_python} {_hallucination_path}/hallucinate.py --pdb {pose} {options_string} {flags_string}"

def collect_hallucination_scores(hallucination_dir: str, scorefile="hallucination_scores.json", rename_pdbs=True) -> pd.DataFrame:
    '''
    Collects scores from .trb and .npz files of inpainting and renames (if <rename_pdbs> is set) output .pdbs from output_1.pdb to output_0001.pdb
    Args:
        <inpaint_dir>                   Output directory of your inpainting run.
        <scorefile>                     Name of the scorefile you want to write
        <rename_pdbs>                   (True)  bool: If set, renames .pdb files in dataframe and inpaint_dir to standard indeces (_0001 instead of _1, etc.)
        <perres_lddt>                   (False) bool: If set, perresidue lddts will be collected as lists into the output DataFrame
        <perres_inapint_lddt>           (False) bool: If set, perresidue inpainting lddts will be collected as lists into the output DataFrame
        
    Returns:
        Pandas DataFrame containing all scores and pdb-file locations (+names)
    '''
    # collect scores from .files into one pandas DataFrame
    pl = glob(f"{hallucination_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"ERROR: No .pdb files were found in the hallucination output direcotry {hallucination_dir}. Hallucination might have crashed (check inpainting error-log), or the path might be wrong!")

    # collect inpaint scores into DataFrame by parsing each .trbfile for which a .pdb file exists
    df = pd.concat([parse_hallucination_trbfile(p.replace(".pdb", ".trb")) for p in pl])

    # if option <rename_pdbs> is set, update indeces to standard indexing (using zfill)
    if rename_pdbs:
        # rename description to standard indexing (_0001 instead of _1, etc...)
        df.loc[:, "new_description"] = ["_".join(desc.split("_")[:-1]) + "_" + str(int(desc.split("_")[-1]) + 1).zfill(4) for desc in df["description"]]
        df.loc[:, "new_loc"] = [loc.replace(old_desc, new_desc) for loc, old_desc, new_desc in zip(list(df["location"]), list(df["description"]), list(df["new_description"]))]

        # rename all inpainting outputfiles according to new indeces:
        _empty = [[os.rename(f, f.replace(old_desc, new_desc)) for f in glob(f"{hallucination_dir}/{old_desc}.*")] for old_desc, new_desc in zip(list(df["description"]), list(df["new_description"]))]

        # Collect information of path to .pdb files into DataFrame under 'location' column
        df = df.drop(columns=["location"]).rename(columns={"new_loc": "location"})
        df = df.drop(columns=["description"]).rename(columns={"new_description": "description"})

    return df.reset_index(drop=True)

def parse_hallucination_trbfile(trbfile: str) -> pd.DataFrame:
    '''
    Reads scores from inpainting trb-file
    Args:
        <trbfile>                        Path to inpainting .trb file.
        <perres_lddt>                    bool: If set, perresidue lddts will be collected as lists into the output DataFrame
        <perres_inpaint_lddt>            bool: If set, perresidue inpainting lddts will be collected as lists into the output DataFrame
    '''
    # read trbfile:
    if trbfile.endswith(".trb"): data_dict = np.load(trbfile, allow_pickle=True)
    else: raise ValueError(f"ERROR: only .trb-files can be passed into parse_inpainting_trbfile. <trbfile>: {trbfile}")

    # instantiate scoresdict and start collecting:
    sd = dict()
    for loss in [x for x in data_dict.keys() if x.startswith("loss_")]:
        sd[loss] = data_dict[loss][0] # take first element, because, for an unknown reason, losses are stored in 1D lists ([loss_value]).
    sd["sampled_mask"] = data_dict["sampled_mask"]
    sd["template"] = data_dict["settings"]["pdb"]
    sd["con_ref_pdb_idx"] = [data_dict["con_ref_pdb_idx"]]
    sd["con_hal_pdb_idx"] = [data_dict["con_hal_pdb_idx"]]
    sd["res_mapping"] = [[(x[1], y[1]) for x, y in zip(data_dict["con_ref_pdb_idx"], data_dict["con_hal_pdb_idx"])]]
    sd["location"] = trbfile.replace(".trb", ".pdb")
    sd["description"] = trbfile.split("/")[-1].replace(".trb", "")

    return pd.DataFrame(sd)

def parse_tuples_into_dict(input_list: list[tuple]) -> dict:
    '''
    Convert a list of tuples into a dictionary where the first element of each tuple is the key and the second element is added to a list associated with that key.

    Parameters:
    input_list (list[tuple]): The list of tuples to be converted
    
    Returns:
    dict: The resulting dictionary
    '''
    d = defaultdict(list)
    [d[a].append(b) for a, b in input_list]
    return dict(d)

def compile_motif_from_inpaint_trb(input_trb: str) -> dict:
    '''
    Compile reference and target motif of an inpainting trb-file into a motif.

    Parameters:
    input_trb (str): The path to the input trb-file
    
    Returns:
    dict: A dictionary containing reference and target motifs
    '''
    # sanity
    if not os.path.isfile(input_trb): raise FileNotFoundError

    # load data_dict from file
    trb_dict = np.load(input_trb, allow_pickle=True)

    # compile motif dict from data_dict and return  
    return parse_tuples_into_dict(trb_dict["con_ref_pdb_idx"]), parse_tuples_into_dict(trb_dict["con_hal_pdb_idx"])

def calc_inpaint_trf_motif_rmsd(inpaint_dir: str, trf_dir:str=None, out_scorefile:str=None):
    '''
    Calculates motif RMSD after running trf_relax.sh on an inpainting output.
    '''
    # specify trf_dir and scorefile:
    trf_dir = trf_dir or f"{inpaint_dir}/trf_relax/"

    # collect trb-files and from them derive poses and description
    if not (fl := glob(f"{inpaint_dir}/*.trb")): raise FileNotFoundError(f"ERROR: No *.trb files found at {trf_dir}")
    poses = [x.split("/")[-1].replace(".trb", ".pdb") for x in fl]
    descriptions = [x.replace(".pdb", "") for x in poses]

    # compile motifs for rmsd calculation from trb-files
    motifs = [compile_motif_from_inpaint_trb(f) for f in fl]

    # run motif_rmsd calculation (inpaint_dir/pose trf_dir/pose motif atoms="CA") for every pose
    rmsds = [motif_rmsd.superimpose_calc_motif_rmsd(f"{inpaint_dir}/{pose}", f"{trf_dir}/{pose}", ref_selection=motif[1], target_selection=motif[1]) for pose, motif in zip(poses, motifs)]

    # compile RMSD DataFrame
    df = pd.DataFrame({"description": descriptions, "trf_motif_bb_ca_rmsd": rmsds})

    # store DataFrame if out_scorefile option is set
    if out_scorefile:
        df.to_json(out_scorefile)

    return df

## ------------------ RFDiffusion -------------------------------------------------
def run_rfdiffusion(poses: list, work_dir:str, options="", pose_options=None, scorefile="rfdiffusion_scores.json", max_gpus=5) -> dict:
    '''run RFDiffusion on acluster'''
    # setup workdir
    work_dir = os.path.abspath(work_dir)
    pdb_dir = f"{work_dir}/output_pdbs/"
    if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

    # Look for output-file in pdb-dir. If output is present and correct, then skip diffusion step.
    rfdiff_out_dict = {"pdb_dir": pdb_dir}
    if os.path.isfile((scorefilepath := f"{work_dir}/{scorefile}")):
        return {"pdb_dir": pdb_dir, "scores": pd.read_json(scorefilepath)}

    # parse options and pose_options:
    if pose_options:
        if len(poses) != len(pose_options): raise ValueError(f"Arguments <poses> and <pose_options> for RFDiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
    else:
        pose_options = ["" for x in poses]

    # write rfdiffusion cmds
    cmds = [write_rfdiffusion_cmd(pose, options, pose_opts, output_dir=rfdiff_out_dict["pdb_dir"]) for pose, pose_opts in zip(poses, pose_options)]

    #options, pose_options = parse_rfdiffusion_opts(options, pose_options, n=len(poses))
    sbatch_options = [f"--gpus-per-node 1 -c1 -e {work_dir}/rfdiffusion_err.log -o {work_dir}/rfdiffusion_out.log"]
    sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="diffuse_poses",
                                  max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=work_dir)

    # collect rfdiff scores and return outputs:
    rfdiff_out_dict["scores"] = collect_rfdiffusion_scores(input_dir=rfdiff_out_dict["pdb_dir"], scorefile=scorefile, rename_pdbs=True)
    rfdiff_out_dict["scores"].to_json(scorefilepath)

    return rfdiff_out_dict

def collect_rfdiffusion_scores(input_dir: str, scorefile: str, rename_pdbs=True) -> pd.DataFrame:
    '''collects and returns (most likely used) RFDiffusion scores'''
    # collect scores from .trb-files into one pandas DataFrame:
    pl = glob(f"{input_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"No .pdb files were found in the diffusion output direcotry {input_dir}. RFDiffusion might have crashed (check inpainting error-log), or the path might be wrong!")

    # collect hallucination scores into a DataFrame:
    df = pd.concat([parse_diffusion_trbfile(p.replace(".pdb", ".trb")) for p in pl])

    # rename pdbs if option is set:
    if rename_pdbs:
        df.loc[:, "new_description"] = ["_".join(desc.split("_")[:-1]) + "_" + str(int(desc.split("_")[-1]) + 1).zfill(4) for desc in df["description"]]
        df.loc[:, "new_loc"] = [loc.replace(old_desc, new_desc) for loc, old_desc, new_desc in zip(list(df["location"]), list(df["description"]), list(df["new_description"]))]

        # rename all diffusion outputfiles according to new indeces:
        print(df[["location", "description", "new_description", "new_loc"]])
        _empty = [[os.rename(f, f.replace(old_desc, new_desc)) for f in glob(f"{input_dir}/{old_desc}.*")] for old_desc, new_desc in zip(list(df["description"]), list(df["new_description"]))]

        # Collect information of path to .pdb files into DataFrame under 'location' column
        df = df.drop(columns=["location"]).rename(columns={"new_loc": "location"})
        df = df.drop(columns=["description"]).rename(columns={"new_description": "description"})

    return df.reset_index(drop=True)

def parse_diffusion_trbfile(path: str) -> pd.DataFrame:
    '''AAA'''
    # read trbfile:
    if path.endswith(".trb"): data_dict = np.load(path, allow_pickle=True)
    else: raise ValueError(f"only .trb-files can be passed into parse_inpainting_trbfile. <trbfile>: {path}")

    # calc mean_plddt:
    sd = dict()
    last_plddts = data_dict["plddt"][-1]
    sd["plddt"] = [sum(last_plddts) / len(last_plddts)]
    sd["perres_plddt"] = [last_plddts]

    # instantiate scoresdict and start collecting:
    scoreterms = ["con_hal_pdb_idx", "con_ref_pdb_idx", "sampled_mask"]
    for st in scoreterms:
        sd[st] = [data_dict[st]]

    # collect metadata
    sd["location"] = path.replace(".trb", ".pdb")
    sd["description"] = path.split("/")[-1].replace(".trb", "")
    sd["input_pdb"] = data_dict["config"]["inference"]["input_pdb"]

    return pd.DataFrame(sd)

def write_rfdiffusion_cmd(pose: str, options: str, pose_opts: str, output_dir: str) -> str:
    '''AAA'''
    # parse description:
    desc = pose.split("/")[-1].split(".")[0]

    # parse options:
    start_opts = parse_rfdiffusion_opts(options, pose_opts)

    start_opts["inference.output_prefix"] = f"{output_dir}/{desc}"
    start_opts["inference.input_pdb"] = pose
    opts_str = " ".join([f"{k}={v}" for k, v in start_opts.items()])

    # return cmd
    return f"{_rfdiffusion_python_env} {_rfdiffusion_inference_script} {opts_str}"

def parse_rfdiffusion_opts(options: str, pose_options: str) -> dict:
    '''AAA'''
    def re_split_rfdiffusion_opts(command) -> list:
        return re.split(r"\s+(?=(?:[^']*'[^']*')*[^']*$)", command)
    splitstr = [x for x in re_split_rfdiffusion_opts(options) + re_split_rfdiffusion_opts(pose_options) if x]# adding pose_opts after options makes sure that pose_opts overwrites options!
    return {x.split("=")[0]: "=".join(x.split("=")[1:]) for x in splitstr}

## ------------------ ProteinMPNN --------------------------------------------------

def check_fixed_positions(fixed_positions_list: list) -> list:
    '''
    Checks if the fixed_positions in fixed_positions_list are of the correct format.
    [
    {"A": [], "B": []}
    {"A": [], "B": []}
    ]
    '''
    print(f"\nChecking fixed_positions passed to ProteinMPNN for correct structure.")
    for fixedpos_dict in fixed_positions_list:
        if not isinstance(fixedpos_dict, dict):
            raise TypeError(f"ERROR: Your fixed_positions_list has the wrong structure. To find the correct structure, look at the examples in the ProteinMPNN directory!")
        for key, values in fixedpos_dict.items():
            try:
                fixedpos_dict[key] = [int(x) for x in values]
            except ValueError as exc:
                raise TypeError(f"Wrong value for fixed position. Has to be of type int! To find the correct structure (with correct types) for fixedpositions dict, look at the examples in the ProteinMPNN directory.") from exc

    return fixed_positions_list

def write_keysvalues_to_file(keys: list, values: list, outfile_name: str) -> str:
    '''
    Writes a key-value pair submitted as two lists [keys], [values] to a json file.
    Function returns name of the json file.
    Args:
        <keys>:          list of keys
        <values>:        list of values
        <outfile_name:   name of the file you want to write
    '''
    # compile dictionary
    write_dict = {k: v for k, v in zip(keys, values)}

    # write dictionary to json-file
    with open(outfile_name, 'w', encoding="UTF-8") as f:
        f.write(json.dumps(write_dict))

    return outfile_name

def write_proteinmpnn_cmd(pose_json, mpnn_options: dict, pose_options=None, use_soluble_model=False):
    '''
    Writes commandline to run ProteinMPNN on the cluster.
    <mpnn_option> are options for running ProteinMPNN that apply to ALL poses.
    Options for individual poses have to be set with <pose_options>
    '''
    # if pose_options are set, then update (and overwrite if necessary) mpnn_options with pose_options
    if pose_options:
        mpnn_options.update(pose_options)

    # parse options as --key=value from {mpnn_options}
    options = ' '.join([f"--{key}={str(value)}" for key, value in mpnn_options.items()])
    cmd = f"{_python_path} {_proteinmpnn_path}/protein_mpnn_run.py --jsonl_path={pose_json} {options}"
    if use_soluble_model: cmd += " --use_soluble_model"
    return cmd

def sbatch_proteinmpnn(mpnn_cmds, mpnn_options, max_gpus=5):
    '''
    Runs ProteinMPNN on a slurm cluster. Assumes that GPUs are available on the cluster.
    '''
    sbatch_options = ["--gpus-per-node 1", "-c2", f'-e {mpnn_options["out_folder"]}/ProteinMPNN_err.log -o {mpnn_options["out_folder"]}/ProteinMPNN_out.log']
    sbatch_array_jobstarter(mpnn_cmds, sbatch_options, jobname="mpnn", max_array_size=max_gpus, 
                            remove_cmdfile=False, cmdfile_dir=mpnn_options["out_folder"]+"/")
    return

def proteinmpnn(poses: list, mpnn_options: dict, pose_options=None, max_gpus=5, scorefile="mpnn_scores.json", use_soluble_model:bool=False):
    '''
    Runs ProteinMPNN on the cluster with <mpnn_options> for all poses. Individual options for poses can be specified with pose_options.
    
    !!! scorefile is written into mpnn_options['out_folder'] !!!
    '''
    # If no pose_options are set, make sure that None is passed for each pose to the function write_proteinmpnn_cmd():
    if not pose_options: pose_options = [None for pose in poses]

    # Check if mpnn_scorefile exists at mpnn_out_folder, if yes, just return the DataFrame from the scorefile.
    if os.path.isfile((mpnn_scorefile := f"{mpnn_options['out_folder']}/{scorefile}")):
        # if scorefile already exists, just read scorefile and return scores from there.
        with open(mpnn_scorefile, 'r', encoding="UTF-8") as f:
            scores_df = json.load(f)
        print(f"\nScorefile for ProteinMPNN run {mpnn_options['out_folder']} already exists at {mpnn_scorefile}. Skipping step.")
        return scores_df

    # create mpnn_out_dir:
    if not os.path.isdir(mpnn_options['out_folder']): os.makedirs(mpnn_options["out_folder"], exist_ok=True)

    # parse sequences
    jsonl_dir = f"{mpnn_options['out_folder']}/jsonl_dir/"
    os.makedirs(jsonl_dir, exist_ok=True)
    print(f"\nParsing poses into .json files at {jsonl_dir}")
    poses_jsonl_list = parse_poses(poses, jsonl_dir) # returns list of paths to jsonl files for poses

    # write mpnn commands and print the options
    mpnn_cmds = [write_proteinmpnn_cmd(pose_json, mpnn_options, pose_option, use_soluble_model=use_soluble_model) for pose_json, pose_option in zip(poses_jsonl_list, pose_options)]
    options_string = '\n'.join([f"--{k} {v}" for k, v in mpnn_options.items()])
    print(f"\nRunning ProteinMPNN with options: \n{options_string}")

    # run ProteinMPNN with sbatch
    sbatch_proteinmpnn(mpnn_cmds, mpnn_options, max_gpus=max_gpus) #(automatically waits for job to finish)
    scores_df = rename_mpnn_fastas(input_dir=f'{mpnn_options["out_folder"]}/seqs/', scorefile=mpnn_scorefile, write=True)

    return scores_df

# Prep ProteinMPNN pose_options string!

## ----------------------- Rosetta ------------------------------------------------------------

def run_rosetta(poses: list, rosetta_executable: str, rosetta_options: dict, rosetta_flags: list, pose_options: list[dict], pose_flags: list[list], n: int, work_dir: str, scorefile="rosetta_scores.sc", max_cpus=256, force_options=None) -> dict:
    '''
    Runs Rosetta executable on Slurm.
    Args:
        <poses>                   List of paths to pdb-files. Rosetta runs are run on all poses in this list.
        <rosetta_executable>      str, name of the Rosetta executable that you want to run.
        <relax_options>           dictionary, containing Rosetta commandline options. Example: {"-extra_res_fa": "rotlib/path"}
        <relax_flags>             list, containing Rosetta commandline flags. Example: ["-beta", "-ex1", "-ex2"]
        <n>                       int, overwrites the Rosetta option -nstruct. This is implemented as a separate argument to maximize parallel running CPUs.
        <work_dir>                str, working directory of the relax application.
    
    Returns:
        Dictionary containg ["pdb_dir"] and ["scores"]
    '''
    # setup output_dir and return dictionary
    rosetta_path = find_rosetta_path(_rosetta_paths, rosetta_executable, stringent=True)
    rosetta_work_dir = os.path.abspath(work_dir)
    if not os.path.isdir(rosetta_work_dir): os.makedirs(rosetta_work_dir, exist_ok=True)

    # if relax outputs are already present, skip relax step
    logfile = f"{rosetta_work_dir}/log.txt"
    return_dict = {"pdb_dir": rosetta_work_dir}
    rosetta_scorefile = f"{rosetta_work_dir}/rosetta_scores.json"
    donefile = f"{rosetta_work_dir}/done.txt"
    if os.path.isfile(rosetta_scorefile):
        scores = pd.read_json(rosetta_scorefile)
        if os.path.isfile(donefile):
            return_dict["scores"] = scores
            print(f"\nOutputs of rosetta run {rosetta_work_dir} already found at {rosetta_work_dir}/rosetta_scores.json\nSkipping rosetta step.")
            with open(logfile, 'w') as f:
                f.write("Outputs of rosetta run already found. Skipping Rosetta Step.")
            return return_dict
        else:
            with open(logfile, 'w') as f:
                f.write(f"Found outputs of Rosetta Run, but number of relaxed poses does not match number of expected poses. \nlen(scores) = {len(scores)}\nlen(poses)={len(poses)}\nn={n}")
            run(f"rm {rosetta_work_dir}/*", shell=True, stdout=True, stderr=True, check=True)

    # update relax_options and relax_flags with global variables!
    rosetta_options["out:file:scorefile"] = f"{rosetta_work_dir}/{scorefile}"
    rosetta_options["out:path:all"] = rosetta_work_dir

    # write relax commands for each pose in poses (for <n> relax runs)
    rosetta_cmds = list()
    for pose, pose_options_d, pose_flags_l in zip(poses, pose_options, pose_flags):
        for i in range(1, n+1):
            rosetta_cmds.append(write_rosetta_cmd(rosetta_path, pose=pose, i=i, options=rosetta_options, flags=rosetta_flags, pose_options=pose_options_d, pose_flags=pose_flags_l, force_options=force_options))

    #assert len(rosetta_cmds) <= 1000 # Slurm does not allow jobarrays bigger than 1000 jobs. Don't ask me why...

    # remove Rosetta Scorefile if one exists:
    if os.path.isfile(rosetta_options["out:file:scorefile"]): run(f"rm {rosetta_options['out:file:scorefile']}", shell=True, stderr=True, stdout=True, check=True)

    # execute relax commands in a jobarray on maximum <max_cpus>
    sbatch_options = [f"-e {rosetta_work_dir}/rosetta_err.log -o {rosetta_work_dir}/rosetta_out.log"]
    sbatch_array_jobstarter(cmds=rosetta_cmds, sbatch_options=sbatch_options, jobname="rosetta_poses", 
                            max_array_size=max_cpus, wait=True, remove_cmdfile=False, cmdfile_dir=rosetta_work_dir)

    # collect relax scores and rename relaxed pdbs.
    time.sleep(60) # Rosetta does not have time to write the last score into the scorefile otherwise?
    scores = collect_rosetta_scores(rosetta_options["out:path:all"], rosetta_options["out:file:scorefile"])

    # write donefile
    with open(donefile, 'w') as f:
        f.write("done")

    # return poses as path to pdb_dir and scores as DataFrame
    return {"scores": scores, "pdb_dir": rosetta_work_dir}

def run_relax(poses: list, relax_options: dict, relax_flags: list, pose_options: dict, pose_flags: list, n: int, work_dir: str, scorefile="relax_scores.sc", max_cpus=256) -> dict:
    '''
    Runs Rosetta Relax application on Slurm.
    <poses>: List of paths to pdb-files. Relax runs are run on all poses in this list.
    <relax_options>: dictionary, containing Rosetta commandline options. Example: {"-extra_res_fa": "rotlib/path"}
    <relax_flags>: list, containing Rosetta commandline flags. Example: ["-beta", "-ex1", "-ex2"]
    <n>: int, overwrites the Rosetta option -nstruct. This is implemented as a separate argument to maximize parallel running CPUs.
    <work_dir>: working directory of the relax application.
    '''
    # setup output_dir and return dictionary
    relax_path = find_rosetta_path(_rosetta_paths, "relax")
    relax_work_dir = os.path.abspath(work_dir)
    if not os.path.isdir(relax_work_dir): os.makedirs(relax_work_dir, exist_ok=True)
    donefile = f'{relax_work_dir}/done.txt'

    # if relax outputs are already present, skip relax step
    return_dict = {"pdb_dir": relax_work_dir}
    relax_scorefile = f"{relax_work_dir}/relax_scores.json"
    if os.path.isfile(donefile):
        if os.path.isfile(relax_scorefile):
            scores = pd.read_json(relax_scorefile)
            return_dict["scores"] = scores
            print(f"\nOutputs of relax run {relax_work_dir} already found at {relax_work_dir}/relax_scores.json\nSkipping relax step.")
            return return_dict
        else:
            print(f"\nInconsistent relax outputs. Removing old files and rerunning relax.")
            #run(f"rm {relax_work_dir}/*", shell=True, stdout=True, stderr=True, check=True)
    else:
        pass

    # update relax_options and relax_flags with global variables!
    relax_options["out:file:scorefile"] = f"{relax_work_dir}/{scorefile}"
    relax_options["out:path:all"] = relax_work_dir

    # write relax commands for each pose in poses (for <n> relax runs)
    relax_cmds = list()
    for pose, pose_options_d, pose_flags_l in zip(poses, pose_options, pose_flags):
        for i in range(1, n+1):
            relax_cmds.append(write_rosetta_cmd(relax_path, pose=pose, i=i, options=relax_options, flags=relax_flags, pose_options=pose_options_d, pose_flags=pose_flags_l))

    #assert len(relax_cmds) <= 1000 # Slurm does not allow jobarrays bigger than 1000 jobs. Don't ask me why...

    # remove Rosetta Scorefile if one exists:
    if os.path.isfile(relax_options["out:file:scorefile"]): run(f"rm {relax_options['out:file:scorefile']}", shell=True, stderr=True, stdout=True, check=True)

    # execute relax commands in a jobarray on maximum <max_cpus>
    sbatch_options = [f"-e {relax_work_dir}/relax_err.log -o {relax_work_dir}/relax_out.log"]
    sbatch_array_jobstarter(cmds=relax_cmds, sbatch_options=sbatch_options, jobname="relax_poses", 
                            max_array_size=max_cpus, wait=True, remove_cmdfile=False, cmdfile_dir=relax_work_dir)

    # collect relax scores and rename relaxed pdbs.
    scores = collect_rosetta_scores(relax_options["out:path:all"], relax_options["out:file:scorefile"])

    # write donefile
    with open(donefile, 'w') as f:
        f.write("Done")

    # return poses as path to pdb_dir and scores as DataFrame 
    return {"scores": scores, "pdb_dir": relax_work_dir}

def write_rosetta_cmd(path_to_executable: str, pose: str, i: int, options={}, flags=[], pose_options=None, pose_flags=None, force_options=None) -> str:
    '''
    Write command to run relax on a <pose>. 
    <i> is the index of the pose when multiple relax trajectories are run (which is default).
    This index (<i>) will be added to the beginning of the pdb as a prefix <r0001_>. 
    Args:
        <path_to_executable>       Path to the Rosetta executable that you would like to use 
        <pose>                     path to the pose on which the executable should be run on
        <i>                        Index that will be added to the pose as a prefix (gets corrected by rosetta scorecollector)
        <options>                  Global Rosetta options, as a dictionary e.g. {"out:path:all": "my_output/path"}
        <flags>                    Global Rosetta flags, as a list e.g. ["beta", "ex1", "ex2"]
        <pose_options>             Pose-level options, as a dictionary e.g. {"parser:script_vars": "resfile=/path/to/resfile"}
        <pose_flags>               Pose-level flags, as a list e.g. ["ignore_unrecognized_res", ...]
        
    Returns:
        Command for Rosetta executable that can be run by slurm.
        
    '''
    # if pose_options are set: overwrite general with pose options
    if pose_options:
        options.update(pose_options)

    # if pose_flags are set: merge general flags with pose_flags
    if pose_flags:
        flags = list(set(flags) | set(pose_flags))

    # parse rosetta options and flags into string
    options["out:prefix"] = f"r{str(i).zfill(4)}_"
    options_string = " ".join([f"-{k} {v}" for k, v in options.items()])
    flags_string = " ".join(["-"+x for x in flags])
    if force_options: 
        if not type(force_options) == str: raise TypeError(f"parameter <force_option> must be of type str. Type(force_options): {type(force_options)}")
        options_string += " " + force_options

    # write and return command
    return f"{path_to_executable} -s {pose} {options_string} {flags_string}"

def parse_rosetta_options_string(relax_options: str) -> tuple:
    '''
    Takes string (Rosetta commandline options) and parses them into options {key: value} and flags {value}
    
    Example: '-s input_pdb -constrain_relax_to_start_coords -nstruct 10 -beta '
        options: {"s": "input_pdb", "nstruct": "10"}
        flags: ["constrain_relax_to_start_coords", "beta"]
    '''
    firstsplit = [x.strip() for x in re.split(r'^-| -', relax_options) if x]

    options = dict()
    flags = list()
    for item in firstsplit:
        if len((x := item.split())) > 1:
            options[x[0]] = " ".join(x[1:])
        else:
            flags.append(x[0])

    return options, flags

def clean_rosetta_scorefile(path_to_file: str, out_path: str) -> str:
    '''AAA'''

    # read in file line-by-line:
    with open(path_to_file, 'r', encoding="UTF-8") as f:
        scores = [line.split() for line in list(f.readlines()[1:])]

    # if any line has a different number of scores than the header (columns), that line will be removed.
    scores_cleaned = [line for line in scores if len(line) == len(scores[0])]
    print(f"WARNING: {len(scores) - len(scores_cleaned)} scores were removed from Rosetta scorefile at {path_to_file}")

    # write cleaned scores to file:
    with open(out_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join([",".join(line) for line in scores_cleaned]))
    return out_path

def collect_rosetta_scores(rosetta_work_dir: str, scorefile: str) -> pd.DataFrame:
    '''
    Collects scores and reindeces .pdb files. Stores scores as .json file.
    '''
    # collect scores from Rosetta Scorefile
    try:
        scores = pd.read_csv(scorefile, delim_whitespace=True, header=[1], na_filter=True)
    except pd.errors.ParserError:
        print(f"WARNING: Error reading Rosetta Scorefile. Removing faulty scorelines. This means that a few calculations will be lost.")
        scores = pd.read_csv(clean_rosetta_scorefile(scorefile, f"{rosetta_work_dir}/clean_rosetta_scores.sc"))

    # remove rows from df that do not contain scores and remove "description" duplicates, because that's what happens in Rosetta...
    scores = scores[scores["SCORE:"] == "SCORE:"]
    scores = scores.drop_duplicates(subset="description")

    # create reindexed names of relaxed pdb-files: [r0003_pose_unrelaxed_0001.pdb -> pose_unrelaxed_0003.pdb]
    scores.rename(columns={"description": "raw_description"}, inplace=True)
    scores.loc[:, "description"] = scores["raw_description"].str.split("_").str[1:-1].str.join("_") + "_" + scores["raw_description"].str.split("_").str[0].str.replace("r", "")

    # wait for all Rosetta output files to appear in the output directory (for some reason, they are sometimes not there after the runs completed.)
    while len((fl := glob(f"{rosetta_work_dir}/r*.pdb"))) < len(scores):
        time.sleep(1)

    # rename .pdb files in work_dir to the reindexed names.
    names_dict = scores[["raw_description", "description"]].to_dict()
    print(f"Renaming and reindexing {len(scores)} Rosetta output .pdb files")
    for oldname, newname in zip(names_dict["raw_description"].values(), names_dict["description"].values()):
        shutil.move(f"{rosetta_work_dir}/{oldname}.pdb", (nf := f"{rosetta_work_dir}/{newname}.pdb"))
        if not os.path.isfile(nf):
            print(f"WARNING: Could not rename file {oldname} to {nf}\n Retrying renaming.")
            shutil.move(f"{rosetta_work_dir}/{oldname}.pdb", (nf := f"{rosetta_work_dir}/{newname}.pdb"))

    # Collect information of path to .pdb files into dataframe under "location" column
    scores.loc[:, "location"] = rosetta_work_dir + "/" + scores["description"] + ".pdb"

    # safetycheck rename all remaining files with r*.pdb into proper filename:
    if (remaining_r_pdbfiles := glob(f"{rosetta_work_dir}/r*.pdb")):
        for pdb_path in remaining_r_pdbfiles:
            pdb_path = pdb_path.split("/")[-1]
            idx = pdb_path.split("_")[0].replace("r", "")
            new_name = "_".join(pdb_path.split("_")[1:-1]).replace(".pdb", "") + "_" + idx + ".pdb"
            shutil.move(f"{rosetta_work_dir}/{pdb_path}", f"{rosetta_work_dir}/{new_name}")

    # reset index and write scores to file
    scores.reset_index(drop="True", inplace=True)
    scores.to_json(scorefile.replace(".sc", ".json"))

    return scores

## ----------------------- Structure Prediction -----------------------------------------------
###### ------------------- AlphaFold2 Functions -----------------------------------------------

def run_AlphaFold2(poses: list, af2_options: dict, pose_options:list=None, max_gpus=10, calc_rmsd=False):
    '''
    
    '''
    # drop 'output_dir' from of_options and prepare directories
    af_work_dir = af2_options.pop("output_dir")
    af_output_dir = af_work_dir + "/af2_preds"
    pdb_dir = af_work_dir + "/af2_pdbs"
    return_dict = {"pdb_dir": pdb_dir}
    af2_scorefile = f"{af_work_dir}/af2_scores.json"

    # check if AlphaFold2 output is already present, if so, skip
    if os.path.isfile(af2_scorefile):
        print(f"\nOutputs of prediction run {af_output_dir} already found at {af2_scorefile}\nSkipping prediction step.")
        return_dict["scores"] = pd.read_json(af2_scorefile)
        return return_dict

    # create output directories 
    print(f"\nPredicting {len(poses)} Structures at {af_output_dir}")
    af_input = af_work_dir + "/af_input/"
    os.makedirs(af_output_dir, exist_ok=True)
    os.makedirs(af_input, exist_ok=True)

    # split poses list into <max_gpus> sublists and write them to files as input to Alphafold.
    splitnum = len(poses) if len(poses) < max_gpus else max_gpus
    poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]
    pose_fastas = [mergefastas(poses, f"{af_input}/af_fasta_{str(i+1).zfill(4)}.fa", replace_colon=True) for i, poses in enumerate(poses_split)]

    # write AlphaFold cmds
    af_cmds = [write_af2_cmd(fasta, af_output_dir, af2_options) for fasta in pose_fastas]

    # sbatch AlphaFold
    sbatch_options = [f"--gpus-per-node 1 -c2 -e {af_output_dir}/AF_err.log -o {af_output_dir}/AF_out.log"]
    sbatch_array_jobstarter(cmds=af_cmds, sbatch_options=sbatch_options, jobname="af2_predict", 
                            max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=af_output_dir)

    # collect scores and pdbs
    return_dict["scores"] = collect_af2_scores(af_output_dir, pdb_dir=pdb_dir, scorefile=af2_scorefile, calc_rmsd=calc_rmsd)

    return return_dict

def summarize_af2_json(input_json: str, input_pdb: str) -> pd.DataFrame:
    '''
    Takes raw AF2_scores.json file and calculates mean pLDDT over the entire structure, also puts perresidue pLDDTs and paes in list.
    
    Returns pd.DataFrame
    '''
    df = pd.read_json(input_json)
    means = df.mean(numeric_only=True).to_frame().T
    means["plddt_list"] = [df["plddt"]]
    means["pae_list"] = [df["pae"]]
    means["json_file"] = input_json
    means["pdb_file"] = input_pdb
    return means

def calc_statistics_over_AF2_models(index, input_tuple_list: "list[tuple[str,str]]", calc_rmsd=None, calc_motif_rmsd=None) -> list:
    '''
    takes list of .json files from af2_predictions and collects scores (mean_plddt, max_plddt, etc.)

    <calc_motif_rmsd>: Has to be list of residue IDs that comprise the motif. Example (calc_motif_rmsd=[1, 2, 10, 15])
    '''
    df = pd.concat([summarize_af2_json(af2_tuple[0], af2_tuple[1]) for af2_tuple in input_tuple_list], ignore_index=True)
    means = df.mean(numeric_only=True).to_frame().T.add_prefix("mean_")
    stds = df.std(numeric_only=True).to_frame().T.add_prefix("std_")
    top = df[df["json_file"].str.split("/").str[-1].str.contains("rank_001_", regex=True, na=False)].add_prefix("top_").reset_index(drop=True)
    top_pdb = df[df["pdb_file"].str.split("/").str[-1].str.contains("rank_001_", regex=True, na=False)].add_prefix("top_").reset_index(drop=True)["top_pdb_file"]
    return_df = pd.concat([top, means, stds], axis=1)
    return_df["description"] = index
    return_df["top_pdb_file"] = top_pdb

    # If calc_rmsd option is set, then calculate bb_ca_rmsd of each model to the rank_1 model.
    if calc_rmsd:
        return_df["mean_af2_ca_rmsd"] = calc_af2_rmsd(df)

    if calc_motif_rmsd:
        assert isinstance(calc_motif_rmsd, list) # calc_motif_rmsd has to be list with motif IDs, e.g.: [1, 2, 10, 15]
        return_df["mean_af2_motif_heavy_rmsd"] = calc_af2_motif_rmsd(df, calc_motif_rmsd)

    return return_df.set_index("description")

def calc_af2_rmsd(df: pd.DataFrame) -> float:
    '''
    Input:
        <df> pd.DataFrame, has to be DataFrame of AF2 prediction (collected by summarize_af2_json)
    Returns mean_ca_rmsd of rank_1 to all other models.
    '''
    # separate rank_1 from the rest.
    #df.loc[:, "pdb"] = df["json_file"].str.replace("_scores.json", ".pdb", regex=False)
    ref = list(df[df["pdb_file"].str.contains("rank_001_", regex=True, na=False)]["pdb_file"])[0]
    targets = list(df[~df["pdb_file"].str.contains("rank_001_", regex=True, na=False)]["pdb_file"])

    # calculate rmsd of "targets" to rank_1
    rmsds = [bb_rmsd.superimpose_calc_rmsd(ref, target) for target in targets]

    return np.mean(rmsds)

def calc_af2_motif_rmsd(df: pd.DataFrame, motif: list) -> dict:
    '''

    '''
    # separate rank_1 from the rest.
    #df.loc[:, "pdb"] = df["json_file"].str.replace("_scores.json", ".pdb", regex=False)
    ref = list(df[df["pdb_file"].str.contains("rank_001_", regex=True, na=False)]["pdb_file"])[0]
    targets = list(df[~df["pdb_file"].str.contains("rank_001_", regex=True, na=False)]["pdb_file"])

    # calculate motif_rmsd
    rmsds = [motif_rmsd.superimpose_calc_rmsd_heavy(ref, target, motif, motif) for target in targets]

    return np.mean(rmsds)

def collect_af2_pdb(description, pdb, pdb_dir):
    '''
    Takes pdb-file of af2_localcolabfold output and copies it (removing the rank_1 stuff..) into <pdb_dir>
    
    Returns new location of pdb file in <pdb_dir>.
    '''
    new_pdb_path = f"{pdb_dir}/{description}.pdb"
    shutil.copy(pdb, new_pdb_path)
    return new_pdb_path

def collect_af2_scores(af_output_dir, pdb_dir, scorefile=f"af2_scores.json", calc_rmsd=False, calc_motif_rmsd=None):
    '''
    
    '''
    def get_json_files_of_description(description: str, dir: str) -> str:
        return sorted([filepath for filepath in glob(f"{dir}/*/{description}*rank*.json") if re.search(f"{description}_scores_rank_..._.*_model_._seed_...\.json", filepath)])

    def get_pdb_files_of_description(description: str, dir: str) -> str:
        pdbfiles = sorted([filepath for filepath in glob(f"{dir}/*/{description}*rank*.pdb") if re.search(f"{description}_.?.?relaxed_rank_..._.*_model_._seed_...\.pdb", filepath)])
        return pdbfiles

    def get_json_pdb_tuples_from_description(description: str, dir: str) -> "list[tuple[str,str]]":
        '''Collects af2-output scores.json and .pdb file for a given 'description' as corresponding tuples (by sorting).'''
        return [(jsonf, pdbf) for jsonf, pdbf in zip(get_json_files_of_description(description, dir), get_pdb_files_of_description(description, dir))]

    # create pdb_dir
    if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

    # collect all unique 'descriptions' leading to predictions
    descriptions = [x.split("/")[-1].replace(".done.txt", "") for x in glob(f"{af_output_dir}/*/*.done.txt")]
    if not descriptions: raise FileNotFoundError(f"ERROR: No AF2 prediction output found at {af_output_dir} Are you sure it was the correct path?")

    # Collect all .json and corresponding .pdb files of each 'description' into a dictionary. (This collects scores from all 5 models)
    scores_dict = {description: get_json_pdb_tuples_from_description(description, af_output_dir) for description in descriptions}
    if not scores_dict: raise FileNotFoundError("No .json files were matched to the AF2 output regex. Check AF2 run logs. Either AF2 crashed or the AF2 regex is outdated (check at function 'collect_af2_scores()'")

    # Calculate statistics over prediction scores for each of the five models.
    af2_preds_df_list = [calc_statistics_over_AF2_models(description, af2_output_tuple_list, calc_rmsd=calc_rmsd, calc_motif_rmsd=calc_motif_rmsd) for description, af2_output_tuple_list in scores_dict.items()]

    # concatenate all individual score DataFrames into one single (and central) DataFrame:
    scores_df = pd.concat(af2_preds_df_list)

    # Copy *_rank_1*.pdb files to pdb_dir and store location in DataFrame
    scores_df = scores_df.reset_index()
    scores_df.loc[:, "location"] = [collect_af2_pdb(desc, top_pdb, pdb_dir) for desc, top_pdb in zip(scores_df["description"].to_list(), scores_df["top_pdb_file"].to_list())]

    scores_df.to_json(scorefile)

    return scores_df

def write_af2_cmd(fa_file, af_output_dir, options):
    '''
    
    '''
     # create of_output_dir/fa_file
    af_output_dir = f"{af_output_dir}/{fa_file.split('/')[-1].replace('.fa', '')}"
    print(f"Creating AlphaFold2 output directory at {af_output_dir}")
    os.makedirs(af_output_dir, exist_ok=True)

    options = " ".join([f"--{k} {v}" for k, v in options.items()])
    af_cmd = f"{_af2_path}colabfold_batch {options} {fa_file} {af_output_dir}"
    return af_cmd

# ---------------------------- ESMFold Functions --------------------------------------------

def run_ESMFold(poses: list, esm_options: dict, pose_options:list=None, max_gpus:int=10) -> dict:
    '''
    Runs <ESMFold> on poses with <esm_options> using maximum <max_gpus>
    '''
    # setup work_dir
    work_dir = os.path.abspath(esm_options.pop("output_dir"))
    out_dict = {"pdb_dir": (pdb_dir := f"{work_dir}/esm_pdbs/")}

    # check if results of run are already present, if so, skip prediction:
    if os.path.isfile((scorefile := f"{work_dir}/ESMFold_scores.json")):
        print(f"\nOutputs of prediction run {work_dir} already found at {scorefile}\nSkipping prediction step.")
        out_dict["scores"] = pd.read_json(scorefile)
        return out_dict

    # create output_dir and directory for input fasta-files:
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs((fasta_dir := f"{work_dir}/input_fastas"), exist_ok=True)
    os.makedirs((esm_preds_dir := f"{work_dir}/esm_preds"), exist_ok=True)

    # split poses list into <max_gpus> sublists and write them to files as input for ESMFold
    pose_fastas = prep_fastas_for_prediction(poses=poses, fasta_dir=fasta_dir, max_filenum=max_gpus)

    # write ESMFold commands:
    esm_cmds = [write_esm_cmd(fasta, esm_preds_dir, esm_options) for fasta in pose_fastas]

    # sbatch ESMFold:
    sbatch_options = [f"--gpus-per-node 1 -c1 -e {work_dir}/esm_err.log -o {work_dir}/esm_out.log"]
    sbatch_array_jobstarter(cmds=esm_cmds, sbatch_options=sbatch_options, jobname="esmfold",
                            max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=work_dir)

    # collect scores and pdbs
    out_dict["scores"] = collect_esm_scores(esm_preds_dir, pdb_dir=pdb_dir, scorefile=scorefile)

    return out_dict

def write_esm_cmd(fa_file: str, output_dir: str, options: dict) -> str:
    '''
    Writes commandlines for ESMFold.
    '''
    # overwrite options with globally set ESMFold options!
    options = dict(options, **_esm_opts)
    options = " ".join([f"--{k} {v}" for k, v in options.items()])
    return f"{_esmfold_inference_script} --fasta {fa_file} --output_dir {output_dir} {options}"

def collect_esm_scores(esm_preds_dir: str, pdb_dir: str, scorefile: str) -> pd.DataFrame:
    '''
    Collects output scores from ESM.
    '''
    # collect all .json files
    fl = glob(f"{esm_preds_dir}/*/*.json")

    # read the files, add origin column, and concatenate into single DataFrame:
    dl = [pd.read_json(f) for f in fl]
    for d, f in zip(dl, fl):
        d.loc[:, f"{[x for x in esm_preds_dir.split('/') if x][-1]}_path"] = f
    df = pd.concat(dl).reset_index(drop=True)

    # collect pdbs into pdb_dir
    for pdb in (pdbs := glob(f"{esm_preds_dir}/*/*.pdb")):
        shutil.copy(pdb, pdb_dir+"/")

    print(f"Collecting {str(len(pdbs))} predicted pdb-files into {pdb_dir}")

    # store location of .pdb files in output dataframe as "location":
    df.loc[:, "location"] = f"{pdb_dir}/" + df["description"] + ".pdb"
    df.to_json(scorefile)

    return df

def prep_fastas_for_prediction(poses: list[str], fasta_dir: str, max_filenum: int) -> list[str]:
    '''
    Args:
        <poses>             List of paths to *.fa files
        <fasta_dir>         Directory to which the new fastas should be written into
        <max_filenum>          Maximum number of *.fa files that should be written
    '''
    # determine how to split the poses into <max_gpus> fasta files:
    splitnum = len(poses) if len(poses) < max_filenum else max_filenum
    poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]

    # Write fasta files according to the fasta_split determined above and then return:
    return [mergefastas(poses, f"{fasta_dir}/fasta_{str(i+1).zfill(4)}.fa", replace=("/",":")) for i, poses in enumerate(poses_split)]

#----------------------------- OmegaFold Functions ------------------------------------------

def run_OmegaFold(poses: list, of_options: dict, pose_options:list=None, max_gpus=10) -> dict:
    '''
    
    '''
    of_work_dir = of_options.pop("output_dir")
    of_output_dir = of_work_dir + "/of_preds"
    pdb_dir = of_work_dir + "/of_pdbs"
    scorefile = f"{of_work_dir}/of_scores.json"
    return_dict = {"pdb_dir": pdb_dir}

    # check if omegafold output is already present, if so, skip
    if os.path.isfile(scorefile):
        print(f"\nOutputs of prediction run {of_output_dir} already found at {scorefile}\nSkipping prediction step.")
        return_dict["scores"] = pd.read_json(scorefile)
        return return_dict

    # drop 'output_dir' from of_options and create output_dir:
    print(f"\nPredicting {len(poses)} Structures at {of_output_dir}")   
    of_input = of_work_dir + "/of_input/"
    os.makedirs(of_output_dir, exist_ok=True)
    os.makedirs(of_input, exist_ok=True)

    # split poses list into <max_gpus> sublists and write them to files as input to omegafold.
    splitnum = len(poses) if len(poses) < max_gpus else max_gpus
    poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]
    pose_fastas = [mergefastas(poses, f"{of_input}/of_fasta_{str(i+1).zfill(4)}.fa", replace_colon=True) for i, poses in enumerate(poses_split)]

    # write omegafold cmds
    of_cmds = [write_of_cmd(fasta, of_output_dir, of_options) for fasta in pose_fastas]

    # sbatch omegafold
    sbatch_options = [f"--gpus-per-node 1 -c2 -e {of_output_dir}/OF_err.log -o {of_output_dir}/OF_out.log"]
    sbatch_array_jobstarter(cmds=of_cmds, sbatch_options=sbatch_options, jobname="of_predict", 
                            max_array_size=max_gpus, wait=True, remove_cmdfile=False, cmdfile_dir=of_output_dir)

    # collect scores and pdbs
    return_dict["scores"] = collect_omegafold_scores(of_output_dir, pdb_dir=pdb_dir, scorefile=scorefile)

    return return_dict

def mergefastas(files: list, path: str, replace_colon=False, replace=None) -> str: 
    '''
    Merges Fastas located in <files> into one single fasta-file called <path>
    '''
    fastas = list()
    for f in files:
        with open(f, 'r') as f:
            fastas.append(f.read().strip())

    if replace_colon: fastas = [x.replace("/", ":") for x in fastas]
    if replace: fastas = [x.replace(replace[0], replace[1]) for x in fastas]

    with open(path, 'w', encoding="UTF-8") as f:
        f.write("\n".join(fastas))

    return path

def write_of_cmd(fa_file, of_output_dir, options):
    '''
    Writes command for OmegaFold that can be executed by sbatch_array_jobstarter.
    '''
    # create of_output_dir/fa_file
    of_output_dir = f"{of_output_dir}/{fa_file.split('/')[-1].replace('.fa', '')}"
    print(f"Creating OmegaFold output directory at {of_output_dir}")
    os.makedirs(of_output_dir, exist_ok=True)

    options = " ".join([f"--{k} {v}" for k, v in options.items()])
    of_cmd = f"{_python_path} {_of_path} {options} {fa_file} {of_output_dir}"
    return of_cmd

def collect_omegafold_scores(input_dir, pdb_dir="../of_pdbs", scorefile=None):
    '''
    Goes into each prediction subfolder in your omegafold prediction directory, combines the scores and writes them into a single .json scorefile if the <scorefile> option is set.

    Returns the dictionary with all combined scores.
    '''
    # Glob all .json files in input_dir.
    fl = glob(f"{input_dir}/*/*.json")

    # read the files with pandas, add "origin" column!
    dl = [pd.read_json(f) for f in fl]
    for d, f in zip(dl, fl):
        d.loc[:, f"{[x for x in input_dir.split('/') if x][-1]}_path"] = f
    df = pd.concat(dl).reset_index(drop=True)

    # collect pdbs into of_pdbs and store their location into the scorefile:

    predicted_pdbs = glob(f"{input_dir}/*/*.pdb")
    print(f"Collecting {str(len(predicted_pdbs))} predicted pdb-files into {pdb_dir}")
    os.makedirs(pdb_dir, exist_ok=True)
    for pdb in predicted_pdbs:
        shutil.copy(pdb, pdb_dir + "/")

    df.loc[:, "location"] = f"{pdb_dir}/" + df["description"] + ".pdb"

    # if scorefile is set, write scorefile
    if scorefile:
        print(f"Writing {len(df)} scores into {scorefile}.")
        df.to_json(scorefile)

    return df

## ------------------------- Metrics ----------------------------------------

def calc_bb_rmsd(poses: list, bb_rmsd_args: dict) -> pd.DataFrame:
    '''
    Calculates bb_ca_rmsd of all poses to ref_poses in <bb_rmsd_args>
    
    <bb_rmsd_args>:         Dictionary with options for bb_rmsd function. Has to contain:
                                "ref_pdb": Path to either a directory containing pdb files or a single pdb file that should be taken as reference for RMSD calculation. Will be matched by description.
                            Optional:
                                "rmsd_scorename": Name of the score you want to add. Default: bb_ca_rmsd
                                "remove_layers"
    '''
    if not bb_rmsd_args["rmsd_scorename"]: bb_rmsd_args["rmsd_scorename"] = "bb_ca_rmsd"
    if not bb_rmsd_args["remove_layers"]: bb_rmsd_args["remove_layers"] = 0

    # setup rmsd_dict
    rmsd_dict = {"description": [], bb_rmsd_args["rmsd_scorename"]: []}

    # This only works, if poses and ref_pdbs are already an aligned list. which they are not!!!
    if bb_rmsd_args["ref_pdb"].endswith(".pdb"):
        ref_pdbs = [bb_rmsd_args["ref_pdb"]]
    for pose, ref_pdb in zip(poses, ref_pdbs):
        pose_description = pose.split("/")[-1].replace(".pdb", "", regex=False)
        ref_pose = [x for x in bb_rmsd_args["ref_pdbs"] if x.replace(".pdb", "", regex=True).endswith(pose_description)][0]
        rmsd_dict["description"].append(pose)
        rmsd_dict[bb_rmsd_args["rmsd_scorename"]].append(bb_rmsd.superimpose_calc_rmsd(ref_pose, pose, atoms=["CA"]))

    scores_df = pd.DataFrame(rmsd_dict)
    return scores_df

# -------------------------- Misc -------------------------------------------

def update_df(new_df: pd.DataFrame, new_df_col: str, old_df: pd.DataFrame, old_df_col: str, new_df_col_remove_layer=0, sep="_") -> pd.DataFrame:
    '''
    Fill all rows of <new_df> where [new_df_col == old_df_col] with values from <old_df>
    
    Args:
        <new_df>                      New DataFrame
        <new_df_col>                  Column in the new DataFrame that contains the 'index' for merging scores from the old DataFrame
        <old_df>                      Old DataFrame
        <old_df_col>                  Column in the old DataFrame that conatins the 'index' for copying its scores into new DataFrame
        <new_df_col_remove_layer>     How many index layers need to be removed from every item in <new_df_col> to reach the name in <old_df_col>
        <sep>                         Index separator, Default="_"
    
    Returns:
        pd.DataFrame where scores from old_df were copied into new_df.
    '''
    startlen = len(new_df)
    # Remove layers if option is set
    if new_df_col_remove_layer: new_df["select_col"] = new_df[new_df_col].str.split(sep).str[:-1*new_df_col_remove_layer].str.join(sep)
    else: new_df["select_col"] = new_df[new_df_col]

    new_df = new_df.merge(old_df, left_on='select_col', right_on=old_df_col)
    new_df.drop(columns="select_col", inplace=True)

    if len(new_df) == 0: raise ValueError(f"ERROR: Merging DataFrames failed. This means there was no overlap found between old_df[old_df_col] and new_df[new_df_col]")
    if len(new_df) < startlen: raise ValueError(f"ERROR: Merging DataFrames failed. Some rows in new_df[new_df_col] were not found in old_df[old_df_col]")

    return new_df

def search_for_path(pathlist: list, glob_pattern: str) -> str:
    '''
    Checks if file is in any path given in <pathlist> and returns the first instance it finds.
    '''
    # glob through all paths in pathlist and look if there is a file that matches regex
    pl = [glob(f"{path}/{glob_pattern}") for path in pathlist]
    return pl

def find_rosetta_path(pathlist: list, executable: str, stringent=False) -> str:
    '''
    Function designated to find paths to Rosetta executables regardless of static or dynamic compilation (static- or default.linuxgccrelease)
    
    Example: relax_path = find_rosetta_path('/home/florian_wieser/Rosetta/', 'relax')
    Returns: relax_path = '/home/florian_wieser/Rosetta/main/source/bin/relax.default.linuxgccrelease'
    '''
    # find all relax executables in all paths in <pathlist> and put them in onedimensional list:
    if stringent:
        searchpaths = [f"{path}/main/source/bin/{executable}" for path in pathlist]
    else:
        searchpaths = [f"{path}/main/source/bin/{executable}.*" for path in pathlist]
    pl = sum([glob(path) for path in searchpaths], [])
    if not pl: raise FileNotFoundError(f'ERROR: no Rosetta executables were found in pathlist {", ".join(searchpaths)}')

    # pick either default, or static version of executable, depending on which is available.
    for path in pl:
        if 'default' in path:
            print(f"Rosetta executable {path} will be used for relax.")
            return path
        elif 'static' in path:
            print(f"Rosetta executable {path} will be used for relax.")
            return path
    raise FileNotFoundError(f"ERROR: No usable Rosetta executable was found in any of the specified paths. check Rosetta path variable _rosetta_paths")

def collect_scorefiles_from_dir(path_to_dir: str, file_extension='sc', index=None, silent=None) -> pd.DataFrame:
    '''
    Reads all files that end with extension <file_extension> and collects them into a singular dataframe.
    '''
    # compile regex pattern:
    regex_pattern = '^.*\.' + file_extension + '$'

    # Collect Dataframes from files that match <regex> into List.
    df_list = list()
    for i in os.listdir(path_to_dir):
        if re.match(regex_pattern, i):
            filename = path_to_dir + i
            if not silent:
                print(f"Scores collected from file: {filename}")
            if file_extension == 'sc':
                re_df = pd.read_csv(filename, delim_whitespace=True, header=[1], na_filter=True)
            elif file_extension == 'json':
                re_df = pd.read_json(filename)
            else:
                raise TypeError(f"ERROR: Unsupported file extension for 'collect_scores_from_dir': {file_extension}\nPick different file-extension!")
            re_df["scorefile_path"] = filename
            df_list.append(re_df)

    # Merge List and return
    merged_df = pd.concat(df_list).reset_index(drop=True)
    if index: merged_df.set_index(index)

    return merged_df

def filter_dataframe(df: pd.DataFrame, col: str, n, remove_layers=None, layer_col="description", sep="_", ascending=True) -> pd.DataFrame:
    '''
    remove_layers option allows to filter dataframe based on groupings after removing index layers.
    If the option remove_layers is set (has to be type: int), then n determines how many c
    '''
    # make sure <col> and <layer_col> (if remove_layers is set) are existing columns in <df>

    # if remove_layers is set, compile list of unique pose descriptions after removing one index layer:
    if remove_layers:
        # TODO: make sure that df[layer_col] returns a Series and not a DataFrame (layer_col must be a unique column)
        if type(remove_layers) != int: raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
        unique_list = list(df[layer_col].str.split(sep).str[:-1*int(remove_layers)].str.join(sep).unique())

        # go through each unique name in unique list, select the matching rows in df
        unique_dfs = [select_rows_by_regex_in_column_x(df, layer_col, unique_name) for unique_name in unique_list]

        # filter them down to the number specified by n and concatenate back into one df
        filtered_df = pd.concat([unique_df.sort_values(by=col, ascending=ascending).head(determine_filter_n(unique_df, n)) for unique_df in unique_dfs]).reset_index(drop=True)

    else:
        filtered_df = df.sort_values(by=col, ascending=ascending).head(determine_filter_n(df, n))

    return filtered_df

def select_rows_by_regex_in_column_x(df, col, regex_pattern):
    '''    
    Selects all rows of <dataframe> in which column <column_name> contains the <regex_pattern>.
    
    Returns: pd.DataFrame
    '''
    if col not in df.columns:
        raise KeyError(f'Scoreterm {col} not found in this Score_Table. Available scoreterms:\n{[x for x in df.columns]}')

    return_df = df[df[col].str.contains(regex_pattern, regex=True, na=False)]
    if return_df.empty:
        raise KeyError(f"Your DataFrame is Empty, regex '{regex_pattern}' was not found in specified column: '{col}'. Change Column, or Regex")
    return return_df

def determine_filter_n(df: pd.DataFrame, n: float) -> int:
    '''
    
    '''
    filter_n = float(n)
    if filter_n < 1:
        filter_n = round(len(df) * filter_n)
    elif filter_n <= 0:
        raise ValueError(f"ERROR: Argument <n> of filter functions cannot be smaller than 0. It has to be positive number. If n < 1, the top n fraction is taken from the DataFrame. if n > 1, the top n rows are taken from the DataFrame")

    return int(filter_n)

def parse_options_string(options_string: str, sep='-', flags=False) -> dict:
        '''
        Parses an options string ('--option_A var_A --option_B var_B') into a dictionary:
        {"option_A": "var_A", "option_B": "var_B"}
        
        WARNING: if options contain dash '-' for example in colabofold_batch (--msa-mode --stop-at-score etc.), then you should provide
                 the <sep> argument. In this cas it would be: sep="--". Otherwise the function will split at all '-' characters and the
                 AF2 options will not be parsed correctly.
        
        Note: If no separator is given, the function also works if optio/ns start with single dash: -option_A var_A -option_B var_B
        
        Passing the <sep> argument is recommended.
        
        Returns dictionary with {option: value}
        '''
        # if options_string was passed with None, return empty dictionary:
        if options_string is None:
            return {}

        # if options_string is a string, split around (sep) and parse options into a list. Convert all first occurances of "=" of each option into " ".
        l = [x.replace("=", " ", 1).strip().split(" ", 1) for x in options_string.split(sep) if x]

        return {k: v for k, v in l}

def parse_options_flags(options: str, sep="-"):
    '''
    Parses an options string with flags ('--option_A var_A --option_B var_B --dump_all --calc_rmsd') into a dictionary and a list:
    {"option_A": "var_A", "option_B": "var_B"}, ["dump_all", "calc_rmsd"]

    WARNING: if options contain dash '-' for example in colabofold_batch (--msa-mode --stop-at-score etc.), then you should provide
             the <sep> argument. In this cas it would be: sep="--". Otherwise the function will split at all '-' characters and the
             AF2 options will not be parsed correctly.

    Note: If no separator is given, the function also works if options start with single dash: -option_A var_A -option_B var_B

    Passing the <sep> argument is recommended.

    Returns dictionary and a list with {option: value}, [flags]
    '''
    # sanity check:
    if "=" in sep: raise ValueError(f"ERROR: Symbol '=' cannot be used in argument sep! sep = '{sep}'")
    if type(options) != str: raise ValueError(f"ERROR: incorrect type of argument <options>. Only str allowed. Type = {type(options)}")

    # return empty dictionary and list for options and flags if no options were passed.
    if options is None:
        return {}, []

    # run first split across separator and replace any first occurences of '=' in the options string with ' '
    firstsplit = [x.replace("=", " ", 1).strip() for x in options.split(sep) if x]

    # split options list (firstsplit) into options and flags based on if they are separatable by " " (whitespace):
    opts = dict()
    flags = list()
    for item in firstsplit:
        if len((x := item.split(" "))) > 1:
            opts[x[0]] = " ".join(x[1:])
        else:
            flags.append(x[0])

    return opts, flags

def make_tied_chains_list(pose: str, chains: list) -> list:
    '''
    Creates a tied_chains_list for ProteinMPNN for a given input <pose> for chains listed in <chains>.
    Args:
        <pose>              Should be path to a .pdb file.
        <chains>            Should be a list of chain Letters (e.g. ["A", "B"]) indicating which chains should be tied.
                            All chains to be tied should have the same length!
        
    Returns:
        List with dictionaries of the format: [{"A": [1], "B": [1]}, ..., {"A": [n], "B": [n]}]
    '''
    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structure
    structure = pdb_parser.get_structure("pose", pose)
    model = structure[0]

    # check if specified <chains> are in model
    model_chains = [chain.id for chain in model.get_chains()]
    for chain in chains:
        if chain not in model_chains:
            raise ValueError(f"ERROR: Chain {chain} not found in {pose}\nPossible chains: {', '.join(model_chains)}")

    # TODO implement check if all chains are of the same length!

    # iterate through residues of first chain and write a list out of it:
    tied_chains_list = list()
    for resi in model[chains[0]].get_residues():
        tied_chains_list.append({chain: [resi.get_id()[1]] for chain in chains})

    return tied_chains_list

def pose_extract_chain(path_to_pose: str, chain: str, new_path=None) -> str:
    '''
        Extracts Chain <chain> from pose at <path_to_pose>.
        Returns path to new pose, that only contains chain <chain>.
        
        Args:
            <path_to_pose>            Path to your pdb-file
            <chain>                   Letter identifier of your chain (E.g. chain="A")
            <new_path>                New path where extracted chain should be saved into. By Default it will be "<path_to_pose>_chain_<chain>.pdb"
        
        Returns:
            Path where the new pose is stored.
        
    '''
    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structure
    pose = pdb_parser.get_structure("pose", path_to_pose)

    # Select chain
    if chain not in [chain.id for chain in pose[0].get_chains()]:
        raise ValueError(f"ERROR: Chain {chain} not found in {path_to_pose}. Are you sure your inputs are correct?")
    extract_chain = pose[0][chain]

    # create path to save the pose to
    new_path_to_pose = new_path or f"{path_to_pose.replace('.pdb', '')}_chain_{chain}.pdb"

    # save the pose
    io = PDBIO()
    io.set_structure(extract_chain)
    io.save(new_path_to_pose)

    return new_path_to_pose

def read_fasta_sequence(file: str) -> str:
    '''
    Reads fasta-formatted file and returns sequence.
    '''
    with open(file, 'r') as f:
        x = "".join(f.readlines()[1:])
    return x

def read_pdb_sequence(file: str, chain_sep="/") -> str:
    '''
    Reads .fasta sequence of a protein from a .pdb file
    '''
    # sanity check
    if not file.endswith(".pdb"): print(f"WARNING: Input file does not end with .pdb Are you sure the file-type is correct?\nFile: {file}")

    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    ppb = Bio.PDB.PPBuilder()

    # Get the structure
    pose = pdb_parser.get_structure(file, file)

    # collect the sequence
    sequence = chain_sep.join([str(x.get_sequence()) for x in ppb.build_peptides(pose)])

    return sequence

def get_list_items_by_index(input_list: list, indeces: list[int]) -> list:
    '''
    Uses Pandas to access items of a list by their index and return them as a list.
    '''
    return list(pd.Series(input_list)[indeces])

def calc_aliphatic_content_poses(poses: list, residues=None) -> pd.DataFrame:
    '''
    Calculates aliphatic content of poses (for a specified set of <residues> if specified).
    Args:
        <poses>             List of paths to .pdb files
        <residues>          list: Has to be a list with indeces (starting with 1, so Rosetta Numbering or PDB numbering (without chains)) of residues for which to calculate aliphatic content 
        
    Returns:
        DataFrame containing the aliphatic content score for each pose.
    '''
    descriptions = [x.split("/")[-1].split(".")[0] for x in poses]

    if all([x.endswith(".fa") for x in poses]):
        # read sequences
        seqs = [read_fasta_sequence(file) for file in poses]
    elif all([x.endswith(".pdb") for x in poses]):
        # convert *.pdb files into sequences
        seqs = [read_pdb_sequence(file) for file in poses]
    else:
        raise ValueError(f"ERROR: Your list of poses contains inconsistent file types. All files must be either *.pdb or *.fa files. \nPoses: {poses}")

    # if residues is set, extract specified residues from sequences:
    if residues:
        seqs = ["".join(list(pd.Series(list(seq))[[int(x)-1 for x in residues]])) for seq in seqs]

    aliphatic_score_l = [calc_aliphatic_content(seq) for seq in seqs]

    return pd.DataFrame({"description": descriptions, "aliphatic_score": aliphatic_score_l})

def calc_aliphatic_content(seq: str) -> float:
    '''
    Calculates aliphatic score for a sequence.
    Aliphatic score is defined as the percentage of amino acids in the total sequence being any one of [alanine, glycine, isoleucine, leucine, proline, phenylalanine, and valine]
    '''
    num = sum(map(seq.upper().count, ['A', 'G', "I", "L", "P", "V", "F"]))
    aliphatic_score = num / len(seq)
    return aliphatic_score

def calc_aliphatic_content_motif(seq: str, motif_idx:list):
    ''''''
    motif_idx = [int(x)-1 for x in motif_idx]
    seq = "".join([seq[i] for i in motif_idx])
    num = sum(map(seq.upper().count, ['A', 'G', "I", "L", "P", "V", "F"]))
    aliphatic_score = num / len(seq)
    return aliphatic_score

def parse_motif(poses_df: pd.DataFrame, motif) -> list:
    '''
    '''
    if type(motif) == str:
        return list(poses_df[motif])
    elif type(motif) == dict:
        return [motif for x in list(poses_df["poses"])]
    elif type(motif) == list:
        return motif
    else:
        raise TypeError(f"ERROR: unsupported datatype for Argument 'motif': {type(motif)}. \n motif: {motif}")

def col_with_prefix_exists_in_df(df: pd.DataFrame, prefix: str) -> bool:
    '''
    Checks if there is any column in <df> that starts with <prefix>.
    Parameters:
        df (pd.DataFrame): DataFrame to check
        prefix (str): Prefix for which you want to check if it exists in the DF
    
    Returns: (bool)
    '''
    return any([x.startswith(prefix) for x in df.columns])

def reassign_motif(motif_res: dict, ref_pdb_idx: list, hal_pdb_idx: list) -> dict:
    '''AAA'''
    # expand motif into list of tuples for dict lookup:
    motif_expanded_nested_list = [[(key, val) for val in values] for key, values in motif_res.items()]
    motif_expanded_list = list(itertools.chain.from_iterable(motif_expanded_nested_list))

    # convert motif to new mapping:
    exchange_dict = {tuple(ref_idx): hal_idx for ref_idx, hal_idx in zip(ref_pdb_idx, hal_pdb_idx)}
    reassigned_motif_list = [exchange_dict[idx] for idx in motif_expanded_list]

    # convert motif_list into dict:
    reassigned_motif = defaultdict(list)
    for idx in reassigned_motif_list:
        reassigned_motif[idx[0]].append(idx[1])

    return dict(reassigned_motif)

def reassign_identity_keys(identity_dict: dict, ref_pdb_idx: list, hal_pdb_idx: list) -> dict:
    '''AAA'''
    # transform idx into pdb_numbering:
    ref_keys = [(x+str(y)) for x, y in ref_pdb_idx]
    inp_keys = [(x+str(y)) for x, y in hal_pdb_idx]

    # collect mapping_dict
    mapping_dict = {ref: inp for ref, inp in zip(ref_keys, inp_keys)}
    return {mapping_dict[pdb_idx]: res_name for pdb_idx, res_name in identity_dict.items()}

def parse_pose_options(input_arg, df: pd.DataFrame) -> list:
    '''AAA'''
    def string_return(input_string, df: pd.DataFrame) -> list:
        if input_string in df.columns: return df[input_string].to_list()
        else: raise KeyError(f"Column {input_string} not found in poses_df (poses DataFrame). Available columns: {df.columns}")
    def dict_return(input_dict, df: pd.DataFrame) -> list:
        return [input_dict for x in range(df.shape[0])]
    def list_return(input_list, df: pd.DataFrame) -> list:
        if len(input_list) == len(df): return input_list
        else: raise ValueError(f":input_list: has to be of the same length as the poses_df. length :input_list: {len(input_list)}\nlength poses_df: {len(df)}")
    typedict = {str: string_return, dict: dict_return, list: list_return}

    return typedict[type(input_arg)](input_arg, df)
