#!/home/mabr3112/anaconda3/bin/python3.9

import os
import argparse
import subprocess
import math
import time
import pandas as pd
import shutil
import Bio
from glob import glob
from subprocess import run
import json
import copy
from Bio.PDB import *

import sys
#sys.path += ["/home/mabr3112/projects/iterative_refinement/"]
#sys.path += ["/home/tripp/riffdiff2/riff_diff/it_test/"]
sys.path.append("/home/mabr3112/riff_diff")

import utils.adrian_utils as my_utils
import rfdiffusion_and_refinement as diffrf
# import custom modules
from iterative_refinement import *
import utils.plotting as plots
import utils.biopython_tools
import utils.pymol_tools
from utils.plotting import PlottingTrajectory
import utils.metrics as metrics
import superimposition_tools as si_tools



def import_json_to_dict(jsonpath: str):
    with open(jsonpath) as jsonfile:
        data = json.load(jsonfile)
    return(data)


def chainresdict_to_str(in_dict:dict):
    cat_res = []
    for chain in in_dict:
        for residue in in_dict[chain]:
            cat_res.append(f'{str(residue)+chain}')
    return ",".join(cat_res)

def calculate_site_score(pose, motif, pose_catres, motif_catres, pose_residue_plddt):
    pose = my_utils.import_structure_from_pdb(pose)
    motif = my_utils.import_structure_from_pdb(motif)

    pose_cat_atoms = extract_residues_from_structure(pose, pose_catres)
    motif_cat_atoms = extract_residues_from_structure(motif, motif_catres)

    Superimposer().set_atoms(pose_cat_atoms, motif_cat_atoms)
    Superimposer().rotran
    catres_rmsd = round(Superimposer().rms, 2)


def extract_residue_atoms_from_structure(structure, chaindict):
    res = []
    for chain in chaindict:
        for resnum in chaindict[chain]:
            res.append(structure[0][chain][resnum])
    atoms = []
    for residue in res:
        for atom in residue.get_atoms():
            if atom.element in ['C', 'N', 'O', 'S']:
                atoms.append(atom)
    return atoms





def main(args):

    #absolute path for xml is necessary because coupled moves output can only be controlled by cd'ing into output directory before starting the run
    xml = os.path.abspath(args.protocol)

    df = pd.read_json(args.json)
    #df = df[df['poses'].str.contains('A12-D7-C34-B14')]
    #TODO: add a filter for input pdbs here
    #if args.filter_input:
    analysis = Poses(args.output_dir, df['poses'].to_list())


    input_dir = f'{args.output_dir}/input_pdbs/'
    os.makedirs(input_dir, exist_ok=True)

    #set maximum number of cpus for coupled moves run
    analysis.max_rosetta_cpus = args.max_cpus
    #merge pose dataframes
    analysis.poses_df = analysis.poses_df.drop(['input_poses', 'poses'], axis=1).merge(df, on='poses_description', how='left')

    analysis.poses_df = analysis.poses_df[analysis.poses_df["post_cm_ligand_clash"] == False]
    print(f'{len(analysis.poses_df.index)} input poses passed ligand clash filter.')
    analysis.poses_df = analysis.poses_df[analysis.poses_df['post_cm_site_score'] >= args.sitescore_cutoff]
    print(f'{len(analysis.poses_df.index)} input poses passed site score filter.')

    options_list = []
    pose_list = []
    #iterate over dataframe, create coupled moves options
    for index, row in analysis.poses_df.iterrows():
        #set filename for input pdbs
        pose_path = os.path.abspath(input_dir + row['poses_description'] + '.pdb')
        #add ligand to input pdbs
        #si_tools.superimpose_add_chain_by_motif(row['updated_reference_frags_location'], row['poses'], args.ligand_chain, row['fixed_residues'], row['fixed_residues'], pose_path, ['CA', 'C', 'O'])
        #identify catalytic residues, format them for rosettascripts input
        cat_res = chainresdict_to_str(row['fixed_residues'])
        motif_res = chainresdict_to_str(row['motif_residues'])
        options = f"-parser:script_vars cat_res={cat_res} motif_res={motif_res} -in:file:native {row['updated_reference_frags_location']}"
        options_list.append(options)
        pose_list.append(pose_path)

    #update path to input pdbs, add analysis options
    analysis.poses_df['analysis_options'] = options_list
    analysis.poses_df['poses'] = pose_list

    #create working directory for coupled moves
    working_dir = f'{args.output_dir}/cm_working_dir/'
    os.makedirs(working_dir, exist_ok=True)

    opts = f"-parser:protocol {xml} -out:path:all {working_dir}"
    if args.options:
        opts = opts + ' ' + args.options


    #run coupled moves
    analysis.rosetta("rosetta_scripts.default.linuxgccrelease", options=opts, pose_options=analysis.poses_df['analysis_options'].to_list(), n=args.nstruct, prefix='analysis')

    #create results dir for coupled moves
    resultsdir = f'{args.output_dir}/results/'
    os.makedirs(resultsdir, exist_ok=True)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--protocol", type=str, required=True, help="path to xmlfile that should be used for analysis")
    argparser.add_argument("--output_dir", type=str, required=True, help="working directory")
    argparser.add_argument("--max_cpus", type=int, default=320, help="maximum number of cpus for analysis")
    argparser.add_argument("--nstruct", type=int, default=50, help="analysis runs per input pdb")
    argparser.add_argument("--options", type=str, default=None, help="additional coupled moves cmd-line arguments in string format, e.g. '-extra_res_fa /path/to/params.pa' for adding a .params file. Use absolute paths!")
    argparser.add_argument("--json", type=str, required=True, help="path to jsonfile containing information about input pdbs (catres etc)")
    argparser.add_argument("--ligand_chain", type=str, default='Z', help="should always be Z in current workflow")
    argparser.add_argument("--sitescore_cutoff", type=float, default=None, help="cutoff for site score (1/e**(motif_bb_rmsd) * motif_plddt / 100) for filtering input pdbs. Recommended values are 0.3 to 0.5")
    #argparser.add_argument("--max_input_per_backbone", type=int, default=5, help="maximum number of input pdbs coming from identical rfdiffusion backbones")

    args = argparser.parse_args()

    main(args)
