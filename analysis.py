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
import matplotlib.pyplot as plt
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

def superimpose_calc_ligand_rmsd(ref_pdbfile: str, target_pdbfile: str, ref_selection: dict, target_selection: dict, target_ligand_chain: str, ref_ligand_chain) -> float:
    '''
    Superimpose two structures based on residue selections via backbone. Sensitive to the order in the selection! 
    Returns RMSD of ligand after superimposition. Useful to check how much the ligand has moved during relax runs.
    Args:
        <ref_pdbfile>                     Path to the reference pdb-file
        <target_pdbfile>                  Path to the target pdb-file (for which rmsd should be calculated)
        <ref_selection>                   Dictionary containing the residue positions for which RMSD should be calculated.
                                          !!! IMPORTANT: Order of amino acids is relevant to RMSD calculation! (for alignment)
                                          E.g. {"A": [1, 3, 4, 12], "B": [1, 15, 4]}
        <target_selection>                Dictionary containing the residue positions for with RMSD should be calculated.
                                          Example (see <ref_selection>)
        <target_ligand_chain>             Chain that contains the ligand in the target pdbfile. 
        <ref_ligand_chain>                Chain that contains the ligand in the reference pdbfile. 
    Returns:
        RMSD between ref_selection and target_selection of pdbfiles. Returns float.
    '''

    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_model = my_utils.import_structure_from_pdb(ref_pdbfile)[0]
    target_model = my_utils.import_structure_from_pdb(target_pdbfile)[0]

    # Extract ligands from models
    ref_ligand = ref_model[ref_ligand_chain]
    target_ligand = target_model[target_ligand_chain]

    # Extract ligand atoms
    ref_lig_atoms = [atom for atom in ref_ligand.get_atoms()]
    target_lig_atoms = [atom for atom in ref_ligand.get_atoms()]
    
    # Make a list of the atoms (in the structures) you wish to align.
    ref_atoms = []
    target_atoms = []

    bb_atoms = ['N', 'CA', 'C', 'O']
    
    for chain in ref_selection:
        for res in ref_selection[chain]:
            [ref_atoms.append(ref_model[chain][(' ', res, ' ')][x]) for x in bb_atoms]

    for chain in target_selection:
        for res in target_selection[chain]:
            [target_atoms.append(target_model[chain][(' ', res, ' ')][x]) for x in bb_atoms]
        
    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    super_imposer.rotran
    super_imposer.apply(target_ligand)

    # Calculate RMSD:
    rmsd = calc_rmsd_without_superposition(ref_ligand, target_ligand)
    return rmsd

def calc_rmsd_without_superposition(entity1, entity2, rmsd_type:str='heavy'):
    if not rmsd_type in ['bb', 'CA', 'heavy', 'all']:
        raise KeyError(f"rmsd_type must be one of 'bb', 'CA', 'heavy', 'all', not {rmsd_type}!")

    if rmsd_type == "bb":
        atoms = ['N', 'CA', 'C', 'O']
    if rmsd_type == "CA":
        atoms = ['CA']
    
    ent1_atoms = [atom for atom in entity1.get_atoms()]
    ent2_atoms = [atom for atom in entity2.get_atoms()]

    if rmsd_type == 'heavy':
        ent1_atoms = [atom for atom in ent1_atoms if not atom.element == 'H']
        ent2_atoms = [atom for atom in ent2_atoms if not atom.element == 'H']
    elif rmsd_type == 'bb' or rmsd_type == 'CA':
        ent1_atoms = [atom for atom in ent1_atoms if atom.name in atoms]
        ent2_atoms = [atom for atom in ent2_atoms if atom.name in atoms]

    distances = [atm1 - atm2 for atm1, atm2 in zip(ent1_atoms, ent2_atoms)]
    rmsd = math.sqrt(sum([dist ** 2 for dist in distances]) / len(ent1_atoms))

    return round(rmsd, 2)

def scatter_plot(x_values, y_values, x_label, y_label, file_path=None, x_upper=None, x_lower=None, y_upper=None, y_lower=None):
    plt.figure()
    plt.scatter(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not x_upper == None:
        plt.xlim(right=x_upper)
    if not x_lower == None:
        plt.xlim(left=x_lower)
    if not y_upper == None:
        plt.ylim(top=y_upper)
    if not y_lower == None:
        plt.ylim(bottom=y_lower)
    plt.text(0.5, -0.1, f"Number of datapoints: {len(x_values)}", ha="center", transform=plt.gca().transAxes)
    plt.savefig(file_path)



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
        si_tools.superimpose_add_chain_by_motif(row['updated_reference_frags_location'], row['poses'], args.ligand_chain, row['fixed_residues'], row['fixed_residues'], pose_path, ['CA', 'C', 'O'])
        #identify catalytic residues, format them for rosettascripts input
        cat_res = chainresdict_to_str(row['fixed_residues'])
        motif_res = chainresdict_to_str(row['motif_residues'])
        options = f"-parser:script_vars cat_res={cat_res} motif_res={motif_res} -in:file:native {row['updated_reference_frags_location']}"
        options_list.append(options)
        pose_list.append(pose_path)

    #update path to input pdbs, add analysis options
    analysis.poses_df['analysis_options'] = options_list
    analysis.poses_df['poses'] = pose_list
    #add LINK records if covalent bonds are present
    print(analysis.poses_df.columns)
    if 'covalent_bonds' in analysis.poses_df.columns:
        if not (analysis.poses_df['covalent_bonds'].str.strip() == "").any():
            print('Covalent bonds present! Adding LINK records to poses...')
            analysis.add_LINK_to_poses('covalent_bonds', 'analysis')


    opts = f"-parser:protocol {xml}"
    if args.options:
        opts = opts + ' ' + args.options

    #save original dataframe
    analysis.poses_df['analysis_input_poses'] = analysis.poses_df['poses']
    analysis.poses_df['analysis_input_poses_description'] = analysis.poses_df['poses_description']
    old_poses = copy.deepcopy(analysis)

    #run analysis
    analysis.rosetta("rosetta_scripts.default.linuxgccrelease", options=opts, pose_options=analysis.poses_df['analysis_options'].to_list(), n=args.nstruct, prefix='analysis')
    #calculate rmsds of ligand and catalytic residues post-relax
    analysis.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', 'analysis_catres')
    analysis.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', 'analysis_catres_bb', ['N', 'CA', 'C', 'O'])
    analysis.calc_bb_rmsd_df('analysis_input_poses', 'analysis')
    lig_rmsd_list = []
    for index, row in analysis.poses_df.iterrows():
        lig_rmsd = superimpose_calc_ligand_rmsd(row['updated_reference_frags_location'], row['poses'], row['fixed_residues'], row['fixed_residues'], args.ligand_chain, args.ligand_chain)
        lig_rmsd_list.append(lig_rmsd)
    analysis.poses_df['analysis_ligand_rmsd'] = lig_rmsd_list


    row_list = []
    for input, df in analysis.poses_df.groupby('analysis_input_poses_description', sort=False):
        analysis_dict = {}
        for i in ['poses', 'analysis_input_poses', 'analysis_ligand_rmsd', 'analysis_catres_motif_heavy_rmsd', 'analysis_catres_bb_motif_rmsd', 'analysis_bb_ca_rmsd', 'analysis_total_score', 'analysis_rotprob', 'analysis_lig_shape_comp', 'analysis_lig_delta_sasa', 'analysis_interaction_energy', 'analysis_sasa', 'analysis_sap_score']:
            analysis_dict[i] = df[i].to_list()
        analysis_dict['analysis_rotprob'] = [abs(rot_prob) for rot_prob in analysis_dict['analysis_rotprob']]
        row = pd.Series(analysis_dict)
        row.rename({'poses': 'analysis_relaxed_poses'}, inplace=True)
        row['poses_description'] = input
        row_list.append(row)
    
    #create new dataframe and merge with old one
    new_df = pd.DataFrame(row_list)
    analysis.poses_df = new_df.merge(old_poses.poses_df, on='poses_description')



    #TODO: filter dataframe --> only output best structures, otherwise it will get overwhelming
    #create plots
    plot_path = os.path.join(args.output_dir, 'plots/')
    for index, row in analysis.poses_df.iterrows():
        scatter_lig_rmsd = scatter_plot(row['analysis_ligand_rmsd'], row['analysis_total_score'], 'RMSD [A]', 'total score [REU]', f"{plot_path}{row['poses_description']}_ligrmsd_totalscore.png", max(row['analysis_ligand_rmsd']), 0, max(row['analysis_total_score']), None)
        scatter_catres_rmsd = scatter_plot(row['analysis_catres_motif_heavy_rmsd'], row['analysis_total_score'], 'RMSD [A]', 'total score [REU]', f"{plot_path}{row['poses_description']}_catressc_totalscore.png", max(row['analysis_catres_motif_heavy_rmsd']), 0, max(row['analysis_total_score']), None)




if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--protocol", type=str, default='rosetta/analysis.xml', help="path to xmlfile that should be used for analysis")
    argparser.add_argument("--output_dir", type=str, required=True, help="working directory")
    argparser.add_argument("--max_cpus", type=int, default=320, help="maximum number of cpus for analysis")
    argparser.add_argument("--nstruct", type=int, default=50, help="analysis runs per input pdb")
    argparser.add_argument("--options", type=str, default=None, help="additional rosetta scripts cmd-line arguments in string format, e.g. '-extra_res_fa /path/to/params.pa' for adding a .params file. Use absolute paths!")
    argparser.add_argument("--json", type=str, required=True, help="path to jsonfile containing information about input pdbs (catres etc)")
    argparser.add_argument("--ligand_chain", type=str, default='Z', help="should always be Z in current workflow")
    argparser.add_argument("--sitescore_cutoff", type=float, default=None, help="cutoff for site score (1/e**(motif_bb_rmsd) * motif_plddt / 100) for filtering input pdbs. Recommended values are 0.3 to 0.5")
    #argparser.add_argument("--max_input_per_backbone", type=int, default=5, help="maximum number of input pdbs coming from identical rfdiffusion backbones")

    args = argparser.parse_args()

    main(args)
