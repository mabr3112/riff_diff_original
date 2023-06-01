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

def aa_three_or_one_letter_code(AA):
    letter_codes = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL"
    }

    return(letter_codes[AA])


def get_protein_sequence(pdb_file):
  # Create a parser object
  parser = Bio.PDB.PDBParser()
  # Parse the pdb file and get the structure object
  structure = parser.get_structure("protein", pdb_file)
  # Create an empty list to store the sequence
  sequence = []
  # Loop over all the residues in the first model and first chain of the structure
  for residue in structure.get_residues():
    # Check if the residue is standard amino acid
    if Bio.PDB.is_aa(residue):
      # Get the one-letter code of the residue and append it to the sequence list
      sequence.append(aa_three_or_one_letter_code(residue.get_resname()))
  # Join the sequence list into a string and return it
  return "".join(sequence)

def create_output_dir_change_filename(output_dir, filename):
    if output_dir:
        output_dir = my_utils.path_ends_with_slash(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        filename = output_dir + filename
    else:
        filename = filename
    return filename


def groups_from_unique_input(scoretable, label):
    all_elements = scoretable[label].values.tolist()
    unique = set(all_elements)
    unique = sorted(list(unique))
    return unique

def extract_designpositions_from_resfiles(resfiles):
    design_positions = []
    for file in resfiles:
        with open(file, "r") as r:
            for line in r:
                if not line.startswith("NAT") and not line.startswith("start"):
                    design_positions.append(int(line.split()[0]))
    #filter for unique elements in list
    design_positions = sorted(list(set(design_positions)))
    return(design_positions)

def statsfile_to_df(statsfile: str):
    #reads in the .stats file output from a coupled-moves run and converts it to a dataframe
    df = pd.read_csv(statsfile, sep=None, engine='python', header=None, keep_default_na=False)
    df_scores = df[4].str.split(expand=True)
    columnheaders = df_scores[df_scores.columns[0::2]]
    columnheaders = columnheaders.loc[0, :].values.tolist()
    columnheaders = [i.replace(':','') for i in columnheaders]
    df_scores = df_scores[df_scores.columns[1::2]]
    df_scores.columns = columnheaders
    df_scores = df_scores.astype(float)
    df_scores["total_score"] = df_scores.sum(axis=1)
    df_scores["sequence"] = df[3]
    return(df_scores)

def statsfiles_to_json(input_dir: str, description:str, filename):
    #gathers all coupled-moves statsfiles and converts to a single dictionary
    statsfiles = []
    resfiles = []
    for file in os.listdir(input_dir):
        if file.endswith(".stats") and file[6:-11] == description:
            statsfiles.append(input_dir + file)
        elif file.endswith(".resfile") and file[6:-13] == description:
            resfiles.append(input_dir + file)

    df = pd.DataFrame()
    for stats in statsfiles:
        statsdf = statsfile_to_df(stats)
        df = pd.concat([df, statsdf])

    #subtract res_type_constraint (from favoring native sequence) from total score to get 'cleaner' score
    df['total_score'] = df['total_score'] - df['res_type_constraint']
    design_positions = extract_designpositions_from_resfiles(resfiles)
    seqlist = df["sequence"].tolist()
    for index, pos in enumerate(design_positions):
        AAlist = []
        for seq in seqlist:
            AAlist.append(seq[index])
        df[str(pos)] = AAlist

    structdict = {}
    total_count = len(df)
    for pos in design_positions:
        unique_AAs = groups_from_unique_input(df, str(pos))
        posdict = {}
        for AA in unique_AAs:
            df_AA = df[df[str(pos)] == AA]
            count = len(df_AA)
            ratio = round(count / total_count, 3)
            total_score_list = [round(score, 3) for score in df_AA["total_score"].tolist()]
            cst_score_list = [round(score, 3) for score in df_AA["coordinate_constraint"].tolist()]
            posdict[AA] = {"pos": pos, "identity": AA, "count": count, "ratio": ratio, "total_score": total_score_list, "coordinate_constraint": cst_score_list}
        structdict[str(pos)] = posdict

    with open(filename, "w") as outfile:
        json.dump(structdict, outfile)
    return(structdict)


def import_json_to_dict(jsonpath: str):
    with open(jsonpath) as jsonfile:
        data = json.load(jsonfile)
    return(data)


def generate_mutations_dict(datadict):
    '''
    only accepts mutations that show up in at least 30% of coupled moves runs. if no mutation is above 30%, picks the most common one.
    '''
    mutations = {}
    for pos in datadict:
        df = pd.DataFrame(datadict[pos]).transpose().sort_values('ratio')
        df_filtered = df[df['ratio'] >= 0.3]
        if df_filtered.empty:
            df_filtered = df.head(1)
        mutations[pos] = df_filtered['identity'].to_list()

    return mutations


def generate_variants(mutation_dict, pdb):
    mutlist = []
    poslist = []
    for pos in mutation_dict:
        poslist.append(pos)
        mutlist.append(mutation_dict[pos])
    combs = list(itertools.product(*mutlist))
    seq = list(get_protein_sequence(pdb))
    variants = []
    for comb in combs:
        var = copy.deepcopy(seq)
        for index, AA in enumerate(comb):
            var[int(poslist[index]) - 1] = AA
        variants.append(''.join(var))
    return variants

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

def update_pose_with_cm_variants(output_dir, working_dir, cm_resultsdir, poses_df):
    statsdict_list = []
    mutationsdict_list = []
    seq_list = []
    variants_list = []

    cm_seqs = f'{output_dir}/cm_sequences/'
    os.makedirs(cm_seqs, exist_ok=True)

    for index, row in poses_df.iterrows():
        df = pd.DataFrame()
        statsdict = statsfiles_to_json(working_dir, row['poses_description'], f"{cm_resultsdir + row['poses_description']}.json")
        statsdict_list.append(statsdict)
        mutations = generate_mutations_dict(statsdict)
        mutationsdict_list.append(mutations)
        sequences = generate_variants(mutations, row['poses'])
        seq_list.append(sequences)
        #write sequences to fasta file
        fa_poses = []
        description = []
        for index, seq in enumerate(sequences):
            seq_name = f"{row['poses_description']}_{index+1:04d}"
            description.append(seq_name)
            fasta_path = f"{cm_seqs}{seq_name}.fa"
            with open(fasta_path, 'w') as fa:
                fa.write(f'>{seq_name}\n')
                fa.write(f'{seq}\n')
            fa_poses.append(fasta_path)
        df['poses'] = fa_poses
        df['poses_origin_description'] = row['poses_description']
        df['poses_description_new'] = description
        variants_list.append(df)

    variants = pd.concat(variants_list)
    poses_df['coupled_moves_stats'] = statsdict_list
    poses_df['coupled_moves_mutations'] = mutationsdict_list
    poses_df['coupled_moves_variants'] = seq_list

    #merge dataframes, set sequences as pose
    poses_df = poses_df.drop('poses', axis=1).merge(variants, left_on='poses_description', right_on='poses_origin_description', how='right')
    poses_df['poses_description'] = poses_df['poses_description_new']
    poses_df.drop(['poses_description_new', 'poses_origin_description'], axis=1, inplace=True)

    return poses_df


def filter_input_poses(poses, perresidue_plddt_column, ligand_chain, sitescore_cutoff, max_input_per_backbone, output_dir):

    diffrf.calc_ligand_stats(input_df=poses.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="pre_cm", ligand_chain=ligand_chain, save_path_list=[os.path.join(output_dir, description + '.pdb') for description in poses.poses_df['poses_description'].to_list()])
    poses.poses_df = poses.poses_df[poses.poses_df["pre_cm_ligand_clash"] == False]

    print(f'{len(poses.poses_df.index)} input poses passed ligand clash filter.')

    poses.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', 'pre_cm_bb', ['N', 'CA', 'C'])
    #TODO: change per residue plddt so that always the one from the last round is picked
    if sitescore_cutoff:
        poses.add_site_score(poses.poses_df['pre_cm_bb_motif_rmsd'].to_list(), poses.poses_df['motif_residues'].to_list(), poses.poses_df[perresidue_plddt_column].to_list(), 'pre_cm')
        poses.poses_df['backbone_id'] = poses.poses_df['poses_description'].str[:-5]
        filtered_dfs = []
        for index, df in poses.poses_df.groupby('backbone_id', sort=False):
            df = df[df['pre_cm_site_score'] >= sitescore_cutoff]
            df = df.sort_values('pre_cm_site_score', ascending=False)
            df = df.head(max_input_per_backbone)
            filtered_dfs.append(df)

        poses.poses_df = pd.concat(filtered_dfs)
        print(f'{len(poses.poses_df.index)} input poses passed site score filter.')

    return(poses)




def main(args):

    #absolute path for xml is necessary because coupled moves output can only be controlled by cd'ing into output directory before starting the run
    xml = os.path.abspath(args.cm_protocol)

    df = pd.read_json(args.json)
    #df = df[df['poses'].str.contains('A12-D7-C34-B14')]
    #TODO: add a filter for input pdbs here
    #if args.filter_input:
    coupled_moves = Poses(args.output_dir, df['poses'].to_list())


    input_dir = f'{args.output_dir}/input_pdbs/'
    os.makedirs(input_dir, exist_ok=True)

    #set maximum number of cpus for coupled moves run
    coupled_moves.max_rosetta_cpus = args.cm_max_cpus
    #merge pose dataframes
    coupled_moves.poses_df = coupled_moves.poses_df.drop(['input_poses', 'poses'], axis=1).merge(df, on='poses_description', how='left')

    #identify last refinement cycle number
    refinement_perresidue_plddt_columns = [column for column in coupled_moves.poses_df.columns if column.startswith('refinement_cycle_') and column.endswith('esm_perresidue_plddt')]

    coupled_moves = filter_input_poses(coupled_moves, 'refinement_cycle_02_esm_perresidue_plddt', args.ligand_chain, args.sitescore_cutoff, args.max_input_per_backbone, input_dir)


    #create directory for input pdbs


    options_list = []
    pose_list = []
    #iterate over dataframe, create coupled moves options
    for index, row in coupled_moves.poses_df.iterrows():
        #set filename for input pdbs
        pose_path = os.path.abspath(input_dir + row['poses_description'] + '.pdb')
        #add ligand to input pdbs
        si_tools.superimpose_add_chain_by_motif(row['updated_reference_frags_location'], row['poses'], args.ligand_chain, row['fixed_residues'], row['fixed_residues'], pose_path, ['CA', 'C', 'O'])
        #identify catalytic residues, format them for rosettascripts input
        cat_res = chainresdict_to_str(row['fixed_residues'])
        motif_res = chainresdict_to_str(row['motif_residues'])
        cm_options = f"-parser:script_vars cat_res={cat_res} motif_res={motif_res} cut1={args.cm_design_shell[0]} cut2={args.cm_design_shell[1]} cut3={args.cm_design_shell[2]} cut4={args.cm_design_shell[3]} favor_native_weight={args.cm_favor_native_weight} -in:file:native {row['updated_reference_frags_location']}"
        options_list.append(cm_options)
        pose_list.append(pose_path)

    #update path to input pdbs, add coupled_moves options
    coupled_moves.poses_df['coupled_moves_options'] = options_list
    coupled_moves.poses_df['poses'] = pose_list

    if 'covalent_bonds' in coupled_moves.poses_df.columns:
        if not (coupled_moves.poses_df['covalent_bonds'].str.strip() == "").any():
            print('Covalent bonds present! Adding LINK records to poses...')
            coupled_moves.add_LINK_to_poses('covalent_bonds', 'coupled_moves')


    cm_opts = f"-parser:protocol {xml} -coupled_moves:ligand_mode true"
    if args.cm_options:
        cm_opts = cm_opts + ' ' + args.cm_options

    #create working directory for coupled moves, cd into it (because cm output is always generated in starting directory)
    starting_dir = os.getcwd()
    working_dir = f'{args.output_dir}/cm_working_dir/'
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    #create copy of poses, because they will be overwritten after running rosettascripts
    old_poses = copy.deepcopy(coupled_moves)

    #run coupled moves
    cm = coupled_moves.rosetta("rosetta_scripts.default.linuxgccrelease", options=cm_opts, pose_options=coupled_moves.poses_df['coupled_moves_options'].to_list(), n=args.cm_nstruct, prefix='coupled_moves')

    #return to starting directory
    os.chdir(starting_dir)

    #create results dir for coupled moves
    cm_resultsdir = f'{args.output_dir}/cm_results/'
    os.makedirs(cm_resultsdir, exist_ok=True)

    #restore old poses
    coupled_moves = old_poses

    #add variants to poses
    coupled_moves.poses_df = update_pose_with_cm_variants(args.output_dir, working_dir, cm_resultsdir, coupled_moves.poses_df)

    #predict variants, calculate scores
    esm_preds = coupled_moves.predict_sequences(run_ESMFold, prefix="cm_predictions_esm")
    catres_sc_rmsd = coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', 'post_cm_sc')
    motif_bb_rmsd = coupled_moves.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', 'post_cm_bb', ['N', 'CA', 'C'])
    site_score = coupled_moves.add_site_score('post_cm_bb_motif_rmsd', 'motif_residues', "cm_predictions_esm_perresidue_plddt", 'post_cm')
    lig_stats = diffrf.calc_ligand_stats(input_df=coupled_moves.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="post_cm", ligand_chain=args.ligand_chain)
    coupled_moves.poses_df.to_json(coupled_moves.scorefile)

    #create final output
    esm_resultsdir = os.path.join(args.output_dir, 'esm_results/')
    os.makedirs(esm_resultsdir, exist_ok=True)
    ref_frags = [shutil.copy(ref_pose, esm_resultsdir) for ref_pose in coupled_moves.poses_df["input_poses"].to_list()]
    cm_esm_results = [shutil.copy(esm_pose, esm_resultsdir) for esm_pose in coupled_moves.poses_df["poses"].to_list()]

    filtered_df = copy.deepcopy(coupled_moves.poses_df[coupled_moves.poses_df["post_cm_ligand_clash"] == False])
    filtered_df['backbone_id'] = filtered_df['poses_description'].str[:-5]
    filtered_dfs = []
    for index, df in filtered_df.groupby('backbone_id', sort=False):
        df = df.sort_values('post_cm_site_score', ascending=False).head(3)
        filtered_dfs.append(df)
    filtered_df = pd.concat(filtered_dfs)
    utils.pymol_tools.pymol_alignment_scriptwriter(df=filtered_df, scoreterm='post_cm_site_score', top_n=len(filtered_df.index), path_to_script=f'{esm_resultsdir}align.pml', ascending=False, pose_col='poses_description', ref_pose_col='input_poses', motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres")




if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cm_protocol", type=str, default='rosetta/coupled_moves.xml', help="path to xmlfile that should be used for coupled moves")
    argparser.add_argument("--output_dir", type=str, required=True, help="working directory")
    argparser.add_argument("--cm_max_cpus", type=int, default=320, help="maximum number of cpus for coupled_moves")
    argparser.add_argument("--cm_nstruct", type=int, default=50, help="coupled_moves runs per input pdb")
    argparser.add_argument("--cm_design_shell", default=[6, 8, 10, 12], nargs=4, help="Design shells around ligand in Angstrom. All below cut1 is set to designable, all within cut2 is set to designable if Calpha Cbeta vector points to ligand, all within cut3 is set to repack, all within cut4 is set to repack if pointing to ligand.")
    argparser.add_argument("--cm_favor_native_weight", type=float, default=1.0, help="Weight for favoring input (mpnn) derived sequence. Low number will lead to more sequence diversity. Recommended values 0 to 2")
    argparser.add_argument("--cm_options", type=str, default=None, help="additional coupled moves cmd-line arguments in string format, e.g. '-extra_res_fa /path/to/params.pa' for adding a .params file. Use absolute paths!")
    argparser.add_argument("--json", type=str, required=True, help="path to jsonfile containing information about input pdbs (catres etc)")
    argparser.add_argument("--ligand_chain", type=str, default='Z', help="should always be Z in current workflow")
    argparser.add_argument("--sitescore_cutoff", type=float, default=None, help="cutoff for site score (1/e**(motif_bb_rmsd) * motif_plddt / 100) for filtering input pdbs. Recommended values are 0.3 to 0.5")
    argparser.add_argument("--max_input_per_backbone", type=int, default=5, help="maximum number of input pdbs coming from identical rfdiffusion backbones")

    args = argparser.parse_args()

    main(args)
