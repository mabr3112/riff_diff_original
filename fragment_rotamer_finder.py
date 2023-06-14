#!/home/tripp/anaconda3/envs/riffdiff/bin/python3.11

import logging
import os
import sys
import copy

# import dependencies
import warnings
import Bio
from Bio.PDB import *
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import itertools


# import custom modules
#sys.path.append("/home/tripp/riff_diff/")

import utils.adrian_utils as utils

def split_pdb_numbering(pdbnum):
    resnum = ""
    chain = ""
    for char in pdbnum:
        if char.isdigit():
            resnum += char
        else:
            chain += char
    resnum = int(resnum)
    if not chain:
        chain = "A"
    return [resnum, chain]

def tip_symmetric_residues():
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    library_folder = utils.path_ends_with_slash(library_folder)
    prefix = residue_identity.lower()
    rotlib = pd.read_csv(f'{library_folder}{prefix}.bbdep.rotamers.lib')
    if residue_identity in AAs_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    return rotlib

def AAs_up_to_chi1():
    AAs = ['CYS', 'SER', 'THR', 'VAL']
    return AAs

def AAs_up_to_chi2():
    AAs = ['ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR']
    return AAs

def AAs_up_to_chi3():
    AAs = ['GLN', 'GLU', 'MET']
    return AAs

def AAs_up_to_chi4():
    AAs = ['ARG', 'LYS']
    return AAs


def rama_plot_old(df, x, y, color_column, size_column, filename):
    '''
    plots phi and psi angles
    '''
    # Normalize the values in the color & size column
    data = copy.deepcopy(df)
    norm_color = Normalize(vmin=0, vmax=data[color_column].max())
    norm_size = Normalize(vmin=0, vmax=data[size_column].max())
    # Create a colormap with white to blue gradient
    cmap = plt.get_cmap("Blues")
    # Map the normalized values to the colormap
    colors = cmap(norm_color(data[color_column]))
    # Set the size of dots according to the values in the size column
    data['norm_size'] = norm_size(data[size_column])
    size = data['norm_size'] * 30
    plt.figure()
    plt.scatter(data[x], data[y], c=colors, s=size, marker=".")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap)
    scalar_mappable.set_array(data[color_column])
    cb = plt.colorbar(mappable=scalar_mappable, label=color_column)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xticks(range(-180, 181, 60))
    plt.yticks(range(-180, 181, 60))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{filename}.png', dpi=300)
    plt.show()

def rama_plot(df, x_col, y_col, color_col, size_col, save_path=None):
    df_list = []
    for phi_psi, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = plt.cm.Blues
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    norm_size = plt.Normalize(0, df[size_col].max())
    ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=df[size_col], norm=norm_color)
    fig.colorbar(sm, label="probability", ax=ax)
    ax.set_xlabel("phi [degrees]")
    ax.set_ylabel("psi [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    fig.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)

def identify_backbone_angles_suitable_for_rotamer(residue_identity:str, rotlib:pd.DataFrame(), output_prefix:str=None, output_dir:str=None, limit_sec_struct:str=None, occurence_cutoff:int=5, max_output:int=None, rotamer_diff_to_best=0.05):
    '''
    finds phi/psi angles most common for a given set of chi angles from a rotamer library
    chiX_bin multiplies the chiXsigma that is used to check if a rotamer fits to the given set of chi angles --> increasing leads to higher number of hits, decreasing leads to rotamers that more closely resemble input chis. Default=1
    if score_cutoff is provided, returns only phi/psi angles above score_cutoff
    if fraction is provided, returns only the top rotamer fraction ranked by score
    if max_output is provided, returns only max_output phi/psi combinations
    '''

    filename = utils.create_output_dir_change_filename(output_dir, output_prefix + f'{residue_identity}_rama_pre_filtering')
    df_list = []

    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurence', filename)


    if limit_sec_struct:
        rotlib = filter_rotamers_by_sec_struct(rotlib, limit_sec_struct)

    if occurence_cutoff:
        rotlib = rotlib.loc[rotlib['phi_psi_occurence'] > occurence_cutoff]

    if rotamer_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - rotamer_diff_to_best)]

    rotlib = rotlib.sort_values(["probability", "phi_psi_occurence"], ascending=False)

    if max_output:
        rotlib = rotlib.head(max_output)
    if rotlib.empty:
        raise RuntimeError('Could not find any rotamers that fit. Try setting different filter values!')

    rotlib.reset_index(drop=True, inplace=True)

    filename = utils.create_output_dir_change_filename(output_dir, output_prefix + f'{residue_identity}_rama_post_filtering')
    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurence', filename)

    return rotlib

def filter_rotamers_by_sec_struct(rotlib:pd.DataFrame, secondary_structure:str):
    filtered_list = []
    sec_structs = [*secondary_structure]
    #phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = [x for x in range(-170, -39, 10)] + [x for x in range(60, 81, 10)]
        psi_range = [x for x in range(-180, -159, 10)] + [x for x in range(-40, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(80, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = [x for x in range(-170, -59, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(90, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = [x for x in range(-130, -39, 10)] + [x for x in range(50, 71, 10)]
        psi_range = [x for x in range(-50, 41, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = [x for x in range(-100, -39, 10)]
        psi_range = [x for x in range(-60, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = [x for x in range(-140, -49, 10)]
        psi_range = [x for x in range(-80, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)] + [x for x in range(50, 111, 10)]
        psi_range = [x for x in range(-180, -149, 10)] + [x for x in range(-60, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = [x for x in range(-130, -40, 10)] + [x for x in range(40, 111, 10)]
        psi_range = [x for x in range(-60, 61, 10)] + [x for x in range(120, 151, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    rotlib = pd.concat(filtered_list)
    return rotlib

def import_fragment_library(library_path:str):
    '''
    reads in a fragment library
    '''
    library = pd.read_csv(library_path, header=0)
    library.drop(library.columns[[0]], axis=1, inplace=True)
    return library

def is_unique(df_column):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = df_column.to_numpy()
    return (a[0] == a).all()

def check_for_chainbreaks(df, columname, fragsize):
    '''
    returns true if dataframe column is consistently numbered
    '''
    if df[columname].diff().sum() + 1 == fragsize:
        return True
    else:
        return False
    
def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):
    
    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        ss_check = True
        for sec_struct in frag_sec_struct_fraction:
            if not df['ss'].str.contains(sec_struct).sum() / len(df.index) >= frag_sec_struct_fraction[sec_struct]:
                ss_check = False

        if ss_check == True:
            frags_df_list.append(df)
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df 
    else:
        return pd.DataFrame()

def filter_frags_df_by_score(frags_df, score_cutoff):
    
    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        if df['bb_score'].mean() < score_cutoff:
            frags_df_list.append(df)
    
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df 
    else:
        return pd.DataFrame()

def filter_fragments_rmsd(AAalphabet, theozyme_residue, rotamer_position, flip_symmetric, flip_histidines, his_central_atom, ligand, covalent_bond, database, frag_df:pd.DataFrame(), rmsd_cutoff:float=1.0, max_output:int=100, frag_sec_struct_fraction=None, bfactor_cutoff=None, score_cutoff=None):
    '''
    checks if fragments generated from dataframe pass filters until maximum number of output has been reached
    '''
    print('Filtering fragments...')
    logging.info('Filtering fragments...')
    filtered_frags = Structure.Structure("filtered_frags")
    rmsd_filtered_frag_df = pd.DataFrame()

    for fragment, df in frag_df.groupby('frag_num', sort=False):
        #check if maximum number of fragments for rotamer position have already been reached
        if len([model for model in filtered_frags.get_models()]) >= max_output:
            break

        #filter for secondary structure content
        ss_check = True
        if frag_sec_struct_fraction:
            for sec_struct in frag_sec_struct_fraction:
                if not df['ss'].str.contains(sec_struct).sum() / len(df.index) >= frag_sec_struct_fraction[sec_struct]:
                    ss_check = False

        if ss_check == False:
            continue

        #filter for bfactor
        if bfactor_cutoff:
            if not df['bfactor'].mean() < bfactor_cutoff:
                continue

        #filter for backbone score
        if not score_cutoff == None:
            if not df['bb_score'].mean() < score_cutoff:
                #break since fragments are sorted by backbone score, if this fails all following will fail as well
                break

        #check number of fragments in output dataframe, break if max_output is reached
        #check rmsd of current fragment against all fragments in the already filtered dataframe.
        frag = create_fragment_from_df(df)
        frag = attach_rotamer_to_fragments(df, frag, AAalphabet)
        frag = align_to_sidechain(frag, frag[rotamer_position], theozyme_residue, False, False)
        filtered_frags, rmsd_filtered_frag_df = check_fragment(frag, filtered_frags, df, rmsd_filtered_frag_df, ligand, database, rotamer_position, covalent_bond, rmsd_cutoff)
        if ((flip_symmetric == True and theozyme_residue.get_resname() in tip_symmetric_residues()) or (flip_histidines == True and theozyme_residue.get_resname() == "HIS")) and len([model for model in filtered_frags.get_models()]) < max_output:
            flipped_frag = copy.deepcopy(frag)
            flipped_frag = align_to_sidechain(flipped_frag, flipped_frag[rotamer_position], theozyme_residue, True, True, his_central_atom, ligand)
            filtered_frags, rmsd_filtered_frag_df = check_fragment(flipped_frag, filtered_frags, df, rmsd_filtered_frag_df, ligand, database, rotamer_position, covalent_bond, rmsd_cutoff)

    return {'data': rmsd_filtered_frag_df, 'structure': filtered_frags}

def add_frag_to_structure(frag, structure):
    frag_num = len([model for model in structure.get_models()])
    model = Model.Model(frag_num)
    model.add(frag)
    structure.add(model)

def check_fragment(frag, frag_list, frag_df, df_list, ligand, vdw_radii, rotamer_position, covalent_bond, rmsd_cutoff, backbone_ligand_clash_detection_vdw_multiplier, rotamer_ligand_clash_detection_vdw_multiplier):
    frag_df['frag_num'] = len(frag_list)
    if ligand:
        #check for backbone clashes
        clash_check = distance_detection(frag, ligand, vdw_radii, True, True, backbone_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond)
        if clash_check == False:
            #check for rotamer clashes
            clash_check = distance_detection(frag[rotamer_position], ligand, vdw_radii, False, True, rotamer_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond, True)
    else:
        clash_check = False
    #add the first encountered fragment without rmsd checking
    if len(frag_list) == 0 and clash_check == False:
        frag_list.append(frag)
        df_list.append(frag_df)
        return frag_list, df_list
    #calculate rmsds for all already accepted fragments
    if len(frag_list) > 0 and clash_check == False:
        rmsdlist = [calculate_rmsd_bb(picked_frag, frag) for picked_frag in frag_list]
        #if the lowest rmsd compared to all other fragments is higher than the set cutoff, add it to the filtered dataframe
        if min(rmsdlist) >= rmsd_cutoff:
            frag_list.append(frag)
            df_list.append(frag_df)

    return frag_list, df_list

def import_vdw_radii(database_dir):
    '''
    from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page), accessed 30.1.2023
    '''
    vdw_radii = pd.read_csv(f'{database_dir}/vdw_radii.csv')
    vdw_radii.drop(['name', 'atomic_number', 'empirical', 'Calculated', 'Covalent(sb)', 'Covalent(tb)', 'Metallic'], axis=1, inplace=True)
    vdw_radii.dropna(subset=['VdW_radius'], inplace=True)
    vdw_radii['VdW_radius'] = vdw_radii['VdW_radius'] / 100
    vdw_radii = vdw_radii.set_index('element')['VdW_radius'].to_dict()
    return vdw_radii

def distance_detection(entity1, entity2, vdw_radii:dict, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, resnum:int=None, covalent_bond:str=None, ignore_func_groups:bool=True):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''
    backbone_atoms = ['CA', 'C', 'N', 'O', 'H']
    if bb_only == True and ligand == False:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms() if atom.name in backbone_atoms)
    elif bb_only == True and ligand == True:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms())
    else:
        entity1_atoms = (atom for atom in entity1.get_atoms())
        entity2_atoms = (atom for atom in entity2.get_atoms())

    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        #skip clash detection for covalent bonds
        covalent = False
        if ignore_func_groups == True and atom_combination[0].name in atoms_of_functional_groups():
            covalent = True
        if resnum and covalent_bond:
            for cov_bond in covalent_bond.split(','):
                if atom_combination[0].get_parent().id[1] == resnum and atom_combination[0].name == cov_bond.split(':')[0] and atom_combination[1].name == cov_bond.split(':')[1]:
                    covalent = True
        if covalent == True:
            continue
        distance = atom_combination[0] - atom_combination[1]
        element1 = atom_combination[0].element
        element2 = atom_combination[1].element
        clash_detection_limit = clash_detection_vdw_multiplier * (vdw_radii[str(element1)] + vdw_radii[str(element2)])
        if distance < clash_detection_limit:
            return True
    return False

def atoms_of_functional_groups():
    return ["NH1", "NH2", "OD1", "OD2", "ND2", "NE", "SG", "OE1", "OE2", "NE2", "ND1", "NZ", "SD", "OG", "OG1", "NE1", "OH"]


def sort_frags_df_by_score(fragment_df):
    score_df_list = []
    for fragment, df in fragment_df.groupby('frag_num', sort=False):
        score_df_list.append((df['bb_score'].mean(), df))

    #sort fragments by average score
    if len(score_df_list) > 1:
        score_df_list.sort(key=lambda x: x[0])

    #assemble dataframe
    score_sorted_list = []
    for score_df in score_df_list:
        df = score_df[1]
        score_sorted_list.append(df)
    fragment_df = pd.concat(score_sorted_list)
    return fragment_df

def calculate_rmsd_bb(entity1, entity2):
    '''
    calculates rmsd of 2 structures considering backbone heavy atoms. does no superposition!
    '''
    sup = Superimposer()
    bb_atoms = ["N", "CA", "C", "O"]
    entity1_atoms = [atom for atom in entity1.get_atoms() if atom.id in bb_atoms]
    entity2_atoms = [atom for atom in entity2.get_atoms() if atom.id in bb_atoms]

    rmsd = math.sqrt(sum([(atom1 - atom2) ** 2 for atom1, atom2 in zip(entity1_atoms, entity2_atoms)]) / len(entity1_atoms))

    return rmsd

def create_fragment_from_df(df:pd.DataFrame()):
    '''
    creates a biopython chain from dataframe containing angles and coordinates
    '''
    chain = Chain.Chain('A')

    serial_num = 1
    resnum = 1
    for index, row in df.iterrows():
        res = Residue.Residue((' ', resnum, ' '), row['AA'], ' ')
        resnum += 1
        for atom in ["N", "CA", "C", "O"]:
            coords = np.array([row[f'{atom}_x'], row[f'{atom}_y'], row[f'{atom}_z']])
            bfactor = 0 if math.isnan(row['probability']) else row['probability'] * 100
            bb_atom = Atom.Atom(name=atom, coord=coords, bfactor=bfactor, occupancy=1.0, altloc=' ', fullname=f' {atom} ', serial_number=serial_num, element=atom[0])
            serial_num += 1
            res.add(bb_atom)
        chain.add(res)

    return chain

def identify_positions_for_rotamer_insertion(fraglib, rotlib, rot_sec_struct, phi_psi_binsize, limit_frags_to_chi, limit_frags_to_res_id):

    phi_psi_binsize = phi_psi_binsize / 2
    #convert string to list
    if rot_sec_struct:
        rot_sec_struct = [*rot_sec_struct]

    #add 360 to all negative angles
    angles = ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4']
    columns = list(rotlib.columns)
    for angle in [column for column in columns if column in angles]:
        rotlib.loc[rotlib[angle] < 0, angle] += 360
    rotamer_positions_list = []
    for index, row in rotlib.iterrows():
        #filter based on difference & amino acid identity
        rotamer_positions = fraglib.loc[(np.minimum(abs(fraglib['phi'] - row['phi']), abs(fraglib['phi'] - row['phi'] + 360)) <= phi_psi_binsize) & (np.minimum(abs(fraglib['psi'] - row['psi']), abs(fraglib['psi'] - row['psi'] + 360)) <= phi_psi_binsize)]
        if limit_frags_to_res_id == True:
            rotamer_positions = rotamer_positions.loc[fraglib['AA'] == row['identity']]
        if rotamer_positions.empty:
            continue
        if rot_sec_struct:
            rotamer_positions = rotamer_positions.loc[fraglib['ss'].isin(rot_sec_struct)]
        if rotamer_positions.empty:
            continue
        for chi in [column for column in columns if column in ['chi1', 'chi2', 'chi3', 'chi4']]:
            if limit_frags_to_chi == True:
                #filter for positions that have chi angles within 2 stdev from input chis
                rotamer_positions = rotamer_positions.loc[(np.minimum(abs(rotamer_positions[chi] - row[chi]), abs(rotamer_positions[chi] - row[chi] + 360)) < 2 * row[f'{chi}sig'])]
            #change chi angles to mean value from bin
            rotamer_positions[chi] = row[chi]

        rotamer_positions['probability'] = row['probability']
        rotamer_positions_list.append(rotamer_positions)

    if not rotamer_positions_list:
        raise RuntimeError('Could not find any fragment positions that fit criteria. Try adjusting rotamer secondary structure or use different rotamers!')
    elif len(rotamer_positions_list) > 1:
        rotamer_positions = pd.concat(rotamer_positions_list)
    return rotamer_positions

def extract_fragments(rotamer_positions_df, fraglib, frag_pos_to_replace, fragsize):
    '''
    finds fragments based on phi/psi combination and residue identity
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    bfactor_cutoff: only return fragments with bfactor below this threshold
    rmsd_cutoff: compare fragments by fragments, only accept fragments that have rmsd above this threshold.
    '''

    #choose fragments from fragment library that contain the positions selected above
    fragnum = 0
    frag_dict = {}
    for pos in frag_pos_to_replace:
        frag_dict[pos] = []
    for index, row in rotamer_positions_df.iterrows():
        for pos in frag_pos_to_replace:
            upper = index + fragsize - pos
            lower = index - pos + 1
            df = fraglib.loc[lower:upper, fraglib.columns]
            #only choose fragments with correct length etc
            if is_unique(df['pdb']) and len(df) == fragsize and check_for_chainbreaks(df, 'position', fragsize) == True:
                fragnum = fragnum + 1
                df.loc[:, 'frag_num'] = fragnum
                df.loc[:, 'rotamer_pos'] = int(pos)
                prob = row['probability']
                df.at[index, 'probability'] = prob
                frag_dict[pos].append(df)

    for pos in frag_dict:
        if len(frag_dict[pos]) > 0:
            frag_dict[pos] = pd.concat(frag_dict[pos])
        else:
            frag_dict[pos] = pd.DataFrame()
    return frag_dict

def is_unique(s):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = s.to_numpy()
    return (a[0] == a).all()

def attach_rotamer_to_fragments(df, frag, AA_alphabet):
    rotamer_on_fragments = Structure.Structure("rotonfrags")

    rotamer = identify_rotamer_position_by_probability(df)
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    chi_angles = [None if math.isnan(rotamer[chi]) else rotamer[chi] for chi in columns]
    rot_pos = int(rotamer['rotamer_pos'])
    to_mutate = frag[rot_pos]
    resid = to_mutate.id
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(AA_alphabet, rotamer['AA'], resid, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], chi_angles[0], chi_angles[1], chi_angles[2], chi_angles[3], rotamer['probability'])
    delattr(res, 'internal_coord')
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments

def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer):
    #fragment = copy.deepcopy(fragment)
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.rotran
    sup.apply(rotamer)

    fragment.insert(rotamer.id[1]-1, rotamer)

    return fragment


def extract_backbone_angles(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    chain = copy.deepcopy(chain)
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if not phi == None:
        phi = round(phi, 1)
    if not psi == None:
        psi = round(psi, 1)
    if not omega == None:
        omega = round(omega, 1)
    return {"phi": phi, "psi": psi, "omega": omega, "carb_angle": carb_angle, "tau": tau}

def extract_backbone_bondlengths(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    chain = copy.deepcopy(chain)
    chain.atom_to_internal_coordinates()
    N_CA = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    CA_C = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    C_O = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": N_CA, "CA_C": CA_C, "C_O": C_O}



def generate_rotamer(AAalphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None):
    '''
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    '''
    alphabet = copy.deepcopy(AAalphabet_structure)
    for res in alphabet[0]["A"]:
        if res.get_resname() == residue_identity:
            #set internal coordinates
            alphabet[0]["A"].atom_to_internal_coordinates()
            #change angles to specified value
            if tau:
                res.internal_coord.set_angle("tau", tau)
            if carb_angle:
                res.internal_coord.bond_set("N:CA:C:O", carb_angle)
            if phi:
                res.internal_coord.set_angle("phi", phi)
            if psi:
                res.internal_coord.set_angle("psi", psi)
            if omega:
                res.internal_coord.set_angle("omega", omega)
            if N_CA_length:
                res.internal_coord.set_length("N:CA", N_CA_length)
            if CA_C_length:
                res.internal_coord.set_length("CA:C", CA_C_length)
            if C_O_length:
                res.internal_coord.set_length("C:O", C_O_length)


            if residue_identity in AAs_up_to_chi4():
                max_chis = 4
            elif residue_identity in AAs_up_to_chi3():
                max_chis = 3
            elif residue_identity in AAs_up_to_chi2():
                max_chis = 2
            elif residue_identity in AAs_up_to_chi1():
                max_chis = 1
            else:
                max_chis = 0

            if max_chis > 0:
                res.internal_coord.bond_set("chi1", chi1)
            if max_chis > 1:
                res.internal_coord.bond_set("chi2", chi2)
            if max_chis > 2:
                res.internal_coord.bond_set("chi3", chi3)
            if max_chis > 3:
                res.internal_coord.set_angle("chi4", chi4)
            alphabet[0]["A"].internal_to_atom_coordinates()
            #change residue number to the one that is replaced (detaching is necessary because otherwise 2 res with same resid would exist in alphabet)
            alphabet[0]["A"].detach_child(res.id)
            res.id = res_id
            if rot_probability:
                for atom in res.get_atoms():
                    atom.bfactor = rot_probability * 100

            return res

def identify_rotamer_position_by_probability(df):
    #drop all rows that do not contain the rotamer, return a series --> might cause issues downstream if more than one rotamer on a fragment
    rotamer_pos = df.dropna(subset = ['probability']).squeeze()
    return(rotamer_pos)

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True, flip_histidines:bool=False, his_central_atom:str="auto", ligand=None):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()
    bbf_residue_identity = entity_residue_to_align.get_resname()
    if flip_histidines == True:
        if not his_central_atom in ["NE2", "ND1", "auto"]:
            raise KeyError(f'his_central_atom must be either NE2 or ND1 or auto, not {his_central_atom}')

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric == True and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    #flip the orientation of His residues, flip direction depends on the orientation of the coordinating atom
    if sc_residue_identity == "HIS" and flip_histidines == True:
        if his_central_atom == "auto":
            if not ligand:
                raise RuntimeError(f"Ligand is required if using <flip_histidines='auto'>!")
            HIS_NE2 = sidechain["NE2"]
            HIS_ND1 = sidechain["ND1"]
            lig_atoms =  [atom for atom in ligand.get_atoms()]
            NE2_distance = min([HIS_NE2 - atom for atom in lig_atoms])
            ND1_distance = min([HIS_ND1 - atom for atom in lig_atoms])
            if NE2_distance < ND1_distance:
                his_central_atom = "NE2"
            else:
                his_central_atom = "ND1"
        if his_central_atom == "NE2":
            bbf_atoms = [entity_residue_to_align["NE2"], entity_residue_to_align["ND1"], entity_residue_to_align["CG"]]
            sc_atoms = [sidechain["NE2"], sidechain["CG"], sidechain["ND1"]]
        elif his_central_atom == "ND1":
            bbf_atoms = [entity_residue_to_align["ND1"], entity_residue_to_align["NE2"], entity_residue_to_align["CD2"]]
            sc_atoms = [sidechain["ND1"], sidechain["CD2"], sidechain["NE2"]]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    sup.rotran
    sup.apply(entity)

    return entity

def identify_rotamer_by_bfactor_probability(entity):
    '''
    returns the residue number where bfactor > 0, since this is where the rotamer probability was saved
    '''
    residue = None
    for atom in entity.get_atoms():
        if atom.bfactor > 0:
            residue = atom.get_parent()
            break
    if not residue:
        raise RuntimeError('Could not find any rotamer in chain. Maybe rotamer probability was set to 0?')
    resnum = residue.id[1]
    return resnum

def atoms_for_func_group_alignment(residue):
    '''
    return the atoms used for superposition via functional groups
    '''
    sc_residue_identity = residue.get_resname()
    func_groups = {
        "ALA": ["CB", "CA", "N"],
        "ARG": ["NH1", "NH2", "CZ"],
        "ASP": ["OD1", "OD2", "CG"],
        "ASN": ["OD1", "ND2", "CG"],
        "CYS": ["SG", "CB", "CA"],
        "GLU": ["OE1", "OE2", "CD"],
        "GLN": ["OE1", "NE2", "CD"],
        "GLY": ["CA", "N", "C"],
        "HIS": ["ND1", "NE2", "CG"],
        "ILE": ["CD1", "CG1", "CB"],
        "LEU": ["CD1", "CD2", "CG"],
        "LYS": ["NZ", "CE", "CD"],
        "MET": ["CE", "SD", "CG"],
        "PHE": ["CD1", "CD2", "CZ"],
        "PRO": ["CD", "CG", "CB"],
        "SER": ["OG", "CB", "CA"],
        "THR": ["OG1", "CG2", "CB"],
        "TRP": ["NE1", "CZ3", "CG"],
        "TYR": ["CE1", "CE2", "OH"],
        "VAL": ["CG1", "CG2", "CB"]
        }
    if not sc_residue_identity in func_groups:
        raise RuntimeError(f'Unknown residue with name {sc_residue_identity}!')
    else:
        return [residue[atom] for atom in func_groups[sc_residue_identity]]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    limit_frags_to_res_id = str2bool(args.limit_frags_to_res_id)
    limit_frags_to_chi = str2bool(args.limit_frags_to_chi)
    flip_histidines = str2bool(args.flip_histidines)
    flip_symmetric = str2bool(args.flip_symmetric)

    output_dir = utils.path_ends_with_slash(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{output_dir}fragment_picker_{args.output_prefix}{args.theozyme_resnum}.log")
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    #import and prepare stuff
    database_dir = utils.path_ends_with_slash(args.database_dir)
    theozyme = utils.import_structure_from_pdb(args.theozyme_pdb)
    resnum, chain = split_pdb_numbering(args.theozyme_resnum)
    theozyme_residue = theozyme[0][chain][resnum]
    residue_identity = theozyme_residue.get_resname()
    AA_alphabet = utils.import_structure_from_pdb(f'{database_dir}AA_alphabet.pdb')
    vdw_radii = import_vdw_radii(database_dir)
    #import fragment library
    fraglib = import_fragment_library(f'{database_dir}/fraglib_2.5A_scored.csv')

    if len(args.frag_pos_to_replace) > 1:
        frag_pos_to_replace = [i for i in range(args.frag_pos_to_replace[0], args.frag_pos_to_replace[1]+1)]
    else:
        frag_pos_to_replace = args.frag_pos_to_replace
    
    #sanity check command line input
    if args.frag_sec_struct_fraction:
        sec_structs = args.frag_sec_struct_fraction.split(',')
        sec_dict = {}
        for i in sec_structs:
            sec, frac = i.split(':')
            frac = float(frac)
            if frac > 1 or frac < 0:
                logging.error(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
                raise ValueError(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
            if (args.fragsize - frac * args.fragsize) < 1 and sec != args.rot_sec_struct and args.rot_sec_struct != None:
                logging.error(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
                raise KeyError(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None

    if args.ligand_chain:
        if not args.ligand_chain in [chain.id for chain in theozyme.get_chains()]:
            raise RuntimeError(f'No ligand found in chain {args.ligand_chain}. Please make sure the theozyme pdb is correctly formatted.')
        ligand = copy.deepcopy(theozyme[0][args.ligand_chain])
        ligand.id = "Z"
        logging.info(f"Found ligand in chain {args.ligand_chain}")
        print(f"Found ligand in chain {args.ligand_chain}")
    else:
        logging.warning(f"WARNING: No ligand chain specified. Was this intentional?")

    if len(args.frag_pos_to_replace) == 1:
        frag_pos_to_replace = frag_pos_to_replace
    else:
        frag_pos_to_replace = [i for i in range(args.frag_pos_to_replace[0], args.frag_pos_to_replace[1]+1)]

    if args.covalent_bond:
        if not args.ligand_chain:
            logging.warning("WARNING: Covalent bonds are only useful if ligand is present!")
        for cov_bond in args.covalent_bond.split(','):
            if not cov_bond.split(':')[0] in [atom.name for atom in theozyme_residue.get_atoms()]:
                raise KeyError(f"Could not find atom {cov_bond.split(':')[0]} from covalent bond {cov_bond} in residue {args.theozyme_resnum}!")
            if not cov_bond.split(':')[1] in [atom.name for atom in ligand.get_atoms()]:
                raise KeyError(f"Could not find atom {cov_bond.split(':')[1]} from covalent bond {cov_bond} in ligand chain {args.ligand_chain}!")

    database = utils.path_ends_with_slash(args.database_dir)
    #find rotamer library for given amino acid
    logging.info(f"Importing backbone dependent rotamer library for residue {residue_identity} from {args.database_dir}")
    print(f"Importing backbone dependent rotamer library for residue {residue_identity} from {args.database_dir}")
    rotlib = return_residue_rotamer_library(database, residue_identity)
    
    #filter rotamer library for most probable rotamers
    logging.info(f"Identifying most probable rotamers for residue {residue_identity}")
    print(f"Identifying most probable rotamers for residue {residue_identity}")
    rotlib = identify_backbone_angles_suitable_for_rotamer(residue_identity, rotlib, f'{args.output_prefix}{args.theozyme_resnum}_', f'{output_dir}rotamer_info', args.rot_sec_struct, args.phipsi_occurence_cutoff, args.max_phi_psis, args.rotamer_diff_to_best)
    logging.info(f"Found {len(rotlib.index)} phi/psi combinations.")
    print(f"Found {len(rotlib.index)} phi/psi combinations.")
    
    #write filtered rotamer library to disk
    rotlibcsv = utils.create_output_dir_change_filename(f'{output_dir}rotamer_info', args.output_prefix + f'rotamers_{args.theozyme_resnum}_{residue_identity}.csv')
    logging.info(f"Writing phi/psi combinations to {rotlibcsv}.")
    print(f"Writing phi/psi combinations to {rotlibcsv}.")
    rotlib.to_csv(rotlibcsv)

    logging.info(f"Looking for positions with phi/psi angles fitting to picked combinations in fragment database...")
    print(f"Looking for positions with phi/psi angles fitting to picked combinations in fragment database...")


    rotamer_positions = identify_positions_for_rotamer_insertion(fraglib, rotlib, args.rot_sec_struct, args.phi_psi_bin, limit_frags_to_chi, limit_frags_to_res_id)
    logging.info(f'Found {len(rotamer_positions.index)} positions for rotamer insertion.')
    print(f'Found {len(rotamer_positions.index)} positions for rotamer insertion.')

    frag_dict = extract_fragments(rotamer_positions, fraglib, frag_pos_to_replace, args.fragsize)
    frag_num = int(sum([len(frag_dict[pos].index) for pos in frag_dict]) / args.fragsize)
    logging.info(f'Found {frag_num} fragments.')
    print(f'Found {frag_num} fragments.')

    #filter fragments
    for pos in frag_dict:
        frag_num = int(len(frag_dict[pos].index) / args.fragsize)
        logging.info(f'Found {frag_num} fragments for position {pos}.')
        print(f'Found {frag_num} fragments for position {pos}.')
        if frag_num == 0:
            continue
        if sec_dict:
            frag_dict[pos] = filter_frags_df_by_secondary_structure_content(frag_dict[pos], sec_dict)
            logging.info(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed secondary structure filtering with filter {args.frag_sec_struct_fraction} for position {pos}.")
            print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed secondary structure filtering with filter {args.frag_sec_struct_fraction} for position {pos}.")
        if args.score_cutoff:
            frag_dict[pos] = filter_frags_df_by_score(frag_dict[pos], args.score_cutoff)
            logging.info(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed backbone score filtering with cutoff {args.score_cutoff} for position {pos}.")
            print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed backbone score filtering with cutoff {args.score_cutoff} for position {pos}.")
        if not frag_dict[pos].empty:
            logging.info(f"Sorting fragments by backbone score for position {pos}.")
            print(f"Sorting fragments by backbone score for position {pos}.")
            frag_dict[pos] = sort_frags_df_by_score(frag_dict[pos])
        else:
            frag_dict.pop(pos)
            logging.info(f"Could not find fragments for position {pos}.")
            print(f"Could not find fragments for position {pos}.")

    if len(frag_dict) == 0:
        raise RuntimeError('Could not find any fragments that fit criteria! Try adjusting filter values!')

    #create fragments from df, only accept fragments with rmsd higher than <rmsd_cutoff> to any of the already accepted fragments
    residual_to_max = 0 
    fragments = Structure.Structure('fragments')
    frags_table = []
    frags_info = []
    frag_num = 0

    for pos in frag_dict:
        logging.info(f'Creating fragments, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {args.rmsd_cutoff} A for position {pos}.')
        print(f'Creating fragments, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {args.rmsd_cutoff} A for position {pos}.')
        picked_frags = []
        frag_dfs = []
        #calculate maximum number of fragments per position, add missing fragments from previous position to maximum
        max_frags = int(args.max_frags / len(frag_dict)) + residual_to_max
        #loop over fragment dataframe, create fragments
        for frag_index, frag_df in frag_dict[pos].groupby('frag_num', sort=False):
            if len(picked_frags) < max_frags:
                frag = create_fragment_from_df(frag_df)
                frag = attach_rotamer_to_fragments(frag_df, frag, AA_alphabet)
                frag = align_to_sidechain(frag, frag[pos], theozyme_residue, False, False)
                picked_frags, frag_dfs = check_fragment(frag, picked_frags, frag_df, frag_dfs, ligand, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier)
                #flip rotamer and fragment if theozyme residue is tip symmetric or a histidine
                if ((flip_symmetric == True and theozyme_residue.get_resname() in tip_symmetric_residues()) or (flip_histidines == True and theozyme_residue.get_resname() == "HIS")) and len(picked_frags) < max_frags:
                    flipped_frag = copy.deepcopy(frag)
                    flipped_frag = align_to_sidechain(frag, frag[pos], theozyme_residue, True, True, args.his_central_atom, ligand)
                    picked_frags, frag_dfs = check_fragment(flipped_frag, picked_frags, frag_df, frag_dfs, ligand, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier)
            else:
                break
        logging.info(f"Found {len(picked_frags)} fragments for position {pos} of a maximum of {max_frags}.")
        print(f"Found {len(picked_frags)} fragments for position {pos} of a maximum of {max_frags}.")
        residual_to_max = max_frags - len(picked_frags)
        for frag, df in zip(picked_frags, frag_dfs):
            row = pd.Series({'model_num': frag_num, 'rotamer_pos': pos, 'AAs': df['AA'].to_list(), 'bb_score': df['bb_score'].to_list(), 'secondary_structure': df['ss'].to_list(), 'bfactor': df['bfactor'].to_list(), 'rotamer_probability': float(df.dropna(subset = ['probability']).squeeze()['probability']), 'covalent_bond': args.covalent_bond})
            model = Model.Model(frag_num)
            model.add(frag)
            if ligand:
                model.add(ligand)
                row['ligand_chain'] = 'Z'
            fragments.add(model)
            df['frag_num'] = frag_num
            frags_table.append(df)
            frags_info.append(row)
            frag_num += 1
        del(picked_frags)

    logging.info(f'Found {len(frags_info)} fragments that passed all filters.')
    print(f'Found {len(frags_info)} fragments that passed all filters.')

    #write fragment info to disk
    frags_table = pd.concat(frags_table)
    fragscsv = utils.create_output_dir_change_filename(f'{output_dir}fragment_info', args.output_prefix + f'fragments_{args.theozyme_resnum}_{residue_identity}.csv')
    logging.info(f'Writing fragment details to {fragscsv}.')
    print(f'Writing fragment details to {fragscsv}.')
    frags_table.to_csv(fragscsv)


    #write multimodel fragment pdb to disk
    filename_pdb = utils.create_output_dir_change_filename(output_dir, args.output_prefix + f'{args.theozyme_resnum}_{residue_identity}.pdb')
    logging.info(f'Writing multimodel fragment pdb to {filename_pdb}.')
    print(f'Writing multimodel fragment pdb to {filename_pdb}.')
    utils.write_multimodel_structure_to_pdb(fragments, filename_pdb)

    #write output json to disk
    frags_info = pd.DataFrame(frags_info)
    frags_info['poses'] = os.path.abspath(filename_pdb)
    frags_info['poses_description'] = f'{args.output_prefix}{args.theozyme_resnum}_{residue_identity}'
    filename_json = utils.create_output_dir_change_filename(output_dir, args.output_prefix + f'{args.theozyme_resnum}_{residue_identity}.json')
    logging.info(f'Writing output json to {filename_json}.')
    print(f'Writing output json to {filename_json}.')
    frags_info.to_json(filename_json)
    logging.info(f"Done!")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--database_dir", type=str, default="/home/tripp/riffdiff2/riff_diff/database/", help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--theozyme_pdb", type=str, required=True, help="Path to pdbfile containing theozyme, must contain all residues in chain A numbered from 1 to n, ligand must be in chain Z (if there is one).")
    argparser.add_argument("--theozyme_resnum", required=True, help="Residue number with chain information (e.g. 25A) in theozyme pdb to find fragments for.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    argparser.add_argument("--output_prefix", type=str, required=True, help="Prefix for all output files")
    argparser.add_argument("--ligand_chain", type=str, default=None, help="Chain name of your ligand chain. Set None if no ligand is present")

    # stuff you might want to adjust
    argparser.add_argument("--frag_pos_to_replace", type=int, default=[2,4], nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. 2,6 if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus!")
    argparser.add_argument("--fragsize", type=int, default=5, help="Size of output fragments.")
    argparser.add_argument("--rot_sec_struct", type=str, default=None, help="Limit fragments to secondary structure at rotamer position. Provide string of one-letter code of dssp secondary structure elements (B, E, G, H, I, T, S, -), e.g. 'HE' if rotamer should be in helices or beta strands.")
    argparser.add_argument("--frag_sec_struct_fraction", type=str, default=None, help="Limit to fragments containing at least fraction of residues with the provided secondary structure. If fragment should have at least 50 percent helical residues OR 60 percent beta-sheet, pass 'H:0.5,E:0.6'")
    argparser.add_argument("--max_frags", type=int, default=30, help="Maximum number of frags that should be returned.")
    argparser.add_argument("--rmsd_cutoff", type=float, default=1.0, help="Set minimum RMSD of output fragments. Increase to get more diverse fragments, but high values might lead to very long runtime!")
    argparser.add_argument("--score_cutoff", type=float, default=0.5, help="Maximum mean backbone score of fragments (sum of omega, rama_prepro, hbond_sr_bb, p_aa_pp Rosetta scoreterms with ref15 weights)")
    argparser.add_argument("--phipsi_occurence_cutoff", type=float, default=0.5, help="Limit how common the phi/psi combination of a certain rotamer has to be. Value is in %")
    argparser.add_argument("--covalent_bond", type=str, default=None, help="Add covalent bond(s) between rotamer and ligand in the form 'RotAtomA:LigAtomA,RotAtomB:LigAtomB'. Atom names should follow PDB numbering schemes, e.g. 'NZ:C3' for a covalent bond between a Lysine nitrogen and the third carbon atom of the ligand.")


    # stuff you probably don't want to touch
    argparser.add_argument("--limit_frags_to_res_id", type=str, default="True", help="Only pick fragments that contain the rotamer residue identity at the specified position with given phi/psi angle combination")
    argparser.add_argument("--limit_frags_to_chi", type=str, default="True", help="Only pick fragments that contain the rotamer with chi angles within the bin range of the target chi angle")
    argparser.add_argument("--phi_psi_bin", type=float, default=10, help="Binsize used to identify if fragment fits to phi/psi combination")
    argparser.add_argument("--max_phi_psis", type=int, default=5, help="maximum number of phi/psi combination that should be returned. Can be increased if not enough fragments are found downstream (e.g. because secondary structure filter was used, and there are not enough phi/psi combinations in the output that fit to the specified secondary structure.")
    argparser.add_argument("--rotamer_diff_to_best", type=float, default=0.15, help="Accept rotamers that have a probability not lower than this percentage of the most probable rotamer")
    argparser.add_argument("--flip_histidines", type=str, default="True", help="Flip the orientation of histidine residues to generate more fragment orientations (doubles number of fragments if set to true!")
    argparser.add_argument("--his_central_atom", type=str, default="auto", help="Only important if rotamer is HIS and flip_histidines is True, sets the name of the atom that should not be flipped.If auto, the histidine nitrogen closest to the ligand is the coordinating atom. Can be manually set to NE2 or ND1")
    argparser.add_argument("--flip_symmetric", type=str, default="True", help="Flip tip symmetric residues.")
    argparser.add_argument("--rotamer_ligand_clash_detection_vdw_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between rotamer and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--backbone_ligand_clash_detection_vdw_multiplier", type=float, default=1.2, help="Multiplier for VanderWaals radii for clash detection between fragment backbone and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")

    #argparser.add_argument("--prob_cutoff", type=float, default=0.0, help="Do not return any phi/psi combinations with chi angle probabilities below this value")
    
    args = argparser.parse_args()


    main(args)
