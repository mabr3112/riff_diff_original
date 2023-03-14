#!/usr/bin/env python3

#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

# import custom modules
#sys.path.append("/home/tripp/riff_diff/")

import utils.adrian_utils as utils

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

def import_rotamer_library(library_path:str):
    '''
    reads in a Rosetta rotamer library, drops everything that is not needed
    '''
    library = pd.read_csv(library_path, skiprows=36, delim_whitespace=True, header=None)
    library = library.drop(library.columns[[4, 5, 6, 7, 9]], axis=1)
    library.columns = ["identity", "phi", "psi", "count", "probability", "chi1", "chi2", "chi3", "chi4", "chi1sig", "chi2sig", "chi3sig", "chi4sig"]
    return library

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    library_folder = utils.path_ends_with_slash(library_folder)
    prefix = residue_identity.lower()
    rotlib = import_rotamer_library(f'{library_folder}{prefix}.bbdep.rotamers.lib')
    if residue_identity in AAs_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    return rotlib

def identify_rotamers_suitable_for_backbone(residue_identity:str, phi:float, psi:float, rotlib:pd.DataFrame(), prob_cutoff:float=None, max_rotamers:int=None, max_stdev:float=2, level:int=3):
    '''
    identifies suitable rotamers by filtering for phi/psi angles
    if fraction is given, returns only the top rotamer fraction ranked by probability (otherwise returns all rotamers)
    if prob_cutoff is given, returns only rotamers more common than prob_cutoff
    '''
    #round dihedrals to the next tens place
    if not phi == None:
        phi = round(phi, -1)
    if not psi == None:
        psi = round(psi, -1)
    #extract all rows containing specified phi/psi angles from library
    if phi and psi:
        rotlib = rotlib.loc[(rotlib['phi'] == phi) & (rotlib['psi'] == psi)].reset_index(drop=True)
    elif not phi or not psi:
        if not phi:
            rotlib = rotlib[rotlib['psi'] == psi].reset_index(drop=True)
        elif not psi:
            rotlib = rotlib[rotlib['phi'] == phi].reset_index(drop=True)
        rotlib = rotlib.loc[rotlib['count'] >= 100]
        rotlib = rotlib.drop_duplicates(subset=['phi', 'psi'], keep='first')
        rotlib.sort_values("probability", ascending=False)
        rotlib = rotlib.head(5)
    #filter top rotamers
    rotlib = rotlib.sort_values("probability", ascending=False)
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if level > 0:
        rotlib = diversify_chi_angles(rotlib, max_stdev, level)
        #filter again, since diversify_chi_angles produces rotamers with lower probability
        if prob_cutoff:
            rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if max_rotamers:
        rotlib = rotlib.head(max_rotamers)
    return rotlib

def diversify_chi_angles(rotlib, max_stdev:float=2, level:int=3):
    '''
    adds additional chi angles based on standard deviation.
    max_stdev: defines how far to stray from mean based on stdev. chi_new = chi_orig +- stdev * max_stdev
    level: defines how many chis should be sampled within max_stdev. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned
    '''
    #check which chi angles exist in rotamer library
    columns = list(rotlib.columns)
    columns = [column for column in columns if column.startswith('chi') and not 'sig' in column]
    #generate deviation parameters
    devs = [max_stdev * i / level for i in range(-level, level +1)]
    #calculate chi angles
    for chi_angle in columns:
        new_chis_list = []
        for dev in devs:
            new_chis = alter_chi(rotlib, chi_angle, f'{chi_angle}sig', dev)
            new_chis_list.append(new_chis)
        rotlib = pd.concat(new_chis_list)
        rotlib.drop([f'{chi_angle}sig'], axis=1, inplace=True)
        rotlib[chi_angle] = round(rotlib[chi_angle], 1)
    rotlib.sort_values('probability', inplace=True, ascending=False)
    rotlib.reset_index(drop=True, inplace=True)
    return rotlib

def normal_dist_density(x):
    '''
    calculates y value for normal distribution from distance from mean TODO: check if it actually makes sense to do it this way
    '''
    y = math.e **(-(x)**2 / 2)
    return y

def alter_chi(rotlib, chi_column, chi_sig_column, dev):
    '''
    calculate deviations from input chi angle for rotamer library
    '''
    new_chis = copy.deepcopy(rotlib)
    new_chis[chi_column] = new_chis[chi_column] + new_chis[chi_sig_column] * dev
    new_chis['probability'] = new_chis['probability'] * normal_dist_density(dev)
    return new_chis


def mutate_bb_res_to_theozyme_rotamer(structure, AAalphabet_structure, residue_position:int, phi:float, psi:float, omega:float, carb_angle:float, tau:float, N_CA_length:float, CA_C_length:float, C_O_length:float, rotlib:pd.DataFrame(), output:str):
    '''
    mutates the given residue position of the backbone fragment to all rotamers provided in the rotlib dataframe
    '''

    #define residue to mutate & detach it
    to_mutate = structure[0]["A"][residue_position]
    structure[0]["A"].detach_child(to_mutate.id)

    # this is disgusting, find a better way
    columns = list(rotlib.columns)
    if 'chi1' in columns:
        chi1_list = rotlib['chi1'].tolist()
        chi1_list = [None if math.isnan(angle) else angle for angle in chi1_list]
    else:
        chi1_list = [None for i in range(0, len(rotlib.index))]
    if 'chi2' in columns:
        chi2_list = rotlib['chi2'].tolist()
        chi2_list = [None if math.isnan(angle) else angle for angle in chi2_list]
    else:
        chi2_list = [None for i in range(0, len(rotlib.index))]
    if 'chi3' in columns:
        chi3_list = rotlib['chi3'].tolist()
        chi3_list = [None if math.isnan(angle) else angle for angle in chi3_list]
    else:
        chi3_list = [None for i in range(0, len(rotlib.index))]
    if 'chi4' in columns:
        chi4_list = rotlib['chi4'].tolist()
        chi4_list = [None if math.isnan(angle) else angle for angle in chi4_list]
    else:
        chi4_list = [None for i in range(0, len(rotlib.index))]

    problist = rotlib['probability'].tolist()
    identity_list = rotlib['identity'].tolist()

    model_num = 0
    rotamers_on_backbone = Structure.Structure("rot_on_bb")


    for index in range(0, len(rotlib.index)):
        model = copy.deepcopy(structure[0])
        model.id = model_num
        residue_identity = identity_list[index]
        #TODO: find a better way than replacing it with a residue from a random structure, but directly add atoms --> would make the whole translation & insertion step obsolete
        res = generate_rotamer(AAalphabet_structure, residue_identity, to_mutate.id, phi, psi, omega, carb_angle, tau, N_CA_length, CA_C_length, C_O_length, chi1_list[index], chi2_list[index], chi3_list[index], chi4_list[index], rot_probability=problist[index])
        #for some reason deepcopy does not work if internal coords are present
        delattr(res, 'internal_coord')
        #superposition the generated rotamer with the backbone fragment
        to_mutate_atoms = []
        res_atoms = []
        for atom in ["N", "CA", "C"]:
            to_mutate_atoms.append(to_mutate[atom])
            res_atoms.append(res[atom])
        sup = Bio.PDB.Superimposer()
        sup.set_atoms(to_mutate_atoms, res_atoms)
        sup.rotran
        sup.apply(res)
        #insert the rotamer
        model["A"].insert(res.id[1] - 1, res)
        rotamers_on_backbone.add(model)
        model_num = model_num + 1

    if output:
        utils.write_multimodel_structure_to_pdb(rotamers_on_backbone, output)

    return rotamers_on_backbone

def atoms_for_func_group_alignment(residue):
    '''
    return the atoms used for superposition via functional groups
    '''
    sc_residue_identity = residue.get_resname()
    if sc_residue_identity == "ALA":
        atoms = ["CB", "CA", "N"]
    elif sc_residue_identity == "ARG":
        atoms = ["NH1", "NH2", "CZ"]
    elif sc_residue_identity == "ASP":
        atoms = ["OD1", "OD2", "CG"]
    elif sc_residue_identity == "ASN":
        atoms = ["OD1", "ND2", "CG"]
    elif sc_residue_identity == "CYS":
        atoms = ["SG", "CB", "CA"]
    elif sc_residue_identity == "GLU":
        atoms = ["OE1", "OE2", "CD"]
    elif sc_residue_identity == "GLN":
        atoms = ["OE1", "NE2", "CD"]
    elif sc_residue_identity == "GLY":
        atoms = ["CA", "N", "C"]
    elif sc_residue_identity == "HIS":
        atoms = ["ND1", "NE2", "CG"]
    elif sc_residue_identity == "ILE":
        atoms = ["CD1", "CG1", "CB"]
    elif sc_residue_identity == "LEU":
        atoms = ["CD1", "CD2", "CG"]
    elif sc_residue_identity == "LYS":
        atoms = ["NZ", "CE", "CD"]
    elif sc_residue_identity == "MET":
        atoms = ["CE", "SD", "CG"]
    elif sc_residue_identity == "PHE":
        atoms = ["CD1", "CD2", "CZ"]
    elif sc_residue_identity == "PRO":
        atoms = ["CD", "CG", "CB"]
    elif sc_residue_identity == "SER":
        atoms = ["OG", "CB", "CA"]
    elif sc_residue_identity == "THR":
        atoms = ["OG1", "CG2", "CB"]
    elif sc_residue_identity == "TRP":
        atoms = ["NE1", "CZ3", "CG"]
    elif sc_residue_identity == "TYR":
        atoms = ["CE1", "CE2", "OH"]
    elif sc_residue_identity == "THR":
        atoms = ["CG1", "CG2", "CB"]
    else:
        raise RuntimeError(f'Unknown residue with name {sc_residue_identity}!')
    res_atoms = []
    for atom in atoms:
        res_atoms.append(residue[atom])
    return res_atoms

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True, flip_histidines:bool=False, his_central_atom:str="NE2"):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()
    bbf_residue_identity = entity_residue_to_align.get_resname()
    if flip_histidines == True:
        if not his_central_atom in ["NE2", "ND1"]:
            raise KeyError(f'his_central_atom must be either NE2 or ND1, not {his_central_atom}')

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric == True and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    #flip the orientation of His residues, flip direction depends on the orientation of the coordinating atom
    if sc_residue_identity == "HIS" and flip_histidines == True:
        if his_central_atom == "NE2":
            bbf_atoms = [entity_residue_to_align["NE2"], entity_residue_to_align["ND1"], entity_residue_to_align["CG"]]
            sc_atoms = [sidechain["NE2"], sidechain["CG"], sidechain["ND1"]]
        if his_central_atom == "ND1":
            bbf_atoms = [entity_residue_to_align["ND1"], entity_residue_to_align["NE2"], entity_residue_to_align["CD2"]]
            sc_atoms = [sidechain["ND1"], sidechain["CD2"], sidechain["NE2"]]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    sup.rotran
    sup.apply(entity)

    return entity

def tip_symmetric_residues():
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def identify_residues_with_equivalent_func_groups(residue):
    '''
    checks if residues with same functional groups exist, returns a list of these residues
    '''
    resname = residue.get_resname()
    if resname in ['ASP', 'GLU']:
        return ['ASP', 'GLU']
    elif resname in ['ASN', 'GLN']:
        return ['ASN', 'GLN']
    elif resname in ['VAL', 'ILE']:
        return ['VAL', 'ILE']
    else:
        return [resname]


def rotamers_for_backbone(resnames, rotlib_path, phi, psi, rot_prob_cutoff:float=0.05, max_rotamers:int=70, max_stdev:float=2, level:int=2):
    rotlib_list = []
    for res in resnames:
        if res in ["ALA", "GLY"]:
            rotlib = pd.DataFrame([[res, phi, psi, float("nan"), 1]], columns=["identity", "phi", "psi", "count", "probability"])
            rotlib_list.append(rotlib)
        else:
            rotlib = return_residue_rotamer_library(rotlib_path, res)
            rotlib_list.append(identify_rotamers_suitable_for_backbone(res, phi, psi, rotlib, rot_prob_cutoff, max_rotamers, max_stdev, level))
    if len(rotlib_list) > 1:
        filtered_rotlib = pd.concat(rotlib_list)
        filtered_rotlib.reset_index(drop=True, inplace=True)
        return filtered_rotlib
    else:
        return rotlib_list[0]

def generate_backbones_for_residue(output_dir, output_prefix, theozyme, theozyme_residue, resnames, backbone, AA_alphabet, rotlib_path, backbone_angles, backbone_bondlengths, rotamer_fraction:float=None, rot_prob_cutoff:float=0.05, max_rotamers:int=70, max_stdev:float=2, level:int=3, frag_pos_to_replace:int=4, rot_on_bb_output:str=None, flip_symmetric:bool=True, flip_histidines:bool=False, his_central_atom:str="NE2"):

    backbone = copy.deepcopy(backbone)
    AA_alphabet = copy.deepcopy(AA_alphabet)
    resnum = theozyme_residue.id[1]
    filtered_rotlib = rotamers_for_backbone(resnames, rotlib_path, backbone_angles["phi"], backbone_angles["psi"], rot_prob_cutoff, max_rotamers, max_stdev, level)
    #display(filtered_rotlib)
    filename = utils.create_output_dir_change_filename(output_dir, f'{output_prefix}{resnum}_{theozyme_residue.get_resname()}_filtered_rotlib.csv')
    filtered_rotlib["rotamer_position"] = frag_pos_to_replace
    cols = filtered_rotlib.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    filtered_rotlib = filtered_rotlib[cols]
    if os.path.exists(filename):
        header = False
    else:
        header = True
    filtered_rotlib.to_csv(filename, mode='a', header=header, index=False)
    rot_on_bb = mutate_bb_res_to_theozyme_rotamer(backbone, AA_alphabet, frag_pos_to_replace, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], filtered_rotlib, rot_on_bb_output)
    collect_rotamer_bbfs = Structure.Structure('rotbbfs')
    model_num = 0
    for model in rot_on_bb.get_models():
        model.detach_parent()
        out = align_to_sidechain(model, model["A"][frag_pos_to_replace], theozyme_residue, False, False)
        out.id = model_num
        collect_rotamer_bbfs.add(out)
        model_num = model_num + 1
        if model["A"][frag_pos_to_replace].get_resname() in tip_symmetric_residues() and flip_symmetric == True:
            flipped = copy.deepcopy(out)
            flipped.id = model_num
            flipped = align_to_sidechain(flipped, flipped["A"][frag_pos_to_replace], theozyme_residue, flip_symmetric, False)
            collect_rotamer_bbfs.add(flipped)
            model_num = model_num + 1
        if model["A"][frag_pos_to_replace].get_resname() == "HIS" and flip_histidines == True:
            flipped = copy.deepcopy(out)
            flipped.id = model_num
            flipped = align_to_sidechain(flipped, flipped["A"][frag_pos_to_replace], theozyme_residue, False, flip_histidines, his_central_atom)
            collect_rotamer_bbfs.add(flipped)
            model_num = model_num + 1
    return collect_rotamer_bbfs

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

def generate_rotamer(AAalphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None, switch_symmetric:bool=False):
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


def main(args):
    '''
    aligns a provided backbone fragment to all rotamers for a given theozyme.
    backbone_fragment: path to pdbfile of backbone fragment
    frag_pos_to_replace: position on the backbone fragment the rotamer is inserted.
    theozyme_pdb: path to pdbfile of theozyme. Individual residues have to be on chain A and not share residue numbers! If copy_ligand is set to true, theozymepdb should contain ligand on chain Z!
    theozyme_resnum: either 'all' or the theozyme residue number a fragment should be built for
    rotlib_folder: path to folder containing bb_dependent rotamers + pdbfile containing all amino acids
    output_prefix: prefix used for final output pdbs
    rotamer_fraction: only accept top X of rotamers ranked by probability
    rot_prob_cutoff: filter rotamers based on probability. Might lead to very few rotamers!
    rot_on_bb_output: if provided a filename, write a pdbfile containing all rotamers on the provided backbone to filename
    '''
    backbone = utils.import_structure_from_pdb(args.fragment_pdb)
    database_dir = utils.path_ends_with_slash(args.database_dir)
    AA_alphabet = utils.import_structure_from_pdb(f'{database_dir}AA_alphabet.pdb')
    theozyme = utils.import_structure_from_pdb(args.theozyme_pdb)
    output_dir = utils.path_ends_with_slash(args.output_dir)

    if args.copy_ligand == True:
        #check if ligand exists
        if not "Z" in [chain.id for chain in theozyme.get_chains()]:
            raise RuntimeError('No ligand found in chain Z. Please make sure the theozyme pdb is correctly formatted')
        ligand = theozyme[0]["Z"]
    else:
        ligand = None

    if len(args.frag_pos_to_replace) > 1:
        frag_pos_to_replace = [i for i in range(args.frag_pos_to_replace[0], args.frag_pos_to_replace[1]+1)]
    else:
        frag_pos_to_replace = args.frag_pos_to_replace


    model_num = 0
    output = Structure.Structure("out")
    theozyme_residue = theozyme[0]['A'][args.theozyme_resnum]

    if args.add_equivalent_func_groups == True:
        rotamer_residues = identify_residues_with_equivalent_func_groups(theozyme_residue)
    else:
        rotamer_residues = [theozyme_residue.get_resname()]

    for pos in frag_pos_to_replace:
        backbone_angles = extract_backbone_angles(backbone[0]["A"], pos)
        backbone_bondlengths = extract_backbone_bondlengths(backbone[0]["A"], pos)
        collect_rotamer_bbfs = generate_backbones_for_residue(output_dir, args.output_prefix, theozyme, theozyme_residue, rotamer_residues, backbone, AA_alphabet, database_dir, backbone_angles, backbone_bondlengths, None, args.rot_prob_cutoff, args.max_rotamers, args.max_stdev, args.level, pos, args.rot_on_bb_output, args.flip_symmetric, args.flip_histidines, args.his_central_atom)
        for model in collect_rotamer_bbfs:
            model.id = model_num
            if ligand:
                model.add(ligand)
            output.add(model)
            model_num = model_num + 1

    filename = utils.create_output_dir_change_filename(output_dir, f'{args.output_prefix}{args.theozyme_resnum}_{theozyme_residue.get_resname()}.pdb')
    utils.write_multimodel_structure_to_pdb(output, filename)

    return


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--fragment_pdb", type=str, required=True, help="Path to backbone fragment pdb")
    argparser.add_argument("--database_dir", type=str, required=True, help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--theozyme_pdb", type=str, required=True, help="Path to pdbfile containing theozyme, must contain all residues in chain A numbered from 1 to n, ligand must be in chain Z (if there is one).")
    argparser.add_argument("--theozyme_resnum", type=int, required=True, help="Residue number in theozyme pdb to find fragments for.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    argparser.add_argument("--output_prefix", type=str, required=True, help="Prefix for all output files")

    # stuff you might want to adjust
    argparser.add_argument("--frag_pos_to_replace", type=int, default=[2,6], nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. [2, 6] if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus!")
    argparser.add_argument("--rot_prob_cutoff", type=float, default=0.08, help="Rotamer probabilities must be above this threshold to be accepted")
    argparser.add_argument("--max_stdev", type=float, default=2, help="Range for sampling within chi angle bin in standard deviations.")
    argparser.add_argument("--level", type=int, default=2, help="Defines how many chis should be sampled within max_stdev of chi angle bin. if level = 0, only mean value will be returned. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned.")
    argparser.add_argument("--max_rotamers", type=int, default=12, help="Maximum number of rotamers per frag_pos and residue identity that should be returned (does not necessarily equal number of output models!")
    argparser.add_argument("--his_central_atom", type=str, default="NE2", help="Only important if rotamer is HIS and flip_histidines is True, sets the name of the atom that should not be flipped. Has to be either NE2 or ND1")

    # stuff you probably don't want to touch
    argparser.add_argument("--copy_ligand", type=bool, default=True, help="Copy ligand to output pdb (only works if ligand is present in theozyme chain Z!")
    argparser.add_argument("--flip_symmetric", type=bool, default=True, help="Flip tip symmetric residues (doubles number of fragments if set to true!")
    argparser.add_argument("--flip_histidines", type=bool, default=True, help="Flip the orientation of histidine residues to generate more fragment orientations (doubles number of fragments if set to true!")
    argparser.add_argument("--add_equivalent_func_groups", type=bool, default=True, help="use ASP/GLU, GLN/ASN and VAL/ILE interchangeably")
    argparser.add_argument("--rot_on_bb_output", type=str, default=None, help="Write fragments to disk before superpositioning with sidechain, mainly for testing purposes")
    args = argparser.parse_args()

    main(args)
