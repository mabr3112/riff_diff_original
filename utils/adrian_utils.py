##################################### ADRIAN ###########################################################
from pathlib import Path

import numpy as np
from pathlib import Path
import Bio
from Bio.PDB import *
import os


def path_ends_with_slash(path):
    '''
    Checks if <path> ends with / and if not, adds a / to the path. Returns string
    '''
    if path.endswith('/'):
        return path
    else:
        newpath = path + '/'
        return newpath

def import_structure_from_pdb(pdbfile):
    '''
    reads in a pdbfile and returns a biopython structure object
    '''
    structname = Path(pdbfile).stem
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structname, pdbfile)
    return structure

def import_pdbs_from_dir(input_dir:str, pdb_prefix:str=None):
    '''
    imports all pdbs from a directory, returns a list of structures
    '''
    input_dir = path_ends_with_slash(input_dir)
    pdb_list = []
    for file in os.listdir(input_dir):
        if file.endswith('.pdb'):
            if pdb_prefix:
                if file.startswith(pdb_prefix):
                    pdb_list.append(f'{input_dir}{file}')
            else:
                pdb_list.append(f'{input_dir}{file}')
    struct_list = []
    for pdb in sorted(pdb_list):
        struct = import_structure_from_pdb(pdb)
        struct_list.append(struct)
    return struct_list

def write_multimodel_structure_to_pdb(structure, filename):
    '''
    saves a pdb of the structure containing multiple models to disk
    '''
    for model in structure:
        model.serial_num = model.id
    io = PDBIO(use_model_flag=True)
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        io.set_structure(structure)
        io.save(f)

def create_output_dir_change_filename(output_dir, filename):
    if output_dir:
        output_dir = path_ends_with_slash(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        filename = output_dir + filename
    else:
        filename = filename
    return filename
