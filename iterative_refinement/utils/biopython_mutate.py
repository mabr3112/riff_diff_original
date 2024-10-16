# Mutation Tools for BioPython

import Bio
import Bio.PDB
from Bio.PDB.PDBIO import PDBIO

def mutate_residue(pose: Bio.PDB.Structure, chain:str, res_id: int, mutation: str) -> Bio.PDB.Structure:
    '''
    chain: chain (letter) as string
    res_id: Residue in PDB numbering
    mutation: three-letter AA code as string
    '''
    pose[chain][(" ", res_id, " ")].resname = mutation
    return pose

def load_structure_from_pdbfile(path_to_pdb: str, model_number:int=0, all_models=False) -> Bio.PDB.Structure:
    '''AAA'''
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    if all_models: return pdb_parser.get_structure("pose", path_to_pdb)
    else: return pdb_parser.get_structure("pose", path_to_pdb)[model_number]

def store_pose(pose: Bio.PDB.Structure, save_path: str) -> str:
    '''Stores Bio.PDB.Structure at <save_path>'''
    io = PDBIO()
    io.set_structure(pose)
    io.save(save_path)
    return save_path

def mutate_pose_residues(pose: Bio.PDB.Structure, mutation_dict: dict) -> Bio.PDB.Structure:
    '''
    requires dict with mutations listed like this:
        {"A10": "ALA", "B12": "CYS", "C3": "PHE"}
    '''
    for res, mutation in mutation_dict.items():
        pose = mutate_residue(pose, res[0], int(res[1:]), mutation)
    return pose

def mutate_pdb(pdb_path: str, mutation_dict: dict, out_pdb_path:str=None, biopython_model_number:int=0) -> str:
    '''AAA'''
    # set output path if option was set, otherwise overwrite pdb_path:
    out_path = out_pdb_path or pdb_path

    # mutate
    pose = mutate_pose_residues(load_structure_from_pdbfile(pdb_path, model_number=biopython_model_number), mutation_dict=mutation_dict)

    # store
    return store_pose(pose, out_path)
