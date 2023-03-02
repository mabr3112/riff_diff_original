import Bio
from Bio.PDB import PDBIO
import Bio.PDB

def load_structure_from_pdbfile(path_to_pdb: str, all_models=False) -> Bio.PDB.Structure:
    '''AAA'''
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    if all_models: return pdb_parser.get_structure("pose", path_to_pdb)
    else: return pdb_parser.get_structure("pose", path_to_pdb)[0]

def store_pose(pose: Bio.PDB.Structure, save_path: str) -> str:
    '''Stores Bio.PDB.Structure at <save_path>'''
    io = PDBIO()
    io.set_structure(pose)
    io.save(save_path)
    return save_path

def residue_mapping_from_motif(motif_old: list, motif_new: list) -> dict:
    '''AAA'''
    return {tuple(old): tuple(new) for old, new in zip(motif_old, motif_new)}

def renumber_pose_by_residue_mapping(pose: Bio.PDB.Structure.Structure, residue_mapping: dict) -> Bio.PDB.Structure.Structure:
    '''AAA'''
    for old_res, new_res in residue_mapping.items():
        # change residue ID
        pose[old_res[0]][(" ", old_res[1], " ")].id = (" ", new_res[1], " ")

        # change Chain if Chain-names in residue mapping don't match:
        if new_res[0] != old_res[0]: pose[new_res[0]].add(pose[old_res[0]][(" ", new_res[1], " ")])
    return pose

def renumber_pdb_by_residue_mapping(pose_path: str, residue_mapping: dict, out_pdb_path=None) -> str:
    '''AAA'''
    # change numbering
    pose = load_structure_from_pdbfile(pose_path)
    pose = renumber_pose_by_residue_mapping(pose=pose, residue_mapping=residue_mapping)
    
    # save pose
    path_to_output_structure = out_pdb_path or pose_path
    store_pose(pose, path_to_output_structure)
    return path_to_output_structure
