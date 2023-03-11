import Bio
from Bio.PDB import PDBIO
import Bio.PDB
import copy

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
    # deepcopy pose and detach all residues from chains.
    out_pose = copy.deepcopy(pose)
    ch = [chain.id for chain in out_pose.get_chains()]
    for chain in ch:
        residues = [res.id for res in out_pose[chain].get_residues()]
        [out_pose[chain].detach_child(resi) for resi in residues]

    # collect residues with renumbered ids and chains into one list:
    for old_res, new_res in residue_mapping.items():
        # remove old residue from original pose
        res = pose[old_res[0]][(" ", old_res[1], " ")]
        pose[old_res[0]].detach_child((" ", old_res[1], " "))
        res.detach_parent()

        # set new residue ID
        res.id = (" ", new_res[1], " ")

        # add to appropriate chain (residue mapping) in out_pose
        out_pose[new_res[0]].add(res)

    # remove chains from pose that are empty:
    chain_ids = [x.id for x in out_pose] # for some reason, iterating over chains in struct directly does not work here...
    for chain_id in chain_ids:
        if not out_pose[chain_id].__dict__["child_dict"]: out_pose.detach_child(chain_id)

    return out_pose

def renumber_pdb_by_residue_mapping(pose_path: str, residue_mapping: dict, out_pdb_path=None) -> str:
    '''AAA'''
    # change numbering
    pose = load_structure_from_pdbfile(pose_path)
    pose = renumber_pose_by_residue_mapping(pose=pose, residue_mapping=residue_mapping)
    
    # save pose
    path_to_output_structure = out_pdb_path or pose_path
    store_pose(pose, path_to_output_structure)
    return path_to_output_structure
