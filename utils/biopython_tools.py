from json import load
import Bio
from Bio.PDB import PDBIO
import Bio.PDB
import copy
import numpy as np
from collections import defaultdict

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

def renumber_pose_by_residue_mapping(pose: Bio.PDB.Structure.Structure, residue_mapping: dict, keep_chain:str="") -> Bio.PDB.Structure.Structure:
    '''AAA'''
    # deepcopy pose and detach all residues from chains.
    out_pose = copy.deepcopy(pose)
    ch = [chain.id for chain in out_pose.get_chains() if chain.id != keep_chain]
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

def renumber_pdb_by_residue_mapping(pose_path: str, residue_mapping: dict, out_pdb_path=None, keep_chain:str="") -> str:
    '''AAA'''
    # change numbering
    pose = load_structure_from_pdbfile(pose_path)
    pose = renumber_pose_by_residue_mapping(pose=pose, residue_mapping=residue_mapping, keep_chain=keep_chain)
    
    # save pose
    path_to_output_structure = out_pdb_path or pose_path
    store_pose(pose, path_to_output_structure)
    return path_to_output_structure

def replace_motif_in_pose(pose_path: str, motif_path: str, pose_motif: dict, ref_motif: dict, new_pose:str=None) -> Bio.PDB.Structure:
    '''Replaces motif in a pose.
    
    #### TODO (itref refactoring): frame pose and ref motifs as lists. Dictionaries do not allow mixing of chains as a motif.
    '''
    # load pose and motif
    pose = load_structure_from_pdbfile(pose_path, all_models=False)
    ref_motif_pose = load_structure_from_pdbfile(motif_path, all_models=False)

    # superimpose pose onto ref_motif based on motif residues.
    pose_superimposed = superimpose_poses_by_motif(pose, ref_motif_pose, pose_motif, ref_motif, atoms=["CA"])

    # replace
    pose = replace_motif(pose, ref_motif_pose, pose_motif, ref_motif)

    # save
    savepath = new_pose or pose_path
    return store_pose(pose, savepath)

def get_residues_of_motif(pose: Bio.PDB.Structure, motif: dict) -> list:
    '''returns motif or pose'''
    return [pose[chain][(" ", res, " ")] for chain in motif for res in motif[chain]]

def replace_motif(pose: Bio.PDB.Structure.Structure, replacement_pose: Bio.PDB.Structure.Structure, pose_motif: dict, replacement_motif: dict) -> Bio.PDB.Structure.Structure:
    """Replace a motif in a pose with a motif from another pose.
    
    This function takes as input two `Pose` objects (`pose` and `replacement_pose`), the motif to be replaced in `pose` (`pose_motif`), and the replacement motif in `replacement_pose` (`replacement_motif`). It replaces the residues in `pose_motif` with the residues in `replacement_motif`, and returns the modified `pose`.
    
    :param pose: The pose containing the motif to be replaced.
    :type pose: Pose
    :param replacement_pose: The pose containing the replacement motif.
    :type replacement_pose: Pose
    :param pose_motif: The motif to be replaced in `pose`.
    :type pose_motif: str
    :param replacement_motif: The replacement motif in `replacement_pose`.
    :type replacement_motif: str
    :return: The modified `pose`.
    :rtype: Pose
    """    
    repl_residues = get_residues_of_motif(replacement_pose, replacement_motif)
    pose_residues = get_residues_of_motif(pose, pose_motif)

    for pose_res, repl_res in zip(pose_residues, repl_residues):
        old_chain = pose_res.get_parent().id
        # remove old residue from pose
        pose[old_chain].detach_child(pose_res.id)

        # add new residue
        pose[old_chain].insert(repl_res.id[1]-1, repl_res)

    return pose

def superimpose_poses_by_motif(mobile_pose: Bio.PDB.Structure.Structure, target_pose: Bio.PDB.Structure.Structure, mobile_motif: dict, target_motif: dict, atoms=["CA"]) -> Bio.PDB.Structure.Structure:
    '''
    Superimpose two structures based on residue selections. Sensitive to the order in the selection!
    Returns RMSD after superimposition.
    Args:
        <ref_pdbfile>                Path to the directory containing the reference pdb
        <target_pdbfile>             Path to the directory containing the target pdb
        <ref_selection>              Dictionary specifying the reference motif residues: {"chain": [res, ...], ...}
        <target_selection>           Dictionary specifying the target motif residues: {"chain": [res, ...], ...} Should be same number of residues as <ref_motif> 
        <atoms>                      List of atoms for which to calculate RMSD with
    '''
    def get_atoms_of_motif(pose: Bio.PDB.Structure.Structure, motif:dict, atm_list:list=["CA"]) -> list:
        atms = []
        for chain in motif:
            for res in motif[chain]:
                [atms.append(pose[chain][(" ", res, " ")][x]) for x in atoms]
        return atms
    
    # Make a list of the atoms (in the structures) you wish to align.
    mobile_atoms = get_atoms_of_motif(mobile_pose, mobile_motif, atm_list=atoms)
    target_atoms = get_atoms_of_motif(target_pose, target_motif, atm_list=atoms)

    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(target_atoms, mobile_atoms)

    # Apply superimposer on mobile pose:
    super_imposer.apply(mobile_pose.get_atoms())
    return mobile_pose

def replace_motif_and_add_ligand(pose_path: str, motif_path: str, pose_motif: dict, ref_motif: dict, new_pose:str=None, ligand_chain:str="Z") -> str:
    '''Replaces motif in a pose.
    
    #### TODO (itref refactoring): frame pose and ref motifs as lists. Dictionaries do not allow mixing of chains as a motif.
    '''
    # load pose and motif
    pose = load_structure_from_pdbfile(pose_path, all_models=False)
    ref_motif_pose = load_structure_from_pdbfile(motif_path, all_models=False)

    # superimpose pose onto ref_motif based on motif residues.
    pose_superimposed = superimpose_poses_by_motif(pose, ref_motif_pose, pose_motif, ref_motif, atoms=["CA"])

    # replace
    pose = replace_motif(pose, ref_motif_pose, pose_motif, ref_motif)

    # add ligand chain if it is not present already:
    if not ligand_chain in [chain.id for chain in pose]:
        pose.add(ref_motif_pose[ligand_chain])

    # save
    savepath = new_pose or pose_path
    return store_pose(pose, savepath)


def get_protein_atoms(pose: Bio.PDB.Structure.Structure, ligand_chain:str=None, atms:list=None) -> list:
    '''Selects atoms from a pose object. If ligand_chain is given, excludes all atoms in ligand_chain'''
    # define chains of pose
    chains = [x.id for x in pose.get_chains()]
    print("chains", chains)
    if ligand_chain: chains.remove(ligand_chain)
    print("after 'removal'", chains)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    print("within function", pose_atoms)
    if atms: pose_atoms = [atom for atom in pose_atoms if atom.id in atms]
    print("within function2", pose_atoms)

    return pose_atoms

def get_atoms_of_motif(pose: Bio.PDB.Structure, motif: dict, atoms:list=None, forbidden_atoms:list=["H", "NE1", "OXT"]) -> list:
    '''
    Extract a list of atoms from a PDB structure based on a motif dictionary.
    The motif dictionary is a nested dictionary where the keys are the IDs of the chains in the PDB structure, and the values are lists of residue numbers. The function will return a list of atoms from the specified residues in the specified chains.

    Args:
        structure (Bio.PDB.Structure): The PDB structure from which to extract atoms.
        motif (dict): A nested dictionary specifying the chains and residues of the atoms to extract.
        atoms (list, optional): A list of atom names to extract. If not provided, all atoms in the specified residues will be extracted.

    Returns:
        list: A list of atoms from the specified residues in the specified chains.

    Examples:
        get_atoms_of_motif(structure, {"A": [10, 20, 30], "B": [15]})
        get_atoms_of_motif(structure, {"A": [10, 20, 30], "B": [15]}, atoms=["CA", "N", "C", "O"])
    '''
    # instantiate the list of atoms that will be returned
    atms_list = []

    # iterate through all chains in the motif dictionary
    for chain in motif:
        # iterate through all residues in each chain
        for resi in motif[chain]:
            if atoms:
                # if atom list is specified, only add specified atoms of residue:
                [atms_list.append(pose[chain][(' ', resi, ' ')][atm]) for atm in atoms]
            else:
                # otherwise add all atoms (including sidechain atoms) of residue:
                [atms_list.append(atm) for atm in pose[chain][(' ', resi, ' ')].get_atoms()]

    # remove atoms that might produce errors when running superimposition (hydrogens, terminal atoms):
    return [atm for atm in atms_list if atm.name not in forbidden_atoms]

def concat_motifs(motif_list: list[dict]) -> dict:
    '''AAA'''
    def collapse_dict_values(in_dict: dict) -> set:
        return set([f"{chain}{str(res)}" for chain in in_dict for res in list(in_dict[chain])])
    # convert motifs in motif_list from dict to list:
    motif_list = [collapse_dict_values(motif_dict) for motif_dict in motif_list]

    # concatenate all motifs (exclude uniques)
    concat_motif_list = list(set([x for motif in motif_list for x in motif]))
    out_dict = defaultdict(list)
    for res in concat_motif_list:
        out_dict[res[0]].append(int(res[1:]))
    return dict(out_dict)

def select_ligand_contacts(pose: Bio.PDB.Structure.Structure, ligand_chain: str, dist:float=3.5, pose_sidechains_only=True) -> dict:
    '''Selects residues that are close to a ligand, given by ligand chain'''
    # extract coordinates of all pose atoms
    lig_coords = np.array([x.coord for x in pose[ligand_chain].get_atoms()])
    pose_atoms = [x for x in get_protein_atoms(pose, ligand_chain=ligand_chain)]
    if pose_sidechains_only: [atom for atom in pose_atoms if atom.id not in ["N", "CA", "C", "O"]]

    pose_coords = np.array([x.coord for x in pose_atoms])

    # generate mask for pose atoms where pose atoms are within :dist:
    dists = np.linalg.norm(pose_coords[:, None] - lig_coords[None], axis=-1)
    mask = np.any(dists <= dist, axis=-1)

    # apply the mask to the pose_atoms and collect the parent (residue) and grandparent(chain) id into a list
    close_contacts = list(set([(atom.get_parent().get_id()[1], atom.get_parent().get_parent().get_id()) for atom in np.array(pose_atoms)[mask]]))

    # convert into a motif dict and return
    motif_dict = {}
    for res_id, chain_id in close_contacts:
        if chain_id not in motif_dict:
            motif_dict[chain_id] = []
        motif_dict[chain_id].append(res_id)

    return motif_dict

def select_motif_centroid_contacts(pose: Bio.PDB.Structure.Structure, motif:dict, dist:float, pose_sidechains_only:bool=True):
    '''Selects residues that are close to the center of mass of an input motif.'''
    # extract pose coords
    pose_atoms = [x for x in get_protein_atoms(pose, atms=["CA"])]
    print(pose_atoms)
    if pose_sidechains_only: pose_atoms = [atom for atom in pose_atoms if atom.id in ["N", "CA", "C", "O"]]
    pose_coords = np.array([atom.get_coord() for atom in pose_atoms])
    print(pose_coords)

    # calculate centroid of all motif atoms:
    motif_atoms = get_atoms_of_motif(pose, motif=motif)
    motif_coords = np.array([atom.coord for atom in motif_atoms])
    motif_centroid = np.mean(motif_coords, axis=0)

    # create mask for pose_atoms where the distance to the motif centroid is less than :dist:
    dists = np.linalg.norm(pose_coords - motif_centroid, axis=-1)
    mask = dists <= dist

    # apply mask to to the pose_atoms and collect the parent (residue) and grandparent(chain) id into a list
    close_contacts = list(set([(atom.get_parent().get_id()[1], atom.get_parent().get_parent().get_id()) for atom in np.array(pose_atoms)[mask]]))

    # convert inot a motif dict and return
 # convert into a motif dict and return
    motif_dict = {}
    for res_id, chain_id in close_contacts:
        if chain_id not in motif_dict:
            motif_dict[chain_id] = []
        motif_dict[chain_id].append(res_id)

    return motif_dict


