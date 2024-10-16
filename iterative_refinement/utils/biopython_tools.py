from os import rename
import Bio
import Bio.PDB
from Bio.PDB import PDBIO
import numpy as np
from collections import defaultdict

from torch import concat

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

def rename_pdb_chains(pose: Bio.PDB.Structure, ref_pose: Bio.PDB.Structure) -> str:
    '''AAA'''
    # load chains and check if they have the same length
    chains = [chain.id for chain in pose]
    ref_chains = [chain.id for chain in ref_pose]
    if len(chains) != len(ref_chains): raise ValueError(f"Pose and ref_pose do not have same number of chains. \npose: {', '.join(chains)}\nref_pose: {', '.join(ref_chains)}")
    if chains == ref_chains: return pose

    # compile renaming dict:
    renaming_dict = {chain: ref_chain for chain, ref_chain in zip(chains, ref_chains)}

    for chain in pose:
        #print(chain.id)
        chain.id = renaming_dict[chain.id]

    return pose

def rename_pdb_chains_pdbfile(pdb_path: str, ref_pdb_path: str, out_path:str=None) -> str:
    '''AAA'''
    # set out_path
    out_path = out_path or pdb_path

    # parse poses
    pose = load_structure_from_pdbfile(pdb_path)
    ref_pose = load_structure_from_pdbfile(ref_pdb_path)

    # renumber pose by using ref_pose as reference:
    pose = rename_pdb_chains(pose, ref_pose)

    # store pose with renamed chains at out_path:
    return store_pose(pose, out_path)

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
    pose_atoms = [x for x in get_protein_atoms(pose)]
    if pose_sidechains_only: pose_atoms = [atom for atom in pose_atoms if not atom.id in ["N", "CA", "C", "O"]]
    pose_coords = np.array([atom.coord for atom in pose_atoms])

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

def get_protein_atoms(pose: Bio.PDB.Structure.Structure, ligand_chain:str=None, atms:list=None) -> list:
    '''Selects atoms from a pose object. If ligand_chain is given, excludes all atoms in ligand_chain'''
    # define chains of pose
    chains = [x.id for x in pose.get_chains()]
    if ligand_chain: chains.remove(ligand_chain)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    if atms: pose_atoms = [atom for atom in pose_atoms if atom.id in atms]
    
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

def concat_motifs(motif_list: "list[dict]") -> dict:
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
