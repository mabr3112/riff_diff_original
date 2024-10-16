## IMPORTS
import numpy as np
import Bio.PDB
import utils.biopython_tools
import pandas as pd
import superimposition_tools
import math
import json


## Functions
def calc_rog(pose: Bio.PDB.Structure.Structure, min_dist=0) -> float:
    '''
    adapted from RFdiffusion's potentials.py
    '''
    # get CA coordinates and calculate centroid
    ca_coords = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
    centroid = np.mean(ca_coords, axis=0)

    # calculate average distance of CA atoms to centroid
    dgram = np.maximum(min_dist, np.linalg.norm(ca_coords - centroid, axis=-1))

    # take root over squared sum of distances and return (rog):
    return np.sqrt(np.sum(dgram**2) / ca_coords.shape[0])

def calc_rog_of_pdb(pdb_path: str, min_dist=0):
    '''AAA'''
    return calc_rog(utils.biopython_tools.load_structure_from_pdbfile(pdb_path), min_dist=min_dist)

def calc_intra_contacts(pose: Bio.PDB.Structure.Structure, d_0:float=7.5, r_0:float=7.5) -> float:
    '''AAA'''
    # get CA coords of all atoms
    Ca = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
    
    # Calculate the pairwise distance between all points in Ca
    dgram = np.linalg.norm(Ca[:, np.newaxis] - Ca[np.newaxis, :], axis=-1)
    
    # Divide the distance by r_0 and subtract d_0
    divide_by_r_0 = (dgram - d_0) / r_0
    
    # Calculate the numerator and denominator of the ncontacts formula
    numerator = np.power(divide_by_r_0, 6)
    denominator = np.power(divide_by_r_0, 12) + 1e-9
    
    # Calculate ncontacts
    ncontacts = (1 - numerator) / (1 - denominator)

     # Set the diagonal elements of ncontacts to 0
    np.fill_diagonal(ncontacts, 0)
    
    # Return the weighted sum of ncontacts
    return ncontacts.sum() * -1

def calc_intra_contacts_of_pdb(pdb_path: str, d_0:float=7.5, r_0:float=7.5) -> float:
    '''AAA'''
    return calc_intra_contacts(utils.biopython_tools.load_structure_from_pdbfile(pdb_path), d_0=d_0, r_0=r_0)

def calc_ligand_contacts(pose: Bio.PDB.Structure.Structure, ligand_chain:str, d_0:float=3.5, r_0:float=3.5, ignore_atoms:list[str]=["H"]) -> float:
    '''Calculates ligand contacts and divides them by number of atoms in the ligand.
    '''
    def get_protein_atoms(pose: Bio.PDB.Structure.Structure, ligand_chain:str) -> list:
        chains = [x.id for x in pose.get_chains()]
        chains.remove(ligand_chain)
        return [atom for chain in chains for atom in pose[chain].get_atoms()]

    if type(ligand_chain) == "str":
        # sanity 
        check_for_chain_in_pose(pose, ligand_chain)

        # get all CA coords of protein:
        Ca = np.array([atom.get_coord() for atom in get_protein_atoms(pose, ligand_chain) if atom.id == "CA"])

        # get Ligand Heavyatoms:
        lig_atms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.id not in ignore_atoms])

    elif type(ligand_chain) == Bio.PDB.Chain.Chain:
        # get all CA coords of protein:
        Ca = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
        
        # get Ligand Heavyatoms:
        lig_atms = np.array([atom.get_coord() for atom in ligand_chain.get_atoms() if atom.id not in ignore_atoms])
    else: raise TypeError(f"Expected 'ligand' to be of type str or Bio.PDB.Chain.Chain, but got {type(ligand_chain)} instead.")
        
    # calculate pairwise distances between Ca and lig_atms
    dgram = np.linalg.norm(Ca[:, np.newaxis] - lig_atms[np.newaxis, :], axis=-1)

    # Divide the distance by r_0 and subtract d_0
    divide_by_r_0 = (dgram - d_0) / r_0
    
    # Calculate the numerator and denominator of the ncontacts formula
    numerator = np.power(divide_by_r_0, 6)
    denominator = np.power(divide_by_r_0, 12) + 1e-9
    
    # Calculate ncontacts
    ncontacts = (1 - numerator) / (1 - denominator)

    return -1 * ncontacts.sum() / len(lig_atms)

def calc_ligand_contacts_of_pdb(pdb_path: str, ligand_chain: str, ligand_pdb_path: str, d_0:float=3.5, r_0:float=3.5, ignore_atoms:list=["H"]):
    '''AAA'''
    if ligand_pdb_path: 
        check_for_chain_in_pdb(ligand_pdb_path, ligand_chain)
        ligand_chain = utils.biopython_tools.load_structure_from_pdbfile(ligand_pdb_path)[ligand_chain]
    return calc_ligand_contacts(utils.biopython_tools.load_structure_from_pdbfile(pdb_path), ligand_chain, d_0, r_0, ignore_atoms)


def check_for_chain_in_pose(pose: Bio.PDB.Structure.Structure, chain:str):
    '''Checking function'''
    if chain in [x.id for x in pose.get_chains()]: return
    else: raise KeyError(f"Chain {chain} not found in pose {pose.id}")

def get_protein_atoms(pose: Bio.PDB.Structure.Structure, ligand_chain:str) -> list:
    chains = [x.id for x in pose.get_chains()]
    chains.remove(ligand_chain)
    return [atom for chain in chains for atom in pose[chain].get_atoms()]

def check_for_ligand_clash(pose: Bio.PDB.Structure.Structure, ligand_chain:str, dist:float=1.8, bb_atoms=["CA", "N", "C", "O"], ignore_atoms=["H"]) -> int:
    '''AAA'''
    # get atoms
    protein_bb_atoms, lig_atoms = get_protein_and_ligand_atoms(pose, ligand_chain, bb_atoms=bb_atoms, ignore_atoms=ignore_atoms)
        
    # calculate pairwise distances between Ca and lig_atms
    dgram = np.linalg.norm(protein_bb_atoms[:, np.newaxis] - lig_atoms[np.newaxis, :], axis=-1)

    # check if there is any pairwise distance lower than dist:
    return np.any(dgram < dist)

def check_for_chain_in_pdb(pdb_path:str, chain:str) -> None:
    '''AAA'''
    p = utils.biopython_tools.load_structure_from_pdbfile(pdb_path)
    if chain not in [x.id for x in p.get_chains()]: raise KeyError(f"Chain {chain} not found in pose at {pdb_path}")
    else: return None

def check_for_ligand_clash_of_pdb(pdb_path: str, ligand_chain: str, ligand_pdb_path:str=None, dist:float=1.8, ignore_atoms=["H"]) -> int:
    '''
    If 'ligand_pdb_path' is passed, the ligand will be taken from this .pdb file. If not, the ligand will be taken from 'pdb_path'.
    'ligand_chain' always has to be specified.
    '''
    if ligand_pdb_path: 
        check_for_chain_in_pdb(ligand_pdb_path, ligand_chain)
        ligand_chain = utils.biopython_tools.load_structure_from_pdbfile(ligand_pdb_path)[ligand_chain]
    return check_for_ligand_clash(utils.biopython_tools.load_structure_from_pdbfile(pdb_path), ligand_chain=ligand_chain, dist=dist, ignore_atoms=ignore_atoms)

def superimpose_and_check_for_ligand_clash(pose: Bio.PDB.Structure.Structure, ligand_pose: Bio.PDB.Structure.Structure, ligand_chain:str, pose_motif:dict, ligand_motif:dict=None, dist:float=1.8) -> bool:
    '''AAA'''
    ligand_pose = superimposition_tools.superimpose_by_motif(pose, ligand_pose, fixed_motif=pose_motif, mobile_motif=ligand_motif, atoms=["CA"])
    return check_for_ligand_clash(pose=pose, ligand_chain=ligand_pose[ligand_chain], dist=dist)

def get_protein_and_ligand_atoms(pose: Bio.PDB.Structure.Structure, ligand_chain, bb_atoms=["CA", "C", "N", "O"], ignore_atoms=["H"]) -> "tuple[list]":
    '''AAA'''
    if type(ligand_chain) == type("str"):
        # get all CA coords of protein:
        check_for_chain_in_pose(pose, ligand_chain)
        protein_atoms = np.array([atom.get_coord() for atom in get_protein_atoms(pose, ligand_chain) if atom.id in bb_atoms])

        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.id not in ignore_atoms])

    elif type(ligand_chain) == Bio.PDB.Chain.Chain:
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
        
        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in ligand_chain.get_atoms() if atom.id not in ignore_atoms])
    else: raise TypeError(f"Expected 'ligand' to be of type str or Bio.PDB.Chain.Chain, but got {type(ligand_chain)} instead.")
    return protein_atoms, ligand_atoms

def calc_ligand_rep(pose: Bio.PDB.Structure.Structure, ligand_chain, r_0:float=1.5, bb_atoms=["CA", "C", "N", "O"], ignore_atoms=["H"]) -> float:
    '''AAA'''
    bb_atoms, ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, bb_atoms=bb_atoms, ignore_atoms=ignore_atoms)

    # calculate pairwise distances between Ca and lig_atms
    dgram = np.linalg.norm(bb_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # get all distances lower than r_0:
    dgram = dgram[dgram <= r_0]
    
    # subtract all distances from r_0, square the difference, sum them up and take the square root over the sum:
    rep = np.sqrt(np.sum(np.power(r_0 - dgram, 2)) / len(ligand_atoms))

    return rep

def calc_pocket_score(pose, ligand_chain, bb_atoms=["CA", "C", "N", "O"], ignore_atoms=["H"], rep_weight:float=5, rep_radius:float=1.5, coordination_strength:float=5, coordination_radius:float=8) -> float:
    ''''''
    bb_atoms, ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, bb_atoms=bb_atoms, ignore_atoms=ignore_atoms)

    # calculate pairwise distances between Ca and lig_atms
    dgram = np.linalg.norm(bb_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # calculate rep
    rep_dgram = dgram[dgram <= rep_radius]
    rep = np.sqrt(np.sum(np.power(rep_radius - rep_dgram, 2)) / len(ligand_atoms)) 

    # calculate coordination
    coord_dgram = dgram[dgram <= coordination_radius]
    coord = np.sqrt(np.sum(np.power(coordination_radius - coord_dgram, 1)) / len(ligand_atoms))

    # calculate pocket score
    return np.exp(-1 * (np.abs(coord - coordination_strength) + rep*rep_weight))

def calc_pocket_score_v2(pose: Bio.PDB.Structure.Structure, ligand_chain, bb_atoms:"list[str]"=["CA"], ignore_atoms:"list[str]"=["H"], dist:float=5) -> float:
    '''Calculates deviation of minimum [Ligand-atom to Protein-atom ['bb_atoms']] distance from specified parameter 'dist' '''
    bb_atoms, ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, bb_atoms=bb_atoms, ignore_atoms=ignore_atoms)

    # calculate pairwise distances between lig_atms and Ca:
    dgram = np.linalg.norm(ligand_atoms[:, np.newaxis] - bb_atoms[np.newaxis, :], axis=-1)

    # get minimum distances:
    min_dgram = np.min(dgram, axis=-1)

    return np.exp(-1 * np.sqrt(np.sum(np.power(dist - min_dgram, 2) / ligand_atoms.shape[0])))

def calc_site_score(sc_rmsd, residue_plddt):
    site_score = (1 / (math.e ** sc_rmsd)) * residue_plddt / 100
    return round(site_score, 4)

def calc_pocket_score_v3(pose: Bio.PDB.Structure.Structure, ligand_chain, bb_atoms:"list[str]"=["CA"], ignore_atoms:"list[str]"=["H"]) -> float:
    '''Calculates Adrian's pocket score.
    Good values range from 3.9 to 6.1 if using CA atoms as backbone atoms.'''
    bb_atoms, ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, bb_atoms=bb_atoms, ignore_atoms=ignore_atoms)

    # calculate pairwise distances between lig_atms and Ca:
    dgram = np.linalg.norm(ligand_atoms[:, np.newaxis] - bb_atoms[np.newaxis, :], axis=-1)

    # get minimum distances:
    min_dgram = np.min(dgram, axis=-1)

    # calculate average distance:
    return round(np.sum(min_dgram) / min_dgram.shape[0], 3)

def calc_sequence_similarity(seq1: str, seq2: str) -> float:
    ''''''
    if len(seq1) != len(seq2): raise ValueError(f"Sequences must be of the same length. Length of seq1: {len(seq1)}, length of seq2: {len(seq2)}")
    matching = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matching / len(seq1)

def all_against_all_sequence_similarity(input_seqs: list[str]) -> list:
    ''''''
    aa_mapping = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    mapped_seqs = np.array([[aa_mapping[s] for s in seq] for seq in input_seqs])

    expanded_a = mapped_seqs[:, np.newaxis]
    expanded_b = mapped_seqs[np.newaxis, :]
    similarity_matrix = np.mean(expanded_a == expanded_b, axis=2)

    # convert diagonal from 1.0 to -inf
    np.fill_diagonal(similarity_matrix, -np.inf)

    return list(np.max(similarity_matrix, axis=1))


## put into AF2 later
def calc_pae_interaction(input_array: np.array, binderlen: int) -> list[float]:
    '''Calculates interaction pae given a 2D pae array and the length of the binder (in chain A)'''
    return np.mean((input_array[:binderlen, binderlen:]) + np.mean(input_array[binderlen:, :binderlen])) / 2

def calc_pae_interaction_from_raw_pae_dict(input_str: str, binderlen: str) -> list[float]:
    '''Calculates pae_interaction from raw pae_list output of af2_scorecollector.'''
    # load pAEs from pae_list that is given as a string. Frist convert string into correct json format (""):
    if type(input_str) == str:
        pae_dict = json.loads(input_str.replace("'", '"'))
    elif type(input_str) == dict:
        pae_dict = input_str
    else:
        TypeError(f"Type {type(input_str)} not supported for this function.")

    # convert to numpy array
    paes = np.array([pae_dict[key] for key in pae_dict])

    return calc_pae_interaction(paes, binderlen)

## put into (maybe) ESM or not. Entropy should be general!
def entropy(prob_distribution):
    # Filter out zero probabilities to avoid log(0)
    prob_distribution = prob_distribution[prob_distribution > 0]
    
    # Compute entropy
    H = np.sum(prob_distribution * np.log2(prob_distribution))
    
    return -H

# new metrics ############
def count_mutations(seq1: str, seq2: str) -> tuple[int, list[str]]:
    """
    Compares two protein sequences and counts the number of mutations, 
    returning both the count and a detailed list of mutations.

    Each mutation is represented in the format: 
    '[original amino acid][position][mutated amino acid]'.
    
    Parameters:
    seq1 (str): The first protein sequence (e.g., wild type).
    seq2 (str): The second protein sequence (e.g., variant).

    Returns:
    tuple[int, list[str]]: A tuple where the first element is an integer 
    representing the number of mutations, and the second element is a list of 
    strings detailing each mutation.

    Raises:
    ValueError: If the input sequences are not of the same length.

    Example:
    >>> count_mutations("ACDEFG", "ACDQFG")
    (1, ['E4Q'])
    """
    # Check if the lengths of the sequences are the same
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")

    mutations = []
    mutation_count = 0

    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            mutation_count += 1
            mutation = f"{a}{i+1}{b}"
            mutations.append(mutation)

    return mutation_count, mutations
