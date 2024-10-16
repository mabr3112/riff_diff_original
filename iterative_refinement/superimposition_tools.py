#!/home/mabr3112/anaconda3/bin/python3.9

from ctypes import util
import Bio
from Bio import PDB
from Bio.PDB.PDBIO import PDBIO
from copy import deepcopy
import utils

def get_atoms_of_motif(structure: Bio.PDB.Structure, motif: dict, atoms:list=None, forbidden_atoms:list=["H", "NE1", "OXT"]) -> list:
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
                [atms_list.append(structure[chain][(' ', resi, ' ')][atm]) for atm in atoms]
            else:
                # otherwise add all atoms (including sidechain atoms) of residue:
                [atms_list.append(atm) for atm in structure[chain][(' ', resi, ' ')].get_atoms()]
            
    # remove atoms that might produce errors when running superimposition (hydrogens, terminal atoms):
    return [atm for atm in atms_list if atm.name not in forbidden_atoms]

def superimpose_by_motif(fixed: Bio.PDB.Structure.Structure, mobile: Bio.PDB.Structure.Structure, fixed_motif:dict, mobile_motif:dict=None, atoms:list=["CA"]) -> Bio.PDB.Structure.Structure:
    '''Superimposes 'moving' onto 'fixed' by [atoms].
    returns moving after superimposition
    '''
    # collect Motif atoms for superimposition
    fixed_ca = get_atoms_of_motif(fixed, fixed_motif, atoms=atoms)
    mobile_ca = get_atoms_of_motif(mobile, mobile_motif or fixed_motif, atoms=atoms)

    # superimpose mobile (pose) onto target motif (ref)
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(fixed_ca, mobile_ca)
    super_imposer.apply(mobile)

    return mobile

def superimpose_pdb_by_motif(fixed_path: str, mobile_path: str, fixed_motif: dict, mobile_motif:dict=None, atoms:list=["CA"], save_path:str=None) -> str:
    '''Implementation of superimpose_by_motif() for pdb-files (where the file is automatically saved somewhere)'''
    # load
    fixed = utils.biopython_tools.load_structure_from_pdbfile(fixed_path)
    mobile = utils.biopython_tools.load_structure_from_pdbfile(mobile_path)
    
    # superimpose
    mobile = superimpose_by_motif(fixed, mobile, fixed_motif, mobile_motif or fixed_motif, atoms=atoms)

    # store at save_path or at mobile_path if no save_path was specified:
    return utils.biopython_tools.store_pose(mobile, save_path or mobile_path)

def get_CA_of_chain(model: Bio.PDB.Structure, chain: str) -> list:
    '''
    Collects C-alpha atoms of a chain into a list.
    Args:
        <structure>           Has to be a model of a Bio.PDB.Structure object (Bio.PDB.Structure[0])
        <chain>               Chain name of which you want to gather C-alpha atoms.

    Returns:
        List of C-alpha atoms
    '''
    # Return C-alpha atom for every residue in structure[chain] if the residue is an amino acid (if res.id[0] == " ")
    return [res["CA"] for res in model[chain] if res.id[0] == " "]

def superimpose_add_chain(ref_path: str, pose_path: str, copy_chain: str, superimpose_chain="A", new_pose_path=None) -> str:
    '''
    Superimposes chain in <ref_path> onto <superimpose_chain> in <pose_path> and copies <copy_chain> into <ref_path>
    Args:
        <ref_path>                   path to the reference pose
        <pose_path>                  path to the pose to which a chain from <ref_path> should be added
        <copy_chain>                 Chain name of the chain that should be copied
        <superimpose_chain>          Chain name of the chain that should be superimposed.

    Returns:
        Path to the modified pose.
    '''
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    tt = pdb_parser.get_structure("tt", pose_path)
    ref = pdb_parser.get_structure("ref", ref_path)

    # collect CA-atoms for superimposition
    mobile = get_CA_of_chain(tt[0], superimpose_chain)
    target = get_CA_of_chain(ref[0], superimpose_chain)

    # superimpose mobile (pose) onto target (ref)
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(target, mobile)
    super_imposer.apply(tt[0][superimpose_chain])

    # Add <copy_chain> into pose:
    tt[0].add(ref[0][copy_chain])
    
    # save the pose at <pose_path>
    save_path = new_pose_path or pose_path
    io = PDBIO()
    io.set_structure(tt)
    io.save(save_path)

    return pose_path

def superimpose_add_chain_by_motif(ref_path: str, pose_path: str, copy_chain: str, ref_motif: dict, pose_motif: dict, new_pose_path=None, superimpose_atoms=["CA"], remove_hydrogens:bool=True, overwrite:bool=True) -> str:
    '''
    Superimpose a chain in a pose PDB structure onto a motif in a reference PDB structure, and add another chain from the reference PDB structure to the pose.
    
    The motifs are specified as dictionaries where the keys are the IDs of the chains in the PDB structures, and the values are lists of residue numbers. The function will use atoms from the specified residues in the specified chains to perform the superimposition.
    
    Args:
        ref_path (str): The path to the reference PDB file.
        pose_path (str): The path to the pose PDB file.
        copy_chain (str): The ID of the chain in the reference PDB file to be added to the pose.
        ref_motif (dict): A nested dictionary specifying the chains and residues of the atoms in the reference structure to use for the superimposition.
        pose_motif (dict): A nested dictionary specifying the chains and residues of the atoms in the pose structure to use for the superimposition.
        new_pose_path (str, optional): The path to save the modified pose. If not provided, the original pose will be overwritten.
        superimpose_atoms (list, optional): A list of atom names to use for the superimposition. If not provided, all atoms in the specified residues will be used.
    
    Returns:
        str: The path to the modified pose PDB file.
    
    Examples:
        superimpose_add_chain_by_motif("ref.pdb", "pose.pdb", "B", {"A": [10, 20, 30]}, {"A": [10, 20, 30]}, "modified_pose.pdb", superimpose_atoms=["CA", "C", "O"])
    '''
    # start the parser and load the structures
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    tt = pdb_parser.get_structure("tt", pose_path)
    ref = pdb_parser.get_structure("ref", ref_path)

    # collect Motif atoms for superimposition
    target = get_atoms_of_motif(ref[0], ref_motif, atoms=superimpose_atoms)
    mobile = get_atoms_of_motif(tt[0], pose_motif, atoms=superimpose_atoms)

    #target = [atom for atom in target if atom.name == "CA"]
    #mobile = [atom for atom in mobile if atom.name == "CA"]

    # remove hydrogens if option is set:
    if remove_hydrogens:
        target = [atom for atom in target if not "H" in atom.name]
        mobile = [atom for atom in mobile if not "H" in atom.name]

    if len(target) != len(mobile): print(f"WARNING: atom lists differ in length, {[(x, y) for x, y in zip(target, mobile)]} target:{target}\n mobile: {mobile}")

    # superimpose mobile (pose) onto target motif (ref)
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(target, mobile)
    super_imposer.apply(tt[0])

    # Add <copy_chain> into pose:
    if overwrite:
        if copy_chain in [chain.id for chain in tt[0].get_chains()]:
            tt[0].detach_child(copy_chain)
    tt[0].add(ref[0][copy_chain])
    
    # save the pose at <pose_path>
    save_path = new_pose_path or pose_path
    io = PDBIO()
    io.set_structure(tt)
    io.save(save_path)

    return save_path

def remove_chain(pose_path: str, chain: str, new_pose_path: str) -> str:
    '''
    Remove a specified chain from a PDB file and save the modified structure to a new file.
    
    Args:
        pose_path (str): The path to the original PDB file.
        chain (str): The ID of the chain to remove.
        new_pose_path (str): The path to save the modified PDB file.
    
    Returns:
        str: The path of the modified PDB file.
    
    Examples:
        remove_chain("original.pdb", "A", "modified.pdb")
    '''
    # start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    
    # load the structure
    tt = pdb_parser.get_structure("tt", pose_path)[0]
    
    # remove chain
    tt.detach_child(chain)

    # Save the modified structure
    io = Bio.PDB.PDBIO()
    io.set_structure(tt)
    io.save(new_pose_path)
    
    return pose_path

def add_chain(pose_path:str, ref_path:str, copy_chain:str, new_pose_path:str=None, translate_x:int=None, exist_ok:bool=False, rename_chain:str=None) -> str:
    '''
    Adds a copy of a chain from a reference structure to a target structure.

    Args:
        pose_path (str): Path to the target structure file.
        ref_path (str): Path to the reference structure file.
        copy_chain (str): ID of the chain to be copied from the reference structure.
        new_pose_path (str, optional): Path to save the modified target structure. If not provided, the target structure will be overwritten. Defaults to None.
        translate_x (int, optional): Value to translate the copied chain in the x-direction. Defaults to None.
        exist_ok (bool, optional): Flag to allow overwriting an existing chain in the target structure. Defaults to False.
    
    Returns:
        str: Path to the target structure file.
    
    Raises:
        ValueError: If the copied chain already exists in the target structure and `exist_ok` is False.
    '''
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    tt = pdb_parser.get_structure("tt", pose_path)
    ref = pdb_parser.get_structure("ref", ref_path)

    # Check if copy_chain already exists in tt[0]
    if tt[0].has_id(rename_chain or copy_chain) and not exist_ok: raise ValueError(f"Chain {copy_chain} already exists in {pose_path}. Choose a different chain or input structure.")

    # Copy chain from ref and change name if rename_chain is provided:
    added_chain = ref[0][copy_chain].copy()

    # Rename the chain if rename_chain is provided
    if rename_chain: added_chain.id = rename_chain

    # Add <copy_chain> into pose:
    tt[0].add(added_chain)

    # translate copy_chain in tt[0] by the value set in translate_x if option is set:
    if translate_x:
        for atom in tt[0][copy_chain].get_atoms():
            x, y, z = atom.get_coord()
            atom.set_coord((x + translate_x, y, z))
    
    # save the pose at <pose_path>
    save_path = new_pose_path or pose_path
    io = PDBIO()
    io.set_structure(tt)
    io.save(save_path)

    return pose_path
