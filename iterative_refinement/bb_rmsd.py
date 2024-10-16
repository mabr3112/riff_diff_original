#!/home/mabr3112/anaconda3/bin/python3
#
#       This script is intended to enable backbone RMSD calculation between two directories containing (input_dir:) altered poses 
#       and (ref_dir:) the template poses. The script assumes that the poses have the same name.
#
#       This script can also be used to calculate backbone RMSD for predictions of sequences that were designed with MPNN.
#       Set the option --mpnn=True and the script removes the _0001 suffix from the sequence.
#
#
#########################################

import Bio.PDB
import numpy as np
from glob import glob

def get_CA_atoms(structure: Bio.PDB.Structure, chains=None) -> list:
    '''
    Scrapes all Ca atoms from the structure object, puts them into a list and returns the list.
    Ignores chain X by default!
    
    Args:
        <structure>         Bio.PDB.Structure object from which to collect atoms.
        <chains>            Optional: chains from with to collect the atoms.
    
    '''
    # Gather all chains from the structure
    ca_atms_list = list()
    if chains:
        chains = [structure[chain] for chain in chains]
    else:
        chains = [chain for chain in structure if chain.id != "X"]
    
    for chain in chains:
        # Only select amino acids in each chain:
        residues = [res for res in chain if res.id[0] == " "]
        for residue in residues:
            ca_atms_list.append(residue["CA"])

    return ca_atms_list

def superimpose_calc_rmsd(ref_pdbfile, pdbfile, ref_chains=None, pose_chains=None, atoms=["CA"]) -> float:
    '''
    Superimpose two structures based on residue selections. Sensitive to the order in the selection!
    Returns RMSD after superimposition.
    
    Args:
        <ref_pdbfile>               Path to the reference pdb-file.
        <pdbfile>                   Path to the pdb-file for which RMSD should be calculated
        <ref_chains>                Optional: List with chains that should be used for RMSD calculation. 
                                    e.g. ["A"] if only chain A should be used.
        <pose_chains>               Optional: same as ref_chains: list with chains that should be used for RMSD calculation.
                                    List of chains should be of same length (Number of atoms has to be the same for RMSD calculation).
        <atoms>                     Optional, list: Which atoms should be taken for RMSD calculation. Default: ["CA"]
        
    Returns:
        returns rmsd as float.
    '''
    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    
    # Get the structures
    ref_pose = pdb_parser.get_structure("ref", ref_pdbfile)
    target_pose = pdb_parser.get_structure("target", pdbfile)
    
    # Select CA atoms from structures:
    ref_atoms = get_CA_atoms(ref_pose[0], chains=ref_chains)
    target_atoms = get_CA_atoms(target_pose[0], chains=pose_chains)

    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    super_imposer.apply(target_pose[0].get_atoms())

    # Return RMSD:
    return super_imposer.rms

def bb_rmsd_dir(input_dir, ref_dir="../trfold_relax", write=None, scorefile="bb_rmsd.json", remove_layers=None, custom_scorename=None):
    '''
    Takes af2_localcolabfold output as a directory (af2_predictions/) and calculates bb-RMSD relative to its input pdb from a reference folder.
    Reference folder can be set with <ref_dir>.

    Returns dictionary with bb-rmsd. Writes rmsd.json file.
    '''
    import json
    # Glob all pdb-files in input_dir into list.
    fl = glob((globstring := f"{input_dir}/*.pdb"))
    if not fl: raise FileNotFoundError(f"ERROR: No .pdb files found in {input_dir}. Are you sure it is the right directory?")

    # If the sequences were designed with MPNN, then remove the sequence index to revert to original pdb name (remove the _0001 ...)
    if remove_layers:
        print(f"remove_layers set to {remove_layers}. Removing {remove_layers} indexing layers '_0001'")
        remove_layers = int(remove_layers)
        pl = [f'{"_".join(x.split("_")[:-1*remove_layers])}.pdb' for x in fl]
    else:
        pl = fl

    # Define columns for dict:
    print(f"Calculating RMSD for {len(fl)} files found in {input_dir}.")
    rd = dict()
    cols = ["origin", "description", "bb_ca_rmsd"]
    for c in cols:
        rd[c] = list()

    for f, p in zip(fl, pl):
        rd["origin"].append(f)
        rd["description"].append(f.split("/")[-1][:-4])
        p = p.split("/")[-1]
        rd["bb_ca_rmsd"].append(superimpose_calc_rmsd(ref_dir + "/" + p, f, ref_chains=[], pose_chains=[], atoms=["CA"]))
    
    # Rename 'bb_ca_rmsd' to custom_scorename if option is set:
    if custom_scorename: rd[custom_scorename] = rd.pop("bb_ca_rmsd")

    if write:
        print(f"Writing file {scorefile}.")
        with open(scorefile, 'w') as f:
            f.write(json.dumps(rd))

    return rd

def main(args):
    s = '#'*50
    print(f"{s}\nRunning bb_rmsd.py on {args.input_dir} with references from {args.ref_dir}\n{s}\n")
    bb_rmsd_dir(input_dir=args.input_dir, ref_dir=args.ref_dir, write=True, scorefile=args.output_path, remove_layers=args.remove_layers, custom_scorename=args.custom_scorename)
    return None

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="Path to the directory that contains input .pdb files.")
    argparser.add_argument("--ref_dir", type=str, required=True, help="Path to the directory that contains all reference pdb files.")
    argparser.add_argument("--output_path", type=str, help="Path and name of the scorefile that should be written. Default: input_dir/bb_rmsd.json")
    argparser.add_argument("--remove_layers", help="How many index layers (_0001) have to be removed from the .pdb filename to match the .pdb filename of the reference_pdb? This option removes index layers for RMSD calculation reference pdb.")
    argparser.add_argument("--custom_scorename", help="Provides a custom name to your bb_rmsd score.")
    args = argparser.parse_args()

    if not args.input_dir.endswith("/"): args.input_dir += "/"
    if not args.output_path: args.output_path = args.input_dir + "/bb_rmsd.json"

    main(args)
