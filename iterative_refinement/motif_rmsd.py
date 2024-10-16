#!/home/mabr3112/anaconda3/bin/python3
#       
#       This script is intended to be used to calculate RFDesign motif and forced_aa RMSDs.
#
#################################################
import json
import pandas as pd
from subprocess import run
import Bio
import numpy as np
from glob import glob
import os
from Bio import PDB
import Bio.PDB

def check_pdb_path(pdb_path: str) -> None:
    '''Checks if file has pdb_path'''
    if not pdb_path.endswith(".pdb"):
        raise ValueError("Input to RMSD calculation requires .pdb file! The following input is not a .pdb file: {pdb_path}")
    return None

def superimpose_calc_motif_rmsd(ref_pdbfile: str, target_pdbfile: str, ref_selection: dict, target_selection: dict, atoms=["CA"]) -> float:
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
    # check for .pdb files
    check_pdb_path(ref_pdbfile)
    check_pdb_path(target_pdbfile)

    # Start the parser
    import Bio.PDB
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("ref", ref_pdbfile)
    hal_structure = pdb_parser.get_structure("hal", target_pdbfile)

    # Use the first model in the pdb-files for alignment
    ref_model = ref_structure[0]
    hal_model = hal_structure[0]

    # Make a list of the atoms (in the structures) you wish to align.
    ref_atoms = []
    hal_atoms = []

    for chain in ref_selection:
        for res in ref_selection[chain]:
            [ref_atoms.append(ref_model[chain][(' ', res, ' ')][x]) for x in atoms]

    for chain in target_selection:
        for res in target_selection[chain]:
            [hal_atoms.append(hal_model[chain][(' ', res, ' ')][x]) for x in atoms]

    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, hal_atoms)

    # Return RMSD:
    return super_imposer.rms


def superimpose_calc_motif_rmsd_heavy(ref_pdbfile: str, target_pdbfile: str, ref_selection: dict, target_selection: dict) -> float:
    '''
    Superimpose two structures based on residue selections. Sensitive to the order in the selection!
    Returns RMSD after superimposition.
    Args:
        <ref_pdbfile>                     Path to the reference pdb-file
        <target_pdbfile>                  Path to the target pdb-file (for which rmsd should be calculated)
        <ref_selection>                   Dictionary containing the residue positions for which RMSD should be calculated.
                                          !!! IMPORTANT: Order of amino acids is relevant to RMSD calculation! (for alignment)
                                          E.g. {"A": [1, 3, 4, 12], "B": [1, 15, 4]}
        <target_selection>                Dictionary containing the residue positions for with RMSD should be calculated.
                                          Example (see <ref_selection>)
    Returns:
        RMSD between ref_selection and target_selection of pdbfiles. Returns float.
    '''
    # check for .pdb files
    check_pdb_path(ref_pdbfile)
    check_pdb_path(target_pdbfile)

    pdbfile = target_pdbfile

    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("ref", ref_pdbfile)
    target_structure = pdb_parser.get_structure("target", target_pdbfile)

    # Use the first model in the pdb-files for alignment
    ref_model = ref_structure[0]
    target_model = target_structure[0]
    
    # Make a list of the atoms (in the structures) you wish to align.
    ref_atoms = []
    target_atoms = []
    
    for chain in ref_selection:
        for i in ref_selection[chain]:
            # sort atoms because we want to calculate RMSD between the same atoms (sometimes the order of atoms in a residue gets mixed up, depending on how the .pdb file was generated)
            [ref_atoms.append(atom) for atom in sorted(list(ref_model[chain][(' ', i, ' ')].get_atoms()), key=lambda a: a.id)]
    
    for chain in target_selection:
        for i in target_selection[chain]:
            [target_atoms.append(atom) for atom in sorted(list(target_model[chain][(' ', i, ' ')].get_atoms()), key=lambda a: a.id)]
    
    # Remove possible hydrogens from reference or hallucination atoms selection:
    forbidden_atoms = ["H", "NE1", "OXT"]
    for s in forbidden_atoms:
        ref_atoms = [x for x in ref_atoms if s not in x.name]
        target_atoms = [x for x in target_atoms if s not in x.name]
        
    if len(ref_atoms) != len(target_atoms): atoms_error_msg(ref_pdbfile, target_pdbfile, ref_atoms, target_atoms, ref_selection, target_selection)
        
    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    #super_imposer.apply(hal_model.get_atoms())

    # Return RMSD:
    rmsd = super_imposer.rms
    return rmsd

def atoms_error_msg(ref_pdbfile: str, target_pdbfile: str, ref_atoms: list, target_atoms: list, ref_selection, target_selection) -> None:
    '''spits out an error MSG'''
    print(f"ERROR: ref_atoms and hal_atoms are not equally long. Were the sidechains fixed?")
    print(f"RMSD value for '{target_pdbfile.replace('.pdb', '')} will be set to NaN!'")
    print(ref_pdbfile, target_pdbfile)
    print("\t".join([f"{x} : {y}" for x, y in zip(ref_atoms, target_atoms)]))
    print(len(ref_atoms), len(target_atoms))

    # get Residue Information:
    ref_res = [atom.get_parent().get_resname() for atom in ref_atoms]
    target_res = [atom.get_parent().get_resname() for atom in target_atoms]
    
    print("\t".join([f"{x} : {y}" for x, y in zip(ref_res, target_res)]))
    
    raise ValueError(f"The RMSD calculation for {target_pdbfile} failed. Reference file: {ref_pdbfile}\nreference_motif: {ref_selection}\n target_motif: {target_selection}")

def main(args):
    '''
    Calculates motif_ca_rmsd and forced_aa_heavy_rmsd for RFDesign outputs in <input_dir>.
    Writes scores into a scorefile (output_path)
    Set MPNN to true to specify if the sequences were designed with MPNN after RFDesign hallucination.
    MPNN=True removes the last [_000n] tag of the input pdb.
    
    Returns scoredict.
    '''
    print(f"{'#'*50}\nRunning motif_rmsd.py on {args.input_dir}\nReference: {args.ref_pdb}\n{'#'*50}\n")

    # initiate the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    
    # collect all files into list
    input_files = glob(f"{args.input_dir}/*.pdb")
    if not input_files: raise FileNotFoundError(f"ERROR: No .pdb files were found in {args.input_dir}. Are you sure it was the correct path?")
    print(f"Found {len(input_files)} files in {args.input_dir}. Starting RMSD calculation.")

    # Setup Scoredict
    cols = ["description", "motif_ca_rmsd", "motif_heavy_rmsd"]
    scoredict = {col: [] for col in cols}

    # parse motif:
    motif = [int(x) for x in args.motif.split(",")]

    for file in input_files:
        desc = file.split("/")[-1].replace(".pdb", "")
        #if remove_layers:
        #    remove_layers = int(remove_layers)
        #    trb = "_".join(file.split("/")[-1].split("_")[:-1*remove_layers]) + ".trb"
        #else:
        #    trb = file.split("/")[-1].replace(".pdb", ".trb") 

        #trb_file_path = f"{trb_dir}/{trb}"
        #trb_file = np.load(trb_file_path, allow_pickle=True)
        #hal_mask = [motif]
        #ref_mask = [x[1] for x in trb_file["con_ref_pdb_idx"]]
        #ref_pdb_path = trb_file["settings"]["pdb"]

        # Calculate motif ca rmsd and append to scoredict
        scoredict["description"].append(desc)
        scoredict["motif_ca_rmsd"].append(superimpose_calc_rmsd(args.ref_pdb, file, ref_selection=motif, target_selection=motif))

        # translate pdb numbering of forced_aa into con_hal_pdb_idx.
        #forced_aa = [int(x[1:-1]) for x in trb_file["settings"]["force_aa"].split(",")] # forced_aa are fixed residues on the ref pose
        #fixed_pos_hal = list() # fixedpos are fixed residues on the hal pose
        #fixed_pos_ref = list()
        #for i, x in enumerate(trb_file["con_ref_pdb_idx"]):
        #    if x[1] in forced_aa:
        #        fixed_pos_hal.append(trb_file["con_hal_pdb_idx"][i][1])
        #        fixed_pos_ref.append(x[1])

        # Calculate motif heavy rmsd and append to scoredict. Note: heavy includes sidechains. This is why we can only calculate heavy over forced_aa!
        scoredict["motif_heavy_rmsd"].append(superimpose_calc_rmsd_heavy(args.ref_pdb, file, motif, motif))
    
    print(f"Calculation finished.")
    #if write == True:
    print(f"Writing scores to file: {args.output_path}")
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(scoredict))
    return scoredict

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", required=True, type=str, help="Input directory. This should be a directory containing pdb-files for which motif_rmsd should be calculated.")
    argparser.add_argument("--ref_pdb", required=True, type=str, help="Path to the reference .pdb file for RMSD calculation.")
    argparser.add_argument("--motif", required=True, type=str, help="Example: --motif='4,5,16,19' -> specifies the amino acids in the motif.")
    argparser.add_argument("--output_path", type=str, default="", help="Path and name of your output scorefile. Otherwise the script will save the scorefile into --input_dir as motif_rmsd.json")
    args = argparser.parse_args()
    
    if not args.output_path: args.output_path = f"{args.input_dir}/motif_rmsd.json"
    if not os.path.isfile(args.ref_pdb): raise FileNotFoundError(f"File {args.ref_pdb} not found at the specified path.")

    main(args)
