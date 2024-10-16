#!/home/mabr3112/anaconda3/bin/python3
#       
#       This Script collects output scores of ProteinMPNN and renames the fasta files 
#       so that they are ready for post-processing (mpnn_prefilter.py and af2_localcolabfold)
#
######################################

import Bio
from Bio import SeqIO
import pandas as pd
from glob import glob
from subprocess import run
import json
import os

def mpnn_fastaparser(filename):
    '''
    
    '''
    records = list(Bio.SeqIO.parse(filename, "fasta"))
    #maxlength = len(str(len(records)))
    maxlength = 4
    
    # Set continuous numerating for the names of mpnn output sequences:
    name = records[0].name.replace(",", "")
    records[0].name = name
    for i, x in enumerate(records[1:]):
        setattr(x, "name", f"{name}_{str(i+1).zfill(maxlength)}")
    
    return records

def convert_mpnn_fastas_to_dict(fl):
    '''
    Takes already parsed list of fastas as input <fl>. Fastas can be parsed with the function mpnn_fastaparser(file).
    Should be put into list.
    Converts mpnn fastas into a dictionary:
    {
        "col_1": [vals]
              ...
        "col_n": [vals]
    }
    '''
    # Define cols and initiate them as empty lists:
    fd = dict()
    cols = ["mpnn_origin", "description", "sequence", "T", "sample", "score", "seq_recovery", "global_score"]
    for col in cols:
        fd[col] = []

    # Read scores of each sequence in each file and append them to the corresponding columns:
    for file in fl:
        for f in file[1:]:
            fd["mpnn_origin"].append(file[0].name)
            fd["sequence"].append(str(f.seq))
            fd["description"].append(f.name)
            d = {k: float(v) for k, v in [x.split("=") for x in f.description.split(", ")]}
            for k, v in d.items():
                fd[k].append(v)
    return fd

def rename_mpnn_fastas(input_dir, scorefile="mpnn_scores.json", write=None):
    '''
        Rename the mpnn-output fastas so that alphafold can take their name as input.
    '''
    # Collect all .fa-files into a list, calculate and write .json scorefile
    files = glob((globstr := f"{input_dir}/*.fa"))
    if not files:
        raise FileNotFoundError(f"No .fa files found in {input_dir}. Are you sure it is the right directory?")
    else:
        print(f"Found {len(files)} .fa files in {input_dir}. Preparing for renaming.")
    fl = [mpnn_fastaparser(file) for file in files]
    fd = convert_mpnn_fastas_to_dict(fl)
    
    # Perform File manipulation only if write is set to True!
    if write:
        # Create directory where you can copy the original sequence files into.
        run(f"mkdir -p {input_dir}/seqs_original", shell=True, stdout=True)
        
        # Move original files out of the way
        print(f"Copying original .fa files into directory {input_dir}/seqs_original.\n")
        [run(f"mv {f} {input_dir}/seqs_original/", shell=True, stdout=True, stderr=True, check=True) for f in files]

        # Write new .fa files by iterating through "description" and "sequence" keys of the fastadict fd
        print(f"Writing new fastafiles at original location ({input_dir}).\n")
        fd["location"] = list()
        for d, s in zip(fd["description"], fd["sequence"]):
            fd["location"].append((fa_file := f"{input_dir}/{d}.fa"))
            with open(fa_file, 'w') as f:
                f.write(f">{d}\n{s}")

        # Write json file with selected scores.
        with open(scorefile, 'w') as outfile:
            print(f"Writing scores into file {scorefile}.\n")
            json.dump(fd, outfile)
            
    else:
        print(f"{'#'*50}\nWarning, write was set to {write}. No Files are being written!!\n{'#'*50}")
        
    return fd

def main(input_dir):
    '''
    Renames mpnn fastas for better output, then passes them to colabfold-batch to be simultaneously predicted by <n_gpus> parallel prediction runs.
    Returns mpnn scores and commands to run colabfold.
    '''
    # Rename MPNN output .fa-files:
    s = '\n' + '#'*50 + '\n'
    print(f"{s}Starting mpnn_scorecollector.py {s}")
    scores = rename_mpnn_fastas(input_dir, scorefile=f"/mpnn_scores.json", write=True)
    
    return scores

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, default="", help="Path to the output directory of ProteinMPNN. Should be a seqs folder. (mpnn_design_dir.py). This directory should contain .fa files")
    args = argparser.parse_args()
    
    # Check input and output directories:
    if not args.input_dir: args.input_dir = os.getcwd()
    
    # Run script:
    main(args.input_dir)
