#!/home/mabr3112/anaconda3/bin/python3.9

import sys
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]

import os
from glob import glob
from iterative_refinement import *
from subprocess import run

def main(args):
    '''
    '''
    # Hello World
    print(f"\n{'#'*50}\nRunning poses_to_fasta.py on {args.input_dir}\n{'#'*50}\n")

    # collect input pdb-files and check if there are any..
    pl = glob(f"{args.input_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"ERROR: No .pdb files were found in the specified --input_dir. Are you sure {args.input_dir} is the correct path?")

    # convert pdb-files to fastas
    p = Poses(f"{args.input_dir}/temp_workdir", pl)
    fastas = p.poses_pdb_to_fasta(chain_sep=args.chain_sep)

    # save fastas to output_dir
    print(f"Found {len(pl)} .pdb files in {args.input_dir}.\nConverting to .fa format and storing them at {args.output_dir}.\n")
    loc = p.dump_poses(args.output_dir)

    # remove work_dir
    run(f"rm -r {args.input_dir}/temp_workdir/", shell=True, stdout=True, stderr=True, check=True)

if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .pdb filess")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory where you want to store your .fa files")
    argparser.add_argument("--chain_sep", type=str, default=":", help="Specify the symbol that should separate chains in the fasta files.")
    args = argparser.parse_args()

    # check if output dir exists and create it if not.
    if not os.path.isdir(args.output_dir): run(f"mkdir -p {args.output_dir}", shell=True, stderr=True, check=True, stdout=True)
    main(args)
