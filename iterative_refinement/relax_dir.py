#!/home/mabr3112/anaconda3/bin/python3.9

import sys
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]

from iterative_refinement import *
from glob import glob
from subprocess import run
import os

def main(args):
    '''
    '''
    pdb_files = glob(f"{args.input_dir}/*.pdb")
    if not pdb_files: raise FileNotFoundError(f"ERROR: No .pdb files found in {args.input_dir} \tDid you specify the correct directory?")
    print(f"Relaxing {len(pdb_files)} poses found in {args.input_dir}\nSaving poses at {args.output_dir}")
    relax_poses = Poses(args.output_dir, glob(f"{args.input_dir}/*.pdb"))
    relaxed_poses = relax_poses.relax_poses(relax_options=f"-beta -ex1 -ex2", n=15)

    return relaxed_poses

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="Path to the directory that contains the .pdb files that you want to relax.")
    argparser.add_argument("--output_dir", type=str, help="Specify the directory to where your .pdb files should be saved to. Default: relaxed/")
    args = argparser.parse_args()

    if not args.output_dir: args.output_dir = args.input_dir + "/relaxed/"
    if not os.path.isdir(args.output_dir): run(f"mkdir -p {args.output_dir}", shell=True, stdout=True, stderr=True, check=True)

    print(f"\n{'#'*50}\nRunning relax_poses.py at {args.input_dir}\n{'#'*50}\n")
    main(args)
