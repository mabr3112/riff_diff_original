#!/home/mabr3112/anaconda3/bin/python3.9
import sys
sys.path.append("/home/mabr3112/riff_diff")
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]

import json
from iterative_refinement import *
from glob import glob
import utils.plotting as plots
import logging

def main(args):
    '''AAA'''
    logging.info(f"Running refine_inpaints.py on {args.input_dir}")

    # parse poses:
    if not (input_pdbs := glob(f"{args.input_dir}")): raise FileNotFoundError(f"No *.pdb files found at {args.input_dir}")
    inpaints = Poses(args.output_dir, glob(input_pdbs))

    # 




if __name__ == "__main__":
    import argparse
    
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be inpainted.")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    args = argparser.parse_args()

    main(args)
