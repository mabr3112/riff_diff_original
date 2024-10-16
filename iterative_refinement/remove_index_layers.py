#!/home/mabr3112/anaconda3/bin/python3.9
#
#   This script removes the "_0001" index layers from pdb-files. 
#   If multiple pdb-files with the same name exist, it adds 
#   a new "_0001" layer to the pdb-files.
#
#   Usage: ./remove_index_layers.py --input_data dir --remove_layers int
#
############################################################

import os
from subprocess import run
from glob import glob
import shutil

def remove_index_layers(input_data, output_dir: str, n_layers: int, force_reindex=None, keep_layers=False) -> dict:
    '''
    Removes the "_0001" index layers from pdb-files.
    If multiple pdb-files with the same name exist, it adds
    a new "_0001" layer to the pdb-files.
    '''
    # collect all *.pdb files into list
    if type(input_data) == str:
        pl = glob((globstr := f"{input_data}/*.pdb"))
    elif type(input_data) == list:
        pl = input_data
    if not pl: raise FileNotFoundError(f"ERROR: no *.pdb files found in {input_data}")

    # sanity
    assert type(n_layers) == int
    directionality = 1 if keep_layers else -1

    # create renaming dictionary
    path_and_file_split = [("/".join(split_path[:-1])+"/", split_path[-1]) for split_path in [fpath.split("/") for fpath in pl]]

    rd = {path+"_".join(filename.split("_")[:directionality*n_layers]): [] for path, filename in path_and_file_split}
    rd_T = {path+filename: path+"_".join(filename.split("_")[:directionality*n_layers]) for path, filename in path_and_file_split}
    for k, v in rd_T.items():
        rd[v].append(k)
    
    copy_dict = dict()

    # check any value (list with names) in renaming dict is longer than 1.
    if any(len(x) > 1 for x in rd.values()):
        print(f"After removing layers, found multiple pdb-files with the same name.\nAdding index layer to files.")
        for k in rd:
            for i, pdb in enumerate(rd[k]):
                filename = f"{k.split('/')[-1].replace('.pdb', '')}_{str(i+1).zfill(4)}.pdb"
                new_loc = output_dir + "/" + filename
                copy_dict[pdb] = new_loc
                shutil.copy(pdb, new_loc)
        return copy_dict

    else:
        print(f"Each pdb-file is unique.")
        #print(rd)
        for k in rd:
            filename = f"{k.split('/')[-1]}.pdb"
            if force_reindex:
                print(f"Option --force_reindex set to {force_reindex}. Index layer _0001 will be added to pdb-filename.")
                filename = filename.replace(".pdb", "_0001.pdb")
            else:
                print(f"Option --force_reindex is not set. No index layers will be added.")
            outfile = f"{output_dir}/{filename}"
            copy_cmd = f"cp {rd[k][0]} {outfile}"
            copy_dict[rd[k][0]] = outfile

            #print(copy_cmd)
            run(copy_cmd, shell=True, stdout=True, stderr=True, check=True)
            if not os.path.isfile(outfile): raise FileNotFoundError(f"File {outfile} not found. Copying failed.")

    return copy_dict

def main(args):
    '''

    '''
    remove_index_layers(args.input_data, args.output_dir, args.n_layers, args.force_reindex)

    return None

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_data", type=str, help="Directory that contains all pdb files that should be renamed.")
    argparser.add_argument("--output_dir", type=str, help="Path to directory where your renamed pdb-files should be saved to.")
    argparser.add_argument("--n_layers", type=int, help="(Int) Number of layers (\"_0001\") that should be removed from the .pdb-files.")
    argparser.add_argument("--force_reindex", help="Force script to add an index layer (_0001) at the end of the file, even if each .pdb file is unique.")
    args = argparser.parse_args()
    
    if not os.path.isdir(args.input_data): raise FileNotFoundError(f"Directory {args.input_data} not found. Are you sure it is the correct path?")
    if not args.output_dir: args.output_dir = args.input_data + "/renamed_pdbs"

    main(args)
