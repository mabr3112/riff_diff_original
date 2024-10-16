#!/home/mabr3112/anaconda3/bin/python3
#
#  This script is intended to be used to extract specific scores from a large scorefile
#  to save storage space.
#
#
########################################

import os
import re
import pandas as pd

def df_from_rosetta_scorefile(path_to_file: str) -> pd.DataFrame:
    '''
    Reads Rosetta Scorefile and returns Pandas DataFrame
    '''
    return pd.read_csv(path_to_file, delim_whitespace=True, header=[1], na_filter=True)

def filter_by_pattern(lst: list, pattern: str) -> list:
    """
    Keep only the elements in a list of strings that match a specified regular expression pattern.
    
    Args:
        lst (list): The input list of strings.
        pattern (str): The regular expression pattern to match.
    
    Returns:
        list: A new list containing only the elements in `lst` that match `pattern`.
    
    Examples:
        filter_by_pattern(["apple", "banana", "cherry"], r"^a")
        # Returns: ["apple"]
        
        filter_by_pattern(["apple", "banana", "cherry"], r"^b")
        # Returns: ["banana"]
        
        filter_by_pattern(["apple", "banana", "cherry"], r"y$")
        # Returns: ["cherry"]
    """
    return [s for s in lst if re.search(pattern, s)]

def main(args):
    '''
    '''
    if args.input_path.endswith(".json"):
        print(f"Scorefile {args.input_path} will be read as json file.")
        df = pd.read_json(args.input_path)
    elif args.input_path.endswith(".sc"):
        print(f"Your scorefile {args.input_path} ends with *.sc This is assumed to be a Rosetta scorefile and will be treated as such.")
        df = df_from_rosetta_scorefile(args.input_path)
    else:
        raise TypeError("ERROR: Your file is of unknown format, and cannot be read.")
    
    if args.regex: df = df[filter_by_pattern(df.columns, args.regex)]
    if args.startswith: df = df[[x for x in df.columns if x.startswith(args.startswith)]]
    if args.endswith: df = df[[x for x in df.columns if x.endswith(args.endswith)]]

    df.to_json(args.output_path)

    return None


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, required=True, help="Path to the directory that contains *.pdb files where chains have to be reassigned.")
    argparser.add_argument("--output_path", type=str, required=True, help="Path to the output directory where the *.pdb files with reassigned chains should be stored.")
    argparser.add_argument("--startswith", type=str, help="Extract scores that start with the specified string.")
    argparser.add_argument("--endswith", type=str, help="Extract scores that end with the specified string.")
    argparser.add_argument("--regex", type=str, help="Extract scores that contain the specified regex.")
    args = argparser.parse_args()

    if not os.path.isfile(args.input_path): raise FileNotFoundError

    main(args)
