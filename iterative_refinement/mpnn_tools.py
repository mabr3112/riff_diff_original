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
import json

def write_mpnn_jsonl(poses: list[str], values, json_filename: str, iterate=False) -> None:
    '''
    Writes jsonl-files for ProteinMPNN.
    Function can be called with a single path or a list of paths to pdb-files in the <poses> argument.
    Function can be called with a single dictionary or a list of dictionaries as <design_chains> argument.
    A design-chain dictionary (chain_id_jsonl) for ProteinMPNN requires the following format: []
    
    Args:
        <poses>                         list[str]: List of paths to .pdb files for which you want to write chain_id_jsonl entries.
                                        str: path to the .pdb file for which you want to write a chain_id_jsonl entry.
        <jsonl_filename>                str: Name of your output file.
        <values>                        Values to be set as value for mpnn_jsonl dictionary. (Key = poses)
        <iterate>                       bool: Set to True, if you want to iterate over values and poses and assign (pose: value) pairs based on iteration.
    Returns:
        json_filename
    '''
    # sanity check <poses> argument
    if type(poses) not in [str, list]: raise ValueError(f"ERROR: poses argument of write_design_chains() function does only take [list|str] as arguments. Type of poses: {type(poses)}\n<poses>\n{poses}")
    
    # if poses is a string, just set <pose_values> as value for pose:
    if type(poses) == str: chain_id_dict = {poses: values}
        
    # if poses is a list, either set <pose_values> as values for all poses, or assign each pose the corresponding value from pose_values (by zipping lists)
    elif not iterate: chain_id_dict = {p: values for p in poses}
    elif len(values) == len(poses): chain_id_dict = {k: v for k, v in zip(poses, values)}
        
    # if lengths of poses and pose_values mismatch, raise error:
    else: raise ValueError(f"ERROR: values and poses arguments are not of the same length. If you want <values> to be set as value for every pose in <poses>, set <iterate> to False.\n<design_chains>:\n{values}\n\n<poses>:\n{poses}")
    
    # write dictionary containing design_chain_ids into json file for ProteinMPNN:
    with open(json_filename, 'w') as f:
        f.write(json.dumps(chain_id_dict))
    
    return json_filename

def check_design_chains(design_chains, poses):
    '''
    Checks structure of <design_chains> argument and returns True or False.
    '''
    if type(design_chains) == str:
        return True
    if type(poses) == str:
        return False
    
    if len(design_chains) <= 2:
        print(f"Length of argument <design_chains> of mpnn_design() [ProteinMPNN] for specifying designable chains is {len(design_chains)}.\n<design_chains> will be set as is for all poses!\nIf you want to set pose-specific designable chains, assign the designable chain lists to the poses_df!")
        return False
    elif len(design_chains) == len(poses):
        return True
    else:
        print(f"Argument <design_chains> of mpnn_design() [ProteinMPNN] for specifying designable chains has different length ({len(design_chains)}) than <poses> ({len(poses)}).\n<design_chains> will be set as is for all poses!")
        return False