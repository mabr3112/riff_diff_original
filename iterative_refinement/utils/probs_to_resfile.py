import numpy as np

def read_probs(mpnn_probs_path: str) -> np.array:
    ''''''
    probs = np.load(mpnn_probs_path)["log_p"]
    return probs if len(probs) > 1 else probs[0]

def convert_res_to_mask(input_res: list, length: int, threshold:float=0.1, threshold_res:float=0.01) -> list:
    '''
    Convert a list of residue indices to a mask.

    Args:
        input_res (list): List of residue indices.
        length (int): Length of the mask.
        threshold (float): Threshold for non-residue positions.
        threshold_res (float): Threshold for residue positions.

    Returns:
        list: Mask.
    '''
    return [threshold_res if i+1 in input_res else threshold for i in range(length)]

def resfile_from_aa_array(aa_array: "list[list]") -> str:
    # header is always the same
    start = "ALLAA\nstart\n"

    # body contains residue specifications:
    body = "\n".join([f"{i+1} A PIKAA {''.join(row)}".replace("X", "") for i, row in enumerate(aa_array)])
    return "\n".join([start, body])

def probs_to_resfile(log_probs: np.array, threshold:float=0.1, mask:list=None, alphabet:str="ACDEFGHIKLMNPQRSTVWYX") -> str:
    '''
    Convert log probabilities to a Rosetta resfile format.

    Args:
        log_probs (np.array): Log probabilities of amino acids.
        threshold (float): Probability threshold for amino acid selection.
        mask (list): List of indices to mask.
        alphabet (str): Amino acid alphabet.

    Returns:
        str: Resfile format string.
        '''
    # if multiple backbones were passed to read_probs():
    if len(log_probs.shape) > 2: return [probs_to_resfile(log_probs=log_prob, threshold=threshold, mask=mask) for log_prob in log_probs]

    # define alphabet:
    alphabet = np.array(list(alphabet), dtype=object)

    # calculate probs
    probs = np.exp(log_probs)
    if type(threshold) == float: reslist = [list(alphabet[row >= threshold]) if np.any(row >= threshold) else list(alphabet[np.argmax(row)]) for row in probs]
    elif len(threshold) == len(probs): reslist = [list(alphabet[row >= threshold[i]]) if np.any(row >= threshold[i]) else list(alphabet[np.argmax(row)]) for i, row in enumerate(probs)]
    else: raise ValueError(f"Invalid parameter specification for :threshold: {threshold}")

    return resfile_from_aa_array(reslist)

def write_resfile(resfile_str: str, save_path: str) -> str:
    with open(save_path, 'w') as f: f.write(resfile_str)
    return save_path

def write_probs_to_resfile_from_path(input_path: str, threshold=0.1, motif_threshold:float=None, motif_list:list=None, save_path:str=None) -> str:
    '''
    Write a Rosetta resfile from a probability file.

    Args:
        input_path (str): Path to the input probability file.
        threshold (float): Probability threshold for amino-acid selection.
        motif_threshold (float): Threshold for residue positions in the motif.
        motif_list (list): List of residue indices (starting from 1) in the motif.
        save_path (str): Path to save the resfile.

    Returns:
        str: Path to the resfile.
    '''
    probs = read_probs(input_path)

    # if motif is given, apply motif mask:
    if motif_list: threshold = convert_res_to_mask(motif_list, len(probs), threshold=threshold, threshold_res=motif_threshold or threshold)

    # compile resfile string:
    resfile_str = probs_to_resfile(probs, threshold=threshold)

    return write_resfile(resfile_str, save_path or input_path.replace(".npz", ".res"))
