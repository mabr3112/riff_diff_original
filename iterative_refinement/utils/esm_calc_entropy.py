#!/home/mabr3112/anaconda3/envs/esm/bin/python

import sys
import os
import torch
import esm
from time import time
import numpy as np
import json
import pandas as pd
import subprocess
import time

alphabet = "ACDEFGHIKLMNPQRSTVWY"
alphabet_idx = [ 5, 23, 13,  9, 18,  6, 21, 12, 15,  4, 20, 17, 14, 16, 10,  8, 11,  7, 22, 19]
standard_alphabet_dict = {char: idx for idx, char in zip(alphabet_idx, alphabet)}

def wait_for_job(jobname: str, interval=5) -> str:
    '''
    Waits for a slurm job to be completed, then prints a "completed job" statement and returns jobname.
    <jobname>: name of the slurm job. Job Name can be set in slurm with the flag -J
    <interval>: interval in seconds, how log to wait until checking again if the job still runs.
    '''
    # Check if job is running by capturing the length of the output of squeue command that only returns jobs with <jobname>:
    while len(subprocess.run(f'squeue -n {jobname} -o "%A"', shell=True, capture_output=True, text=True).stdout.strip().split("\n")) > 1:
        time.sleep(interval)
    print(f"Job {jobname} completed.\n")
    time.sleep(10)
    return jobname

def add_timestamp(x: str) -> str:
    '''
    Adds a unique (in most cases) timestamp to a string using the "time" library.
    Returns string with timestamp added to it.
    '''
    return "_".join([x, f"{str(time.time()).replace('.', '')}"])

def generate_masked_seqs(description: str, seq: str) -> "dict[str]":
    '''Creates an array of len(seq) sequences where each position in the sequence is replaced with <mask> once.'''
    return {f"{description}_{str(i).zfill(4)}": seq[:i] + "<mask>" + seq[i+1:] for i, char in enumerate(seq)}

def predict_masked_probs(description: str, seq: str, model, batch_converter, alphabet) -> torch.tensor:
    '''Runs forward pass and measures time needed for forward pass'''
    # measure how long preds take
    a = time()
    
    # create array with masked sequences
    data = [(desc, seq) for desc, seq in generate_masked_seqs(description, seq).items()]
    
    # prep ESM
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # run preds
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # results workup
    log_probs = torch.softmax(results["logits"], dim=-1)
    del results
    
    return log_probs

def predict_single_probs(fa_path: str, output_dir: str, script_path: str, description: str) -> np.array:
    '''runs esm extraction script on single-entry fasta.'''
    # prep and errorcheck inputs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    if not os.path.isfile(fa_path):
        raise FileNotFoundError(f"File does not exist: {fa_path}")

    # run script
    jn = add_timestamp(f"esm_entropy")
    cmd = f"sbatch -c1 -J {jn} -e extract.err -o extract.out -vvv --wrap=\"/home/mabr3112/anaconda3/envs/esm/bin/python {script_path} --return_logits --include mean --toks_per_batch 132 esm2_t33_650M_UR50D {fa_path} {output_dir}\""
    print(cmd)
    subprocess.run(cmd, shell=True, stderr=True, stdout=True, check=True)
    
    # wait for output
    jn = wait_for_job(jn)

    # check for output
    out_fn = f"{output_dir}/{description}.pt"
    if not os.path.isfile(out_fn):
        raise FileNotFoundError(f"Output file of extract.py not found. Something went wrong with running the script. Check error and output logs!\nOutput file {out_fn}")

    # read output
    out_in = torch.load(out_fn)
    logprobs = out_in["logits"][0,1:-1,4:24] # extract first, 1:-1 removes start and end token, 4:24 are residue ids
    return logprobs.numpy()

def probs_from_file(file_path: str) -> np.array:
    '''extracts perresidue probs from esm output. (needs modified esm script that extracts logits.'''
    out_in = torch.load(out_fn)
    logprobs = out_in["logits"][0,1:-1,4:24] # extract first, 1:-1 removes start and end token, 4:24 are residue ids
    return logprobs

def extract_masked_preds(raw_preds: torch.tensor) -> torch.tensor:
    '''extracted masked position from raw_preds (diagonal of output matrix) for output of 'predict_masked_seqs'. '''
    return [raw_preds[i][i+1] for i, _ in enumerate(raw_preds)]

def extract_seqprobs(masked_preds: list, seq: str, alphabet_dict:dict=standard_alphabet_dict) -> torch.tensor: 
    ''''''
    seq_idx = [alphabet_dict[char] for char in seq]
    return np.array([masked_preds[i][idx] for i, idx in enumerate(seq_idx)])

def calc_perplexity_from_masked_preds(masked_preds: np.array):
    '''calculates pseudo_perplexity according to ESMFold paper for sequence of masked predictions.'''
    return np.exp(-np.sum(masked_preds) / masked_preds.shape[0])

def read_fasta(input_path: str) -> "list[str]":
    '''reads sequences and descriptions out of fasta files'''
    with open(input_path, 'r') as f:
        x = f.read()

    # split along fasta entries (if multiline fasta)
    records = [[l.strip() for l in x.split("\n")] for x in x.split(">") if x]

    # split into description and sequence and return as dictionary {description: sequence}
    return {entry[0]: "".join(entry[1:]) for entry in records}

def generate_custom_colormap(t_points:"list[float]"=[0,0.5,1]):
    '''Creates a custom colormap
    '''
    # Define the RGB values for dark red, yellow, and gray
    dark_red = [139/255, 0, 0]
    yellow = [1, 0.8, 0]
    gray = [0.5, 0.5, 0.5]
   
    # Define the transition points for the colors
    t_points = [0, 0.5, 1]
   
    # Initialize the colormap
    colormap = []
   
    # Generate the colormap
    for t in np.linspace(0, 1, 256):
        if t < t_points[1]:
            weight = (t - t_points[0]) / (t_points[1] - t_points[0])
            color = [(1 - weight) * dark_red[i] + weight * yellow[i] for i in range(3)]
        else:
            weight = (t - t_points[1]) / (t_points[2] - t_points[1])
            color = [(1 - weight) * yellow[i] + weight * gray[i] for i in range(3)]
        colormap.append(color)

    return np.array(colormap)

def convert_values_to_colors(values: "list[float]", t_points:"list[float,float,float]"=[0, 0.5, 1], color1:"list[float]"=[1, 0, 0], color2:"list[float]"=[0, 1, 0], color3:"list[float]"=[0,0,1]) -> "list[list[float]]":
    '''Takes list of scalars and returns list of RGB values (as lists)'''
    out_colors = []

    # make sure your colors are np.arrays
    color1 = np.array(color1)
    color2 = np.array(color2)
    color3 = np.array(color3)
    
    # convert values to colors based on three colors and transitionpoints
    for value in values:
        if value <= t_points[0]:
            color = list(color1)
        elif t_points[0] < value < t_points[1]:
            weight = value / (t_points[1] - t_points[0])
            color = list((1-weight)*color1 + weight * color2)
        elif t_points[1] < value < t_points[2]:
            weight = (value - t_points[1]) / (t_points[2] - t_points[1])
            color = list((1-weight) * color2 + weight * color3)
        elif value >= t_points[2]:
            color = list(color3)

        out_colors.append(color)
    return out_colors

def create_entropy_pml_coloring(entropies: "list[float]", pdb_description: str, t_points:"list[float,float,float]"=[0, 2, 4], color1:"list[float]"=[139/255, 0, 0], color2:"list[float]"=[0.5,0.5,0.5], color3:"list[float]"=[1, 0.8, 0]) -> str:
    '''Writes pml script-string'''
    # in case entropies is still a numpy array
    if not type(entropies) == list: entropies = list(entropies)

    # create color-list based on entropy values
    colors = convert_values_to_colors(values=entropies, t_points=t_points, color1=color1, color2=color2, color3=color3)

    out = f"load {pdb_description}.pdb\n"

    for i, color in enumerate(colors):
        i = i+1
        out += f"set_color tmp{str(i)}, {color}\n"
        out += f"color tmp{str(i)}, resi {str(i)} and {pdb_description}\n"

    return out

def entropy(prob_distribution: np.array) -> float:
    # Filter out zero probabilities to avoid log(0)
    prob_distribution = prob_distribution[prob_distribution > 0]
    
    # Compute entropy
    return -np.sum(prob_distribution * np.log2(prob_distribution))


def main(args):
    ""
    print(f"\n{'#'*50}\nRunning calc_perplexity.py on {args.input_fasta}\n{'#'*50}\n")

    # parse output scorefile name:
    filename = args.input_fasta.split("/")[-1].replace(".fa","")

    # calc perplexities
    tt = time.time()

    # load model (650M params)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # parse inputs:
    pred_dict = read_fasta(args.input_fasta) # returns dict {"description": "seq", ...}

    # go over every fasta-file and output immediately
    out_dict = dict()
    for description in pred_dict:
        out_dict[description] = dict()
        if not args.singlepass:
            # run masked preds
            raw_outs = predict_masked_probs(description, seq=pred_dict[description], model=model, batch_converter=batch_converter, alphabet=alphabet)

            # extract masked probabilities
            masked_probs = np.array([x.numpy()[4:24] for x in extract_masked_preds(raw_outs)])
        else:
            print(f"Calculating Singlepass probs")
            masked_probs = predict_single_probs(fa_path = args.input_fasta, output_dir=f"{args.output_dir}/singlepass/", script_path=args.singlepass_script_path, description=description)

        # write output
        perres_entropy = [entropy(residue_distribution) for residue_distribution in masked_probs]
        out_dict[description]["perresidue_probabilities"] = masked_probs
        out_dict[description]["perresidue_entropy"] = [entropy(residue_distribution) for residue_distribution in masked_probs]
        out_dict[description]["sequence_entropy"] = np.mean(perres_entropy)

        df = pd.DataFrame(out_dict).T.reset_index().rename(columns={"index": "description"})
        df.to_json(f"{args.output_dir}/{filename}_scores.json")

        # write .pml script to color by residue
        with open(f"{args.output_dir}/{description}_entropy_coloring.pml", "w") as f:
            f.write(create_entropy_pml_coloring(perres_entropy, description, t_points=[0,2,4]))

    print(f"Finished predicting pseudo-perplexities for {len(pred_dict)} sequences in {time() - tt} seconds")

def str_to_bool(input_str: str) -> bool:
    if input_str in ["True", "true", "1", "yes", "Yes"]: return True
    elif input_str in ["False", "false", "0", "no", "No"]: return False
    else: raise ValueError(f"Input '{input_str}' cannot be interpreted as bool!")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_fasta", type=str, required=True, help="FastA formatted file (can be single, or multile, or multiple fastas in one file)")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory. Default: path/to/input_fasta/esm_perplexity.json")
    argparser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D", help="Which model from the transformers library would you like to use to predict perplexity?")
    argparser.add_argument("--singlepass", type=str, default="False", help="Do you want to return perplexity of a singlepass through the network?")
    argparser.add_argument("--singlepass_script_path", type=str, default="/home/mabr3112/esm/scripts/extract.py", help="Path to esm script path to extract singlepass perplexities. Use /path/to/esm/scripts/extract.py")
    argparser.add_argument("--from_file", type=str, default=None, help="Path to an extract.py output that contains logits. This extracts the residue probabilities directly from the precomputed file")

    args = argparser.parse_args()
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    
    # prep inputs
    args.singlepass = str_to_bool(args.singlepass)

    main(args)


