#!/home/mabr3112/anaconda3/bin/python3.9

# import builtins
import sys
import os
import json
import logging
from glob import glob

# import dependencies
import pandas as pd

# import custom modules
sys.path.append("/home/mabr3112/riff_diff")
sys.path.append("/home/mabr3112/projects/iterative_refinement/")

import utils.plotting as plots
from iterative_refinement import *

def parse_cycle_scoreterms(prefix: str, scoreterms:str, weights:str) -> tuple[str]:
    ''''''
    return [prefix + x for x in scoreterms.split(",")], weights.split(",")

def main(args):
    '''AAA'''
    logging.info(f"Running refine_inpaints.py on {args.input_dir}")
    ref_dir = f"{args.input_dir}/ref_fragments"

    # parse poses:
    if not (input_pdbs := glob(f"{args.input_dir}/pdb_in/*.pdb")): raise FileNotFoundError(f"No *.pdb files found at {args.input_dir}")
    inpaints = Poses(args.output_dir, input_pdbs)
    if not os.path.isdir((plotdir := f"{inpaints.dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # merge fastrelax options into poses_df
    fastdesign_pose_opts_df = pd.read_json(f"{args.input_dir}/rosetta_pose_opts.json").T.reset_index().rename(columns={0: "fastdesign_pose_opts"})
    fastdesign_pose_opts_df["fastdesign_pose_opts"] = fastdesign_pose_opts_df["fastdesign_pose_opts"].str.replace("ref_fragments", f"{args.input_dir}/ref_fragments")
    print(fastdesign_pose_opts_df.values)
    inpaints.poses_df = inpaints.poses_df.merge(fastdesign_pose_opts_df, left_on="poses_description", right_on="index").drop(columns=["index"])

    if len(inpaints.poses_df) == len(inpaints.poses): print(f"Loading of Pose contigs into poses_df successful. Continuing to refinement.")
    else: raise ValueError(f"Merging of inpaint_opts into poses_df failed! Check if keys in inpaint_opts match with pose_names!")

    # initial diversify?
    initial_diversify = inpaints.create_relax_decoys(relax_options="-constrain_relax_to_start_coords -beta -ex1 -ex2", n=args.num_fastdesign_inputs, prefix="initial_diversify")

    # initialize DataFrame for plotting cycle trajectory:
    cycle_df = pd.DataFrame()

    # refinement cycles:
    for i in range(1, args.cycles + 1):
        cycle_prefix = f"cycle_{str(i).zfill(4)}"
        
        # run Constraint-Biased FastDesign RosettaScript
        fastdesign_opts = f"-parser:protocol /home/mabr3112/riff_diff/rosetta/refine.xml -beta -ex1 -ex2"
        fastdesign = inpaints.rosetta("rosetta_scripts.default.linuxgccrelease", options=fastdesign_opts, pose_options=inpaints.poses_df["fastdesign_pose_opts"].to_list(), n=args.num_fastdesign_outputs, prefix=f"{cycle_prefix}_fastdesign")

        # Calculate Motif BB-Ca RMSD 
        fastdesign_rmsd = inpaints.calc_motif_bb_rmsd_dir(ref_pdb_dir=ref_dir, ref_motif=inpaints.poses_df["motif_res"].to_list(), target_motif=inpaints.poses_df["motif_res"].to_list(), metric_prefix=f"{cycle_prefix}", remove_layers=2)

        # redesign Sequence with ProteinMPNN
        mpnn_designs = inpaints.mpnn_design(mpnn_options="--num_seq_per_target={args.num_mpnn_seqs} --sampling_temp={args.mpnn_sampling_temp}", prefix=f"{cycle_prefix}_mpnn", fixed_positions_col="mpnn_fixedpos")
        mpnn_filter = inpaints.filter_poses_by_score(args.num_esm_inputs, "mpnn_score", prefix="{cycle_prefix}_mpnn_seqfilter", remove_layer=1)

        # predict sequences with ESMFold and calculate stats
        esm_preds = inpaints.predict_sequences(run_ESMFold, prefix=f"{cycle_prefix}_esm")
        esm_bb_ca_rmsd = inpaints.calc_bb_rmsd_dir(ref_pdb_dir=fastdesign, ref_chains=["A"], pose_chains=["A"], remove_layers=1, metric_prefix=f"{cycle_prefix}_esm")
        esm_bb_ca_motif_rmsd =  inpaints.calc_motif_bb_rmsd_dir(ref_pdb_dir=ref_dir, ref_motif=inpaints.poses_df["motif_res"].to_list(), target_motif=inpaints.poses_df["motif_res"].to_list(), metric_prefix=f"{cycle_prefix}_esm_bb_ca", remove_layers=3)

        # filter down by desired scoreterms:
        cycle_scoreterms, cycle_scoreterm_weights = parse_cycle_scoreterms(cycle_prefix, args.cycle_filter_scoreterms, args.cycle_filter_scoreterm_weights)
        esm_comp_score = inpaints.calc_composite_score(f"{cycle_prefix}_esm_comp_score", cycle_scoreterms, cycle_scoreterm_weights)

        # if in last cycle, then do not reindex and do not filter!
        if i == args.cycles: break
        cycle_filter = inpaints.filter_poses_by_score(args.num_fastdesign_inputs, f"{cycle_prefix}_esm_comp_score", prefix=f"{cycle_prefix}_esm_filter", remove_layer=3, plot=cycle_scoreterms)

        # reindex poses for next cycle
        reindexed = inpaints.reindex_poses(out_dir=f"{cycle_prefix}/reindexed_poses/", remove_layers=3)
    
    cols = [f"{cycle_prefix}_esm_plddt", "{cycle_prefix}_esm_bb_ca_rmsd", "{cycle_prefix}_esm_bb_ca_motif_rmsd", f"{cycle_prefix}_esm_catres_motif_heavy_rmsd"]
    titles = ["ESM pLDDT", "ESM BB-Ca RMSD", "ESM Motif-Ca RMSD", "ESM Catres\nSidechain RMSD"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,100), (0,15), (0,8), (0,8)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/esm_stats.png")

    print("Done")



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be inpainted.")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")
    argparser.add_argument("--cycles", type=int, default=3, help="Number of refinement cycles you would like to run.")

    # plotting
    argparser.add_argument("--plot_scoreterms", type=str, default="esm_plddt,esm_bb_ca_rmsd,esm_bb_ca_motif_rmsd,fastdesign_total_score", help="Scoreterms for which refinement trajectories should be plotted.")

    # cyclic refinement
    argparser.add_argument("--num_fastdesign_outputs", type=int, default=5, help="Number of poses that should be kept after FastDesign.")
    argparser.add_argument("--num_fastdesign_inputs", type=int, default=5, help="Number of inputs into fastdesign for each cycle.")
    argparser.add_argument("--num_mpnn_seqs", type=int, default=20, help="Number of sequences to generate using ProteinMPNN.")
    argparser.add_argument("--mpnn_sampling_temp", type=float, default=0.1, help="Sampling Temperature for ProteinMPNN")
    argparser.add_argument("--num_esm_inputs", type=int, default=5, help="Number of Sequences per backbone that should be predicted by ProteinMPNN.")
    argparser.add_argument("--cycle_filter_scoreterms", type=str, default="esm_plddt,esm_bb_ca_rmsd", help="Scoreterms that you want to filter the poses on during each cycle of refinement")
    argparser.add_argument("--cycle_filter_scoreterm_weights", type=str, default="-1,1", help="Weights for --cycle_filter_scoreterms. Both arguments need to have the same number of elements!")

    args = argparser.parse_args()
    main(args)
