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
from utils.plotting import PlottingTrajectory
import utils.pymol_tools
from iterative_refinement import *

def parse_cycle_scoreterms(prefix: str, scoreterms:str, weights:str) -> tuple[str]:
    ''''''
    return [prefix + "_" + x for x in scoreterms.split(",")], [float(x) for x in weights.split(",")]

def main(args):
    '''AAA'''
    logging.info(f"Running refine_inpaints.py on {args.input_dir}")
    ref_dir = f"{args.input_dir}/ref_fragments/"

    # parse poses:
    if not (input_pdbs := glob(f"{args.input_dir}/pdb_in/*.pdb")): raise FileNotFoundError(f"No *.pdb files found at {args.input_dir}")
    inpaints = Poses(args.output_dir, input_pdbs)
    if not os.path.isdir((plotdir := f"{inpaints.dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # merge fastrelax options into poses_df
    fastdesign_pose_opts_df = pd.read_json(f"{args.input_dir}/rosetta_pose_opts.json").T.reset_index().rename(columns={0: "fastdesign_pose_opts"})
    motif_res_df = pd.read_json(f"{args.input_dir}/motif_res.json").reset_index()
    fastdesign_pose_opts_df["fastdesign_pose_opts"] = fastdesign_pose_opts_df["fastdesign_pose_opts"].str.replace("ref_fragments", f"{args.input_dir}/ref_fragments")
    inpaints.poses_df = inpaints.poses_df.merge(fastdesign_pose_opts_df, left_on="poses_description", right_on="index").drop(columns=["index"])
    inpaints.poses_df = inpaints.poses_df.merge(motif_res_df, left_on="poses_description", right_on="index").drop(columns=["index"])

    # set reference poses into poses_df
    inpaints.poses_df["ref_poses"] = ref_dir + inpaints.poses_df["poses_description"].str + ".pdb"

    if len(inpaints.poses_df) == len(inpaints.poses): print(f"Loading of Pose contigs into poses_df successful. Continuing to refinement.")
    else: raise ValueError(f"Merging of inpaint_opts into poses_df failed! Check if keys in inpaint_opts match with pose_names!")

    # initial diversify?
    initial_diversify = inpaints.create_relax_decoys(relax_options="-constrain_relax_to_start_coords -beta -ex1 -ex2", n=args.num_fastdesign_inputs, prefix="initial_diversify")

    # initialize PlottingTrajectory objects for storing plots. 
    plot_dir = f"{inpaints.dir}/plots/"
    esm_plddt_traj = PlottingTrajectory(y_label="ESMFold pLDDT", location=f"{plot_dir}/esm_plddt_trajectory.png", title="ESMFold Trajectory", dims=(0,100))
    esm_bb_ca_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_bb_ca_trajectory.png", title="ESMFold BB-Ca\nRMSD Trajectory", dims=(0,10))
    esm_motif_ca_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_motif_ca_trajectory.png", title="ESMFold Motif-Ca\nRMSD Trajectory", dims=(0,8))
    esm_catres_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_catres_rmsd_trajectory.png", title="ESMFold Motif\nSidechain RMSD Trajectory", dims=(0,8))
    fastdesign_total_score_traj = PlottingTrajectory(y_label="Rosetta total score [REU]", location=f"{plot_dir}/rosetta_total_score_trajectory.png", title="FastDesign Total Score Trajectory")
    fastdesign_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/fastdesign_rmsd_traj.png", title="FastDesign Motif RMSD Trajectory")

    # refinement cycles:
    for i in range(1, args.cycles + 1):
        cycle_prefix = f"cycle_{str(i).zfill(4)}"
        
        # run Constraint-Biased FastDesign RosettaScript
        fastdesign_opts = f"-parser:protocol /home/mabr3112/riff_diff/rosetta/refine.xml -beta -ex1 -ex2"
        fastdesign = inpaints.rosetta("rosetta_scripts.default.linuxgccrelease", options=fastdesign_opts, pose_options=inpaints.poses_df["fastdesign_pose_opts"].to_list(), n=args.num_fastdesign_outputs, prefix=f"{cycle_prefix}_fastdesign")

        # Calculate Motif BB-Ca RMSD 
        fastdesign_rmsd = inpaints.calc_motif_bb_rmsd_dir(ref_pdb_dir=ref_dir, ref_motif=inpaints.poses_df["motif_residues"].to_list(), target_motif=inpaints.poses_df["motif_residues"].to_list(), metric_prefix=f"{cycle_prefix}", remove_layers=2)
        fastdesign_rmsd_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_motif_rmsd"].to_list(), cycle_prefix)
        fastdesign_total_score_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_fastdesign_total_score"], cycle_prefix)

        # calculate mixed score between total-score and motif_rmsd:
        fd_st = [f"{cycle_prefix}_fastdesign_total_score", f"{cycle_prefix}_motif_rmsd"]
        comp_score = inpaints.calc_composite_score(f"{cycle_prefix}_fastdesign_comp_score", fd_st, [1,1])
        fd_filter = inpaints.filter_poses_by_score(10, f"{cycle_prefix}_fastdesign_comp_score", prefix=f"{cycle_prefix}_fastdesign_filter", remove_layers=2, plot=fd_st)

        # redesign Sequence with ProteinMPNN
        mpnn_designs = inpaints.mpnn_design(mpnn_options=f"--num_seq_per_target={args.num_mpnn_seqs} --sampling_temp={args.mpnn_sampling_temp}", prefix=f"{cycle_prefix}_mpnn", fixed_positions_col="fixed_residues")
        mpnn_filter = inpaints.filter_poses_by_score(args.num_esm_inputs, "mpnn_score", prefix=f"{cycle_prefix}_mpnn_seqfilter", remove_layers=1)

        # predict sequences with ESMFold and calculate stats
        esm_preds = inpaints.predict_sequences(run_ESMFold, prefix=f"{cycle_prefix}_esm")
        esm_bb_ca_rmsd = inpaints.calc_bb_rmsd_dir(ref_pdb_dir=fastdesign, ref_chains=["A"], pose_chains=["A"], remove_layers=1, metric_prefix=f"{cycle_prefix}_esm")
        esm_bb_ca_motif_rmsd = inpaints.calc_motif_bb_rmsd_dir(ref_pdb_dir=ref_dir, ref_motif=inpaints.poses_df["motif_residues"].to_list(), target_motif=inpaints.poses_df["motif_residues"].to_list(), metric_prefix=f"{cycle_prefix}_esm_bb_ca", remove_layers=3)
        esm_catres_rmsd = inpaints.calc_motif_heavy_rmsd_dir(ref_pdb_dir=ref_dir, ref_motif=inpaints.poses_df["fixed_residues"].to_list(), target_motif=inpaints.poses_df["fixed_residues"].to_list(), metric_prefix=f"{cycle_prefix}_esm_catres", remove_layers=3)

        # Plot trajectory:
        esm_plddt_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_esm_plddt"], cycle_prefix)
        esm_bb_ca_rmsd_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_esm_bb_ca_rmsd"], cycle_prefix)
        esm_motif_ca_rmsd_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_esm_bb_ca_motif_rmsd"], cycle_prefix)
        esm_catres_rmsd_traj.add_and_plot(inpaints.poses_df[f"{cycle_prefix}_esm_catres_motif_heavy_rmsd"], cycle_prefix)

        # filter down by desired scoreterms:
        cycle_scoreterms, cycle_scoreterm_weights = parse_cycle_scoreterms(cycle_prefix, args.cycle_filter_scoreterms, args.cycle_filter_scoreterm_weights)
        esm_comp_score = inpaints.calc_composite_score(f"{cycle_prefix}_esm_comp_score", cycle_scoreterms, cycle_scoreterm_weights)

        # if in last cycle, then do not reindex and do not filter!
        if i == args.cycles: break
        cycle_filter = inpaints.filter_poses_by_score(args.num_fastdesign_inputs, f"{cycle_prefix}_esm_comp_score", prefix=f"{cycle_prefix}_esm_filter", remove_layers=3, plot=cycle_scoreterms)

        # reindex poses for next cycle
        reindexed = inpaints.reindex_poses(out_dir=f"{cycle_prefix}/reindexed_poses/", remove_layers=3)
    
    # plot final results   
    cols = [f"{cycle_prefix}_esm_plddt", f"{cycle_prefix}_esm_bb_ca_rmsd", f"{cycle_prefix}_esm_bb_ca_motif_rmsd", f"{cycle_prefix}_esm_catres_motif_heavy_rmsd"]
    titles = ["ESM pLDDT", "ESM BB-Ca RMSD", "ESM Motif-Ca RMSD", "ESM Catres\nSidechain RMSD"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,100), (0,15), (0,8), (0,8)]
    _ = plots.violinplot_multiple_cols(inpaints.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/esm_stats.png")
    
    # create pymol alignment script:
    results_dir = f"{inpaints.dir}/results" + "/"
    out_filter_comp_score = inpaints.calc_composite_score("out_filter_comp_score", [f"{cycle_prefix}_esm_plddt", f"{cycle_prefix}_esm_bb_ca_rmsd"], [-1,1])
    top_pdb_df = inpaints.poses_df.sort_values(by="out_filter_comp_score").head(args.top_n)
    pml_script_path = utils.pymol_tools.pymol_alignment_scriptwriter(top_pdb_df, scoreterm="out_filter_comp_score", top_n=top_n, path_to_script=f"{results_dir}/align.pml", pose_col="poses_description", ref_pose_col="ref_poses", motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="motif_residues", ref_fixed_res_col="fixed_residues")

    # copy top pdbs into results directory.
    for idx in top_pdb_df.index:
        shutil.copy(top_pdb_df.loc[idx]["ref_poses"], results_dir)
        shutil.copy(top_pdb_df.loc[idx][f"{cycle_prefix}_esm_location"], results_dir)

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
    argparser.add_argument("--num_mpnn_seqs", type=int, default=50, help="Number of sequences to generate using ProteinMPNN.")
    argparser.add_argument("--mpnn_sampling_temp", type=float, default=0.1, help="Sampling Temperature for ProteinMPNN")
    argparser.add_argument("--num_esm_inputs", type=int, default=25, help="Number of Sequences per backbone that should be predicted by ProteinMPNN.")
    argparser.add_argument("--cycle_filter_scoreterms", type=str, default="esm_plddt,esm_bb_ca_motif_rmsd", help="Scoreterms that you want to filter the poses on during each cycle of refinement")
    argparser.add_argument("--cycle_filter_scoreterm_weights", type=str, default="-1,1", help="Weights for --cycle_filter_scoreterms. Both arguments need to have the same number of elements!")

    # final results
    argparser.add_argument("--top_n", type=int, default=50, help="Number of top outputs for plotting.")

    args = argparser.parse_args()
    main(args)
