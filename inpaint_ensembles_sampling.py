#!/home/mabr3112/anaconda3/bin/python3.9
import sys
sys.path.append("/home/mabr3112/riff_diff")
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]

import json
from iterative_refinement import *
from glob import glob
import utils.plotting as plots

def main(args):
    # print Status
    print(f"\n{'#'*50}\nRunning inpaint_ensembles.py on {args.input_dir}\n{'#'*50}\n")

    # Parse Poses
    pdb_dir = f"{args.input_dir}/pdb_in/"
    ensembles = Poses(args.output_dir, glob(f"{pdb_dir}/*.pdb"))
    ensembles.max_inpaint_gpus = args.max_inpaint_gpus

    # add contigs from json file to poses_df for pose options call
    with open(f"{args.input_dir}/inpaint_pose_opts.json", 'r') as f:
        contigs_dict = json.loads(f.read())
    with open(f"{args.input_dir}/fixed_res.json", 'r') as f:
        fixedres_dict = json.loads(f.read())
    with open(f"{args.input_dir}/motif_res.json", 'r') as f:
        motif_res_dict = json.loads(f.read())
    with open(f"{args.input_dir}/res_identities.json", 'r') as f:
        identity_dict = json.loads(f.read())

    # properly setup Contigs DataFrame
    residue_identities_df = pd.DataFrame.from_dict(identity_dict, orient="index").reset_index().rename(columns={"index": "0description", 0: "catres_identities"})
    pose_opts_df = pd.DataFrame.from_dict(contigs_dict, orient="index").reset_index().rename(columns={"index": "1description", 0: "inpainting_pose_opts"})
    fixedres_df = pd.DataFrame.from_dict(fixedres_dict, orient="index").reset_index().rename(columns={"index": "2description", 0: "fixed_residues"})
    motif_res_df = pd.DataFrame.from_dict(motif_res_dict, orient="index").reset_index().rename(columns={"index": "3description", 0: "motif_residues"})
    motif_cols = ["fixed_residues", "motif_residues"]

    # replace translation_magnitude and rotation_degrees in pose opts with arguments from commandline:
    pose_opts_df["inpainting_pose_opts"] = pose_opts_df["inpainting_pose_opts"].str.replace("translate_sampling_magnitude", str(args.translation_sampling_magnitude))
    pose_opts_df["inpainting_pose_opts"] = pose_opts_df["inpainting_pose_opts"].str.replace("rotate_sampling_degrees", str(args.rotation_sampling_degrees))

    # Read scores of selected paths from ensemble_evaluator and store them in poses_df:
    path_df = pd.read_json(f"{args.input_dir}/selected_paths.json").reset_index().rename(columns={"index": "rdescription"})
    ensembles.poses_df = ensembles.poses_df.merge(path_df, left_on="poses_description", right_on="rdescription")
    print(len(ensembles.poses_df))

    # Now merge with ensembles.poses_df:
    ensembles.poses_df = ensembles.poses_df.merge(residue_identities_df, left_on="poses_description", right_on="0description")
    ensembles.poses_df = ensembles.poses_df.merge(pose_opts_df, left_on="poses_description", right_on="1description")
    ensembles.poses_df = ensembles.poses_df.merge(fixedres_df, left_on="poses_description", right_on="2description")
    ensembles.poses_df = ensembles.poses_df.merge(motif_res_df, left_on="poses_description", right_on="3description").drop(columns=["1description", "2description", "3description", "rdescription"])
    if len(ensembles.poses_df) == len(ensembles.poses): print(f"Loading of Pose contigs into poses_df successful. Continuing to inpainting.")
    else: raise ValueError(f"Merging of inpaint_opts into poses_df failed! Check if keys in inpaint_opts match with pose_names!!!")
    ensembles.poses_df["template_motif"] = ensembles.poses_df["motif_residues"]
    ensembles.poses_df["template_fixedres"] = ensembles.poses_df["fixed_residues"]

    # Inpaint, relax and calc pLDDT
    inpaints = ensembles.inpaint(options=f"--n_cycle 15 --num_designs {args.num_inpaints}", pose_options=list(ensembles.poses_df["inpainting_pose_opts"]), prefix="inpainting", perres_lddt=True, perres_inpaint_lddt=True)
    #inpaints = ensembles.inpaint(options=f"--n_cycle 15 --num_designs 1", pose_options=list(ensembles.poses_df["inpainting_pose_opts"]), prefix="inpainting", perres_lddt=True, perres_inpaint_lddt=True)

    # Update motif_res and fixedres to residue mapping after inpainting
    _ = [ensembles.update_motif_res_mapping(motif_col=col, inpaint_prefix="inpainting") for col in motif_cols]
    _ = ensembles.update_res_identities(identity_col="catres_identities", inpaint_prefix="inpainting")
    print(ensembles.poses_df["catres_identities"].to_list())

    # Filter down (first, to one inpaint per backbone, then by half) based on pLDDT and RMSD
    inpaint_template_rmsd = ensembles.calc_motif_bb_rmsd_dir(ref_pdb_dir=pdb_dir, ref_motif=list(ensembles.poses_df["template_motif"]), target_motif=list(ensembles.poses_df["motif_residues"]), metric_prefix="inpaint_template_bb_ca", remove_layers=1)
    inpaint_comp_score = ensembles.calc_composite_score("inpaint_comp_score", ["inpainting_lddt", "inpaint_template_bb_ca_motif_rmsd"], [-1, args.inpaint_rmsd_weight])
    inpaint_sampling_filter = ensembles.filter_poses_by_score(args.num_mpnn_inputs, "inpaint_comp_score", prefix="inpaint_sampling_filter", remove_layers=1)
    #inpaint_filter = ensembles.filter_poses_by_score(100, "inpaint_comp_score", prefix="inpaint_filter")
    
    # mutate any residues in the pose back to what they are supposed to be (inpainting sometimes does not keep the sequence)
    _ = ensembles.biopython_mutate("catres_identities")

    # Run MPNN and filter (by half)
    mpnn_designs = ensembles.mpnn_design(mpnn_options=f"--num_seq_per_target={args.num_mpnn_seqs} --sampling_temp=0.1", prefix="mpnn", fixed_positions_col="fixed_residues")
    mpnn_seqfilter = ensembles.filter_poses_by_score(args.num_esm_inputs, "mpnn_score", prefix="mpnn_seqfilter", remove_layers=1)

    # Run ESMFold and calc bb_ca_rmsd, motif_ca_rmsd and motif_heavy RMSD
    esm_preds = ensembles.predict_sequences(run_ESMFold, prefix="esm")
    esm_bb_ca_rmsds = ensembles.calc_bb_rmsd_dir(ref_pdb_dir=inpaints, metric_prefix="esm", ref_chains=["A"], pose_chains=["A"], remove_layers=1)
    esm_motif_rmsds = ensembles.calc_motif_bb_rmsd_dir(ref_pdb_dir=pdb_dir, ref_motif=list(ensembles.poses_df["template_motif"]), target_motif=list(ensembles.poses_df["motif_residues"]), metric_prefix="esm_bb_ca", remove_layers=2)
    esm_motif_heavy_rmsds = ensembles.calc_motif_heavy_rmsd_dir(ref_pdb_dir=pdb_dir, ref_motif=ensembles.poses_df["template_fixedres"].to_list(), target_motif=ensembles.poses_df["fixed_residues"].to_list(), metric_prefix="esm_catres", remove_layers=2)

    # Filter Redesigns based on confidence and RMSDs
    esm_comp_score = ensembles.calc_composite_score("esm_comp_score", ["esm_plddt", "esm_catres_motif_heavy_rmsd"], [-1, 1])
    esm_filter = ensembles.filter_poses_by_score(1, "esm_comp_score", remove_layers=1, prefix="esm_filter")
    
    # Plot Results
    if not os.path.isdir((plotdir := f"{ensembles.dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # Inpainting stats:
    cols = ["inpainting_lddt", "inpainting_inpaint_lddt", "inpaint_template_bb_ca_motif_rmsd", "inpainting_trf_motif_bb_ca_rmsd"]
    titles = ["Full Inpainting\npLDDT", "Inpaint-ONLY\npLDDT", "TRF-Inpaint\nMotif RMSD", "TRF-Template\nMotif RMSD"]
    y_labels = ["pLDDT", "pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,1), (0,1), (0,2), (0,2)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/inpainting_stats.png")

    # ESM stats:
    cols = ["mpnn_score", "esm_plddt", "esm_bb_ca_rmsd", "esm_bb_ca_motif_rmsd"]
    titles = ["MPNN score", "ESM pLDDT", "ESM BB-Ca RMSD", "ESM Motif-Ca RMSD"]
    y_labels = ["-log(prob)", "pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,2), (0,100), (0,15), (0,8)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/esm_stats.png")
    # Store filtered poses away:
    ensembles.dump_poses(f"{args.output_dir}/final_pdbs/")



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be inpainted (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    # inpainting options
    argparser.add_argument("--num_inpaints", type=int, default=10, help="Number of inpaints for each input fragment for subsampling.")
    argparser.add_argument("--num_inpaint_cycles", type=int, default=15, help="Number of inpainting recycles to run for each inpainting.")
    argparser.add_argument("--translation_sampling_magnitude", type=float, default=0.1, help="Magnitude of random translation of fragments for randomized sampling during inpainting.")
    argparser.add_argument("--rotation_sampling_degrees", type=float, default=1, help="Degrees. How much to rotate fragments during inpainting.")
    argparser.add_argument("--inpaint_rmsd_weight", type=float, default=3.0, help="Weight of inpainting RMSD score for filtering sampled inpaints.")
    argparser.add_argument("--max_inpaint_gpus", type=int, default=10, help="On how many GPUs at a time to you want to run inpainting?")

    # mpnn options
    argparser.add_argument("--num_mpnn_inputs", type=int, default=1, help="Number of inpaints for each input fragment that should be passed to MPNN.")
    argparser.add_argument("--num_mpnn_seqs", type=int, default=20, help="Number of MPNN Sequences to generate for each input backbone.")
    argparser.add_argument("--num_esm_inputs", type=int, default=10, help="Number of MPNN Sequences for each input backbone that should be predicted. Typically quarter to half of the sequences generated by MPNN is a good value.")
    args = argparser.parse_args()

    main(args)
