#!/home/mabr3112/anaconda3/bin/python3.9
import sys
sys.path.append("/home/mabr3112/riff_diff")
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]

import json
from iterative_refinement import *
from glob import glob
import utils.plotting as plots
import utils.biopython_tools
import utils.pymol_tools

def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, motif_prefix: str, out_pdb_path=None) -> list[str]:
    ''''''
    list_of_mappings = [utils.biopython_tools.residue_mapping_from_motif(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{motif_prefix}_inpainting_con_ref_pdb_idx"].to_list(), input_df["{motif_prefix}_inpainting_con_hal_pdb_idx"].to_list())]
    output_pdb_names_list = [f"{out_pdb_path}/{desc}" for desc in input_df[desc_col].to_list()]

    list_of_output_paths = [utils.biopython_tools.renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

    return list_of_output_paths

def parse_outfilter_args(scoreterm_str: str, weights_str: str, df: pd.DataFrame) -> tuple[list]:
    ''''''
    def check_for_col_in_df(col: str, datf: pd.DataFrame) -> None:
        if col not in datf.columns: raise KeyError("Scoreterm {col} not found in poses_df. Available scoreterms: {','.join(datf.columns)}")
    scoreterms = scoreterm_str.split(",")
    weights = [float(x) for x in weights_str.split(",")]
    check = [check_for_col_in_df(scoreterm, df) for scoreterm in scoreterms]

    if not len(scoreterms) == len(weights): raise ValueError(f"Length of --output_scoreterms ({scoreterm_str}: {len(scoreterm_str)}) and --output_scoreterm_weights ({weights_str}: {len(weights_str)}) is not the same. Both arguments must be of the same length!")

    return scoreterms, weights

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
    inpaint_sampling_filter = ensembles.filter_poses_by_score(args.num_mpnn_inputs, "inpaint_comp_score", prefix="inpaint_sampling_filter", remove_layers=1, plot=["inpaint_comp_score", "inpainting_lddt", "inpainting_inpaint_lddt", "inpaint_template_bb_ca_motif_rmsd", "inpainting_trf_motif_bb_ca_rmsd"])
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
    esm_comp_score = ensembles.calc_composite_score("esm_comp_score", ["esm_plddt", "esm_bb_ca_motif_rmsd"], [-1, 1])
    esm_filter = ensembles.filter_poses_by_score(1, "esm_comp_score", remove_layers=1, prefix="esm_filter", plot=["esm_comp_score", "esm_plddt", "esm_bb_ca_rmsd", "esm_bb_ca_motif_rmsd", "esm_catres_motif_heavy_rmsd"])
    
    # Plot Results
    if not os.path.isdir((plotdir := f"{ensembles.dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # Inpainting stats:
    cols = ["inpainting_lddt", "inpainting_inpaint_lddt", "inpaint_template_bb_ca_motif_rmsd", "inpainting_trf_motif_bb_ca_rmsd", "mpnn_score"]
    titles = ["Full Inpainting\npLDDT", "Inpaint-ONLY\npLDDT", "TRF-Inpaint\nMotif RMSD", "TRF-Template\nMotif RMSD", "MPNN score"]
    y_labels = ["pLDDT", "pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "-log(prob)"]
    dims = [(0,1), (0,1), (0,2), (0,2), (0,2)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/inpainting_stats.png")

    # ESM stats:
    cols = ["esm_plddt", "esm_bb_ca_rmsd", "esm_bb_ca_motif_rmsd", "esm_catres_motif_heavy_rmsd"]
    titles = ["ESM pLDDT", "ESM BB-Ca RMSD", "ESM Motif-Ca RMSD", "ESM Catres\nSidechain RMSD"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,100), (0,15), (0,8), (0,8)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/esm_stats.png")
    # Store filtered poses and scores away:
    ensembles.dump_poses(f"{args.output_dir}/esm_output_pdbs/")

    # Filter down to final set of .pdbs that will be input for Rosetta Refinement:
    scoreterms, weights = parse_outfilter_args(args.output_scoreterms, args.output_scoreterm_weights, ensembles.poses_df)
    out_filterscore = ensembles.calc_composite_score("out_filter_comp_score", scoreterms, weights)
    out_filter = ensembles.filter_poses_by_score(args.num_outputs, f"out_filter_comp_score", prefix="out_filter", plot=scoreterms)
    results_dir = f"{args.output_dir}/results/"
    ref_frag_dir = f"{results_dir}/ref_fragments/"
    ensembles.dump_poses(results_dir)

    # Copy and rewrite Fragments into output_dir/reference_fragments
    updated_ref_pdbs = update_and_copy_reference_frags(ensembles.poses_df, ref_col="input_poses", desc_col="poses_description", motif_prefix="inpainting_", out_pdb_path=ref_frag_dir)

    # Write PyMol Alignment Script
    ref_originals = [shutil.copyfile(ref_pose, f"{results_dir}/") for ref_pose in ensembles.poses_df["input_poses"].to_list()]
    pymol_script = utils.pymol_tools.write_pymol_alignment_script(ensembles.poses_df, scoreterm="out_filter_comp_score", top_n=args.num_outputs, path_to_script=f"{results_dir}/align.pml")

    # Plot final stats of selected poses
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{results_dir}/final_esm_stats.png")

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

    # output options
    argparser.add_argument("--num_outputs", type=int, default=25, help="Number of .pdb files that will be stored into the final output directory.")
    argparser.add_argument("--output_scoreterms", type=str, default="esm_plddt,esm_bb_ca_motif_rmsd", help="Scoreterms to use to filter ESMFolded PDBs to the final output pdbs. IMPORTANT: if you supply scoreterms, also supply weights and always check the filter output plots in the plots/ directory!")
    argparser.add_argument("--output_scoreterm_weights", type=str, default="-1,1", help="Weights for how to combine the scoreterms listed in '--output_scoreterms'")
    

    args = argparser.parse_args()

    main(args)
