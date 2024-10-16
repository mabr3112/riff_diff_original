# module for composite protocols using iterative_refinement.
import sys
sys.path.append("../")

# dependencies
import pandas as pd

# custom modules
import iterative_refinement

def calculate_fastrelax_sidechain_and_motif_rmsd(poses: iterative_refinement.Poses, prefix: str, options: str, pose_options: str, sidechain_residues: str, motif_residues: str, sidechain_ref_pdb_col: str, motif_ref_pdb_col: str, n:int=5):
    '''runs fastrelax on poses 'n' times and then calculates average sidechain RMSD to the motif specified in sidechain_residues and motif_ca RMSD to the residues specified motif_residues.'''
    # run fastrelax
    poses.poses_df[f"{prefix}_poses_description_copy"] = poses.poses_df["poses_description"] # for merging later
    fr = poses.rosetta("rosetta_scripts.default.linuxgccrelease", options=options, pose_options=pose_options, prefix=f"{prefix}_fr")

    # calculate rmsd to motif
    sidechain_rmsd = poses.calc_motif_heavy_rmsd_df(ref_pdb=sidechain_ref_pdb_col, ref_motif=sidechain_residues, target_motif=sidechain_residues, metric_prefix=f"{prefix}_sidechain")
    motif_rmsd = poses.calc_motif_bb_rmsd_df(ref_pdb=motif_ref_pdb_col, ref_motif=motif_residues, target_motif=motif_residues, metric_prefix=f"{prefix}_bb_ca")

    # calculate mean rmsds and add them to poses_df
    mean_sidechain_rmsds = poses.poses_df.groupby(f"{prefix}_poses_description_copy")[f"{prefix}_sidechain_motif_heavy_rmsd"].mean()
    poses.poses_df[f"{prefix}_mean_sidechain_motif_heavy_rmsd"] = poses.poses_df[f"{prefix}_poses_description_copy"].map(mean_sidechain_rmsds)

    mean_motif_rmsds = poses.poses_df.groupby(f"{prefix}_poses_description_copy")[f"{prefix}_bb_ca_motif_rmsd"].mean()
    poses.poses_df[f"{prefix}_mean_bb_ca_motif_rmsd"] = poses.poses_df[f"{prefix}_poses_description_copy"].map(mean_motif_rmsds)

    # filter to lowest rmsd pose
    filter_comp_score = poses.calc_composite_score(f"{prefix}_composite_score", [f"{prefix}_fr_total_score", f"{prefix}_bb_ca_motif_rmsd", f"{prefix}_sidechain_motif_heavy_rmsd"], [0.5, 0.1, 1])
    filtered = poses.filter_poses_by_score(1, f"{prefix}_composite_score", remove_layers=1, prefix=f"{prefix}_filter", plot=[f"{prefix}_fr_total_score", f"{prefix}_bb_ca_motif_rmsd", f"{prefix}_sidechain_motif_heavy_rmsd", f"{prefix}_mean_sidechain_motif_heavy_rmsd", f"{prefix}_mean_bb_ca_motif_rmsd"])
    return poses

def calculate_fastrelax_sidechain_rmsd(poses: iterative_refinement.Poses, prefix: str, options: str, sidechain_residues: str, sidechain_ref_pdb_col: str, n:int=5, pose_options:str=None):
    '''runs fastrelax on poses 'n' times and then calculates average sidechain RMSD to the motif specified in sidechain_residues.'''
    # run fastrelax
    poses.poses_df[f"{prefix}_poses_description_copy"] = poses.poses_df["poses_description"] # for merging later
    fr = poses.rosetta("rosetta_scripts.default.linuxgccrelease", options=options, n=n, pose_options=pose_options, prefix=f"{prefix}_fr")

    # calculate rmsd to motif
    sidechain_rmsd = poses.calc_motif_heavy_rmsd_df(ref_pdb=sidechain_ref_pdb_col, ref_motif=sidechain_residues, target_motif=sidechain_residues, metric_prefix=f"{prefix}_sidechain")

    # calculate mean rmsds and add them to poses_df
    mean_sidechain_rmsds = poses.poses_df.groupby(f"{prefix}_poses_description_copy")[f"{prefix}_sidechain_motif_heavy_rmsd"].mean()
    poses.poses_df[f"{prefix}_mean_sidechain_motif_heavy_rmsd"] = poses.poses_df[f"{prefix}_poses_description_copy"].map(mean_sidechain_rmsds)

    # filter to lowest rmsd pose
    filter_comp_score = poses.calc_composite_score(f"{prefix}_composite_score", [f"{prefix}_fr_total_score", f"{prefix}_sidechain_motif_heavy_rmsd"], [0.5, 1])
    filtered = poses.filter_poses_by_score(1, f"{prefix}_composite_score", remove_layers=1, prefix=f"{prefix}_filter", plot=[f"{prefix}_fr_total_score", f"{prefix}_sidechain_motif_heavy_rmsd", f"{prefix}_mean_sidechain_motif_heavy_rmsd"])
    return poses

def rosetta_scripts_and_mean(poses: iterative_refinement.Poses, prefix: str, n: int, options: str, pose_options: str, filter_scoreterm: str, scoreterms:list[str]=None, std_scoreterms:list[str]=False, min_scoreterms:list[str]=False, filter_ascending=True):
    '''
    Runs RosettaScripts (has to be provided in <options>) -nstruct times over poses, then calculates statistics over <scoreterms> and <filter_scoreterm> and finally filters down to <filter_scoreterm>.
    '''
    def map_to_df(orig_df, map_df, map_col) -> None:
        for col in map_df.columns:
            print(col)
            orig_df[col] = orig_df[map_col].map(map_df[col])
        return orig_df

    # sanity
    if type(scoreterms) == str: scoreterms = [scoreterms]

    def check_for_col(df, col):
        if col not in df.columns: raise KeyError(f"Scoreterm {col} not found in poses_df. Available Scoreterms: {', '.join(df.columns.to_list())}")

    # fix old description
    poses.poses_df.loc[:, "tmp_desc"] = poses.poses_df["poses_description"]

    # run Rosetta
    ros_pos = poses.rosetta("rosetta_scripts.default.linuxgccrelease", n=n, options=options, pose_options=pose_options, prefix=prefix)

    # check if scoreterms exist now in poses_df:
    scoreterms = list(set([filter_scoreterm] + scoreterms)) if scoreterms else [filter_scoreterm] # make sure every scoreterm is in the scoreterms list once.
    [check_for_col(poses.poses_df, scoreterm) for scoreterm in scoreterms]

    # calculate statistics over specified scoreterms:
    means = poses.poses_df.groupby("tmp_desc")[scoreterms].mean().add_prefix("mean_")
    poses.poses_df = map_to_df(poses.poses_df, means, "tmp_desc")

    if std_scoreterms:
        stds = poses.poses_df.groupby("tmp_desc")[scoreterms].std().add_prefix("std_")
        poses.poses_df = map_to_df(poses.poses_df, stds, "tmp_desc")

    if min_scoreterms:
        mins = poses.poses_df.groupby("tmp_desc")[scoreterms].min().add_prefix("min_")
        poses.poses_df = map_to_df(poses.poses_df, mins, "tmp_desc")

    # filter down to best pose by <filter_scoreterm>
    poses.filter_poses_by_score(1, f"{filter_scoreterm}", remove_layers=1, prefix=f"{prefix}_filter")

    return poses
