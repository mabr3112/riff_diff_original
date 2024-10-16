#!/home/mabr3112/anaconda3/bin/python3.9

from glob import glob
from iterative_refinement import *

# initialize poses from "test_inputs" as object
poses = Poses("mo_binder_refine", glob("mo_binder_inputs/*.pdb"))

# create 20 relax decoys and filter them down to 10 by total_score
#relax_run1 = poses.relax_poses(relax_options="-ex1 -ex2 -ignore_unrecognized_res", n=25, prefix="initial_relax")
relax_decoys = poses.create_relax_decoys(relax_options="-ex1 -ex2 -ignore_unrecognized_res", n=20, prefix="relax_decoy_run1")
filtered_poses = poses.filter_poses_by_score(10, score_col="relax_decoy_run1_total_score")

# now create sequences for the relaxed backbones with ProteinMPNN. Set up ProteinMPNN options first.
mpnn_options = {"num_seq_per_target": "100", "sampling_temp": "0.2"}
mpnn_run1 = poses.mpnn_design(mpnn_options)

# filter mpnn structures down again to top 25 by mpnn score
mpnn_run1_filtered_poses = poses.filter_poses_by_score(10, score_col="mpnn_run_0001_score")

# Now run OmegaFold to predict the structures of the sequences, afterwards relax the poses and sort them by total_score and confidence
of_predictions = poses.predict_sequences(run_OmegaFold)
relaxed_predictions = poses.relax_poses(relax_options="-ex1 -ex2", n=25, prefix="postpredict_relax")

print(f"Finished")
