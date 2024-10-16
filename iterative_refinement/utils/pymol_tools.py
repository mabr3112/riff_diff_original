#############################
#
#   Tools to work with poses using PyMol
#
#
#############################

import pandas as pd

def pymol_motif_color_scriptwriter(df: pd.DataFrame, path_to_script: str, motif_col: str, description_col:str="poses_description", color_motif:float=[1, 0.8, 0], color_bg:float=[0.5,0.5,0.5]) -> str:
    '''AAA'''
    cmds = [write_motif_color_command(df.loc[i, description_col], motif=df.loc[i, motif_col], color_motif=color_motif, color_bg=color_bg) for i in df.index]
    with open(path_to_script, 'w', encoding="UTF-8") as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_motif_color_command(description: str, motif:dict, color_motif:tuple[float]=(1,0.8,0), color_bg=(0.5,0.5,0.5)) -> str:
    '''writes a command that colors the protein according to a motif in specified colors for 'color_motif' and 'color_bg'. '''
    def collapse(in_dict: dict) -> list[str]:
        return [f"(resi {'+'.join([str(x) for x in in_dict[key]])} and chain {key})" for key in in_dict]

    # load protein
    color_cmds = [f"load {description}.pdb, {description}"]
    
    # define selections motif and not_motif
    color_cmds.append(f"select motif, ({' or '.join(collapse(motif))}) and {description}")
    color_cmds.append(f"select background, not motif and {description}")

    # define colors
    color_cmds.append(f"set_color motif_color_{description}, {list(color_motif)}")
    color_cmds.append(f"set_color background_color_{description}, {list(color_bg)}")

    # color
    color_cmds.append(f"color motif_color_{description}, motif")
    color_cmds.append(f"color background_color_{description}, background")
    color_cmds.append("center motif")

    # store the scene
    color_cmds.append(f"scene {description}, store")
    
    # clean up the mess
    color_cmds.append(f"delete motif")
    color_cmds.append(f"delete background")
    color_cmds.append(f"disable {description}")

    return "\n".join(color_cmds)

def write_pymol_alignment_script(df:pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, use_original_location=False) -> str:
    '''
    '''
    cmds = [write_align_cmds(df.loc[index], use_original_location=use_original_location) for index in df.sort_values(scoreterm, ascending=ascending).head(top_n).index]
    
    with open(path_to_script, 'w') as f:
        f.write("\n".join(cmds))
    return path_to_script

def pymol_alignment_scriptwriter(df: pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, pose_col="poses_description", ref_pose_col="input_poses", motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres"):
    ''''''
    top_df = df.sort_values(scoreterm, ascending=ascending).head(top_n)
    cmds = [write_align_cmds_v2(top_df.loc[index], pose_col=pose_col, ref_pose_col=ref_pose_col, motif_res_col=motif_res_col, fixed_res_col=fixed_res_col, ref_motif_res_col=ref_motif_res_col, ref_fixed_res_col=ref_fixed_res_col) for index in top_df.index]

    with open(path_to_script, 'w') as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_pymol_motif_selection(obj: str, motif: dict) -> str:
    '''AAA'''
    residues = [f"chain {chain} and resi {'+'.join([str(x) for x in res_ids])}" for chain, res_ids in motif.items()]
    pymol_selection = ' or '.join([f"{obj} and {resis}" for resis in residues])
    return pymol_selection

def write_align_cmds(input_data: pd.Series, use_original_location=False):
    '''AAA'''
    cmds = list()
    if use_original_location: 
        ref_pose = input_data["input_poses"].replace(".pdb", "")
        pose = input_data["esm_location"]
    else: 
        ref_pose = input_data["input_poses"].split("/")[-1].replace(".pdb", "")
        pose = input_data["poses_description"] + ".pdb"

    # load pose and reference
    cmds.append(f"load {pose}")
    ref_pose_name = input_data['poses_description'] + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {input_data['poses_description']}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data['poses_description'], input_data['motif_residues'])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data['template_motif'])}")

    # superimpose inpaint_motif_residues:
    cmds.append(f"cealign temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data['poses_description'], input_data['fixed_residues'])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data['template_fixedres'])}")
    cmds.append(f"show sticks, temp_cat_res")
    cmds.append(f"show sticks, temp_refcat_res")
    cmds.append(f"hide sticks, hydrogens")
    cmds.append(f"color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data['poses_description']}, store")
    cmds.append(f"disable {input_data['poses_description']}")
    cmds.append(f"disable {ref_pose_name}")
    cmds.append(f"delete temp_cat_res")
    cmds.append(f"delete temp_refcat_res")
    cmds.append(f"delete temp_motif_res")
    cmds.append(f"delete temp_ref_res")
    return "\n".join(cmds)

def write_align_cmds_v2(input_data: pd.Series, pose_col="poses_description", ref_pose_col="input_poses", motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres"):
    '''AAA'''
    cmds = list()
    ref_pose = input_data[ref_pose_col].split("/")[-1].replace(".pdb", "")
    pose_desc = input_data[pose_col]
    pose = pose_desc + ".pdb"

    # load pose and reference
    cmds.append(f"load {pose}, {pose_desc}")
    ref_pose_name = pose_desc + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {pose_desc}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data[pose_col], input_data[motif_res_col])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_motif_res_col])}")

    # superimpose inpaint_motif_residues:
    cmds.append(f"cealign temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data[pose_col], input_data[fixed_res_col])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_fixed_res_col])}")
    cmds.append(f"show sticks, temp_cat_res")
    cmds.append(f"show sticks, temp_refcat_res")
    cmds.append(f"hide sticks, hydrogens")
    cmds.append(f"color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data[pose_col]}, store")
    cmds.append(f"disable {input_data[pose_col]}")
    cmds.append(f"disable {ref_pose_name}")
    cmds.append(f"delete temp_cat_res")
    cmds.append(f"delete temp_refcat_res")
    cmds.append(f"delete temp_motif_res")
    cmds.append(f"delete temp_ref_res")
    return "\n".join(cmds)
