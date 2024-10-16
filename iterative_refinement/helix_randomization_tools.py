import torch
import numpy as np
import pandas as pd
import os
from glob import glob

import Bio
import Bio.PDB
from copy import deepcopy
from Bio.PDB import Atom

from scipy.spatial.transform import Rotation as R
from scipy.spatial import procrustes

from scipy.linalg import svd

from Bio import SVDSuperimposer

def get_rotran(mobile, ref, normalize_vec=False, return_non_rotated_vector=False, as_quaternion=False):
    '''
    Returns rotation and translation to put points in <mobile> on <ref>
    '''
    def normalize_vector(input_vector):
        mag = np.linalg.norm(input_vector)
        norm = input_vector / mag
        return np.append(norm, np.array(mag))
    
    # setup coords
    super_imposer = SVDSuperimposer.SVDSuperimposer()
    super_imposer.set(reference_coords=ref, coords=mobile)
    super_imposer.run()
    
    # run superimposer
    rot, tran = super_imposer.get_rotran()
    
    # transform if options are specified
    if return_non_rotated_vector: tran = np.array(np.mean(mobile, axis=0) - np.mean(ref, axis=0))
    translation = normalize_vector(tran) if normalize_vec else tran
    rotation = R.from_matrix(np.dot(U, Vt)).as_quat() if as_quaternion else rot

    return rotation, translation

def randomize_triangle(triangle: np.array, translation_scale:float=1) -> np.array:
    """
    Randomly rotate and translate a triangle.

    Parameters:
        triangle (numpy.ndarray): A 3x3 matrix where each row represents a vertex of the triangle.

    Returns:
        numpy.ndarray: A 3x3 matrix representing the randomized triangle.
    """
    # Create a random translation vector
    translation = np.random.rand(3) * translation_scale

    # Create a random rotation quaternion
    r = R.random()

    # Apply the transformations to the triangle
    randomized_triangle = (triangle + translation) @ r.as_matrix()

    return randomized_triangle


def create_atom_at_coords(coords: list[float], atom_type="CA") -> Bio.PDB.Atom:
    return Atom.Atom(name=atom_type, coord=coords, bfactor=0.0, occupancy=1.0, altloc=" ", fullname=atom_type, serial_number=1, element="C")


def compare_triangle_shape(triangle1, triangle2):
    """
    Compare the shape of two triangles.

    Parameters:
        triangle1 (numpy.ndarray): A 3x3 matrix where each row represents a vertex of the first triangle.
        triangle2 (numpy.ndarray): A 3x3 matrix where each row represents a vertex of the second triangle.

    Returns:
        bool: True if the two triangles have the same shape, False otherwise.
    """
    # Compute the Procrustes similarity between the two triangles
    _, _, similarity = procrustes(triangle1, triangle2)

    # Compare the similarity to a threshold
    threshold = 0.1
    if similarity < threshold:
        return True
    else:
        return False
    
def random_perturb(pose: Bio.PDB.Structure, save_path: str, chain: str, resi: int, atms:list[str]=["N", "CA", "O"], rand_chain:str="B", translation_scale:float=15) -> None:
    '''
    '''
    # make a copy of the chain
    pose = deepcopy(pose)
    chain_B = deepcopy(pose[chain])
    chain_B.detach_parent()
    chain_B.id = "B"

    # read specified atoms of specified residue and get coordinates.
    bb_triangle_atms = [pose[chain][(" ", resi, " ")][atom] for atom in atms]
    bb_triangle_coords = np.array([atom.coord for atom in bb_triangle_atms])

    # apply random rotation and translation to the coordinates.
    rand_triangle_coords = randomize_triangle(bb_triangle, translation_scale=translation_scale)

    # create atoms at the coordinates
    rand_triangle_atms = [create_atom_at_coords(coords) for coords in rand_triangle_coords]

    # set atoms N, Ca and O and the transformed coordinates as atoms for the Bio.PDB superimposer
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(rand_triangle_atms, bb_triangle_atms)

    # apply the superimposer to chain_B
    super_imposer.apply(chain_B)

    # add the chain to the pose
    pose.add(chain_B)

    # save the pose containing the original chain and the superimposed chain.
    io = PDBIO()
    io.set_structure(pose)
    io.save(save_path)
    
    return None

def read_trans_rot_from_pdb(path_to_pdb: str, res_a: str, res_b: str, atoms:list[str]=["N", "CA", "O"], return_non_rotated_vector=False) -> pd.DataFrame:
    '''
    res_a, res_b: pdb_numbering!
    '''
    # sanity
    if not os.path.isfile(path_to_pdb): raise FileNotFoundError(f"{path_to_pdb} is not a file!")
    if not path_to_pdb.endswith(".pdb"): print(f"WARNING: Input File {path_to_pdb} does not have .pdb extension. Make sure it is in a dataformat that is readable by BioPython!")
    
    # start the parser and read the structure
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    description = path_to_pdb.split("/")[-1].replace(".pdb", "")
    pose = pdb_parser.get_structure("pose", path_to_pdb)[0]
    
    # collect atoms from residues
    atms_a = [pose[res_a[0]][(" ", int(res_a[1:]), " ")][atom] for atom in atoms]
    atms_b = [pose[res_b[0]][(" ", int(res_b[1:]), " ")][atom] for atom in atoms]
    
    # convert atoms to coords (triangles)
    a = np.array([atom.coord for atom in atms_a])
    b = np.array([atom.coord for atom in atms_b])
    
    # calculate translation and rotation
    rot, trans = get_rotran(a, b, normalize_vec=True, return_non_rotated_vector=return_non_rotated_vector)
    quat = R.from_matrix(rot).as_quat()
    
    # put into DataFrame and return
    return pd.DataFrame({"description": [description], "translation": [trans], "quaternion": [quat]})

def collect_trans_rot_from_dir(input_dir: str, res_a: str, res_b: str, return_non_rotated_vector=False) -> pd.DataFrame:
    '''
    res_a, res_b: pdb_numbering!
    '''
    # sanity
    if not os.path.isdir(input_dir): raise FileNotFoundError(f"{input_dir} is not a directory!")
    
    # read all pdb files into list
    fl = glob(f"{input_dir}/*.pdb")
    
    # collect translation and rotation DataFrames from pdb-files
    dat = [read_trans_rot_from_pdb(pdb, res_a, res_b, return_non_rotated_vector=return_non_rotated_vector) for pdb in fl]
    
    # convert translation and rotations into DataFrame and return:
    return pd.concat(dat).reset_index(drop=True)
