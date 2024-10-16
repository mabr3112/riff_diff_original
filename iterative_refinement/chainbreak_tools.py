import Bio
from Bio import PDB
import numpy as np
import re
import os

def calc_atm_dist(structure: Bio.PDB.Structure, res_a: int, res_b: int, atom_a:str="CA", atom_b:str="CA") -> float:
    '''
    res_a, res_b must be in PDB-numbering
    '''
    def get_coords(struct, resi, atom):
        if type(resi) != str: raise TypeError(f"input {resi} of wrong type. Must be PDB numbering!")
        return struct[resi[0]][(" ", int(resi[1:]), " ")][atom].coord
    
    coord_res_a = get_coords(structure, res_a, "CA")
    coord_res_b = get_coords(structure, res_b, "CA")
    
    vec = coord_res_a - coord_res_b
    dist = np.linalg.norm(vec)
    
    return dist

def get_pdb_numbering_of_struct(structure: Bio.PDB.Structure):
    '''
    '''
    return [f"{resi.parent.id}{resi.id[1]}" for resi in structure.get_residues()]

def parse_pdb_contig(contig: str) -> list:
    '''
    This function parses a contig string of PDB format residues. 
    It converts residue spans (e.g. A1-10) to a list of individual residues (e.g. A1, A2, A3 ... A10)

    Parameters:
    contig (str): The contig string to be parsed. It should contain residues separated by commas, 
                  and residue spans can be represented by a chain identifier followed by a range of residue ids 
                  (e.g. A1-10)

    Returns:
    list: A list of individual residues in PDB format

    Example:
    parse_pdb_contig("A1-10,B5,C8-12")
    Output: ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'B5', 'C8', 'C9', 'C10', 'C11', 'C12']
    '''
    def convert_resi_span_to_list(span):
        res_ids = [int(x) for x in span[1:].split("-")]
        chain = span[0]
        for resi in range(res_ids[0], res_ids[1]+1):
            yield f"{chain}{resi}"
        
    residues = list()
    elements = contig.split(",")
    
    # expand residue span (A1-10) if it is present (A1, A2, A3 ... A10)
    for element in elements:
        if "-" in element:
            [residues.append(x) for x in convert_resi_span_to_list(element)]
        else:
            residues.append(element)

    return residues

def search_for_chainbreak(structure: Bio.PDB.Structure, contig: str, deviation:float=0.2) -> bool:
    '''
    Example for region: 
        contig="A1-10,A12,A13"
    '''
    def get_next_resi(resi):
        return resi[0] + str(int(resi[1:])+1)
    
    # parse all residues from the contig into a list
    residues = parse_pdb_contig(contig)
    
    # calculate "CA-CA" distance of every residue in list(residues) to its downstream residue
    distances = [calc_atm_dist(structure, resi, get_next_resi(resi)) for resi in residues]
    
    # check if all of the calculated distances are inside of the accepted (set by deviation) CA-CA range:
    found_chainbreak = not all([3.8-deviation < distance < 3.8+deviation for distance in distances])
    
    return 0 if found_chainbreak else 1

def search_chainbreak_in_pdb(path_to_pdb: str, contig: str, deviation:float=0.2) -> int:
    '''
    
    '''
    # sanity
    if not os.path.isfile(path_to_pdb): raise FileNotFoundError(path_to_pdb)
    if not path_to_pdb.endswith(".pdb"): print(f"WARNING: Input File {path_to_pdb} does not have .pdb extension. Make sure it is of .pdb Format!")
    
    # load Bio.PDB.Structure from .pdb
    parser = Bio.PDB.PDBParser(QUIET=True)
    pose = parser.get_structure("tt", path_to_pdb)[0]
    
    # check chainbreak:
    return search_for_chainbreak(pose, contig, deviation)

def get_linker_contig(path_to_trb: str, chain:str="A"):
    '''
    Extracts the linker residues from a .trb file and returns a contig of the linker residues.

    Parameters:
        path_to_trb (str): The path to the .trb file.
        chain (str): The Chain identifier of the residues (default A)

    Returns:
        str: A contig of the linker residues in the format 'A11,A12,A13,A14'
    '''
    # sanity
    if not os.path.isfile(path_to_trb): raise FileNotFoundError(path_to_trb)
    if not path_to_trb.endswith(".trb"): print("WARNING: Input to get_linker_contig {path_to_trb} does not have .trb extension. Are you sure it is a .trb formatted file?")
    
    # load PDB res_IDs of template residues:
    trb = np.load(path_to_trb, allow_pickle=True)
    resis = trb["con_hal_pdb_idx"]
    ids = [i[1] for i in resis]
    
    # check for missing res_IDs within range of PDB res_IDs (these are the linker res_IDs!)
    linker_res = list()
    for i in range(ids[0], ids[-1] + 1):
        if i not in ids:
            linker_res.append(f"{chain}{i}")
    
    # return contig
    return ",".join(linker_res)

