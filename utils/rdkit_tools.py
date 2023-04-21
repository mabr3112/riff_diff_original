from rdkit import Chem

def convert_pdb_to_mol(pdb_path: str, out_path:str=None, removeHs:bool=False) -> str:
    '''Converts .pdb file of a ligand into a .mol file using RDkit'''
    out_path = out_path or pdb_path.replace(".pdb", ".sdf")
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=removeHs)
    sdf = Chem.MolToMolBlock(mol, kekulize=True)
    with open(out_path, 'w') as f:
        f.write(sdf)

    return out_path 