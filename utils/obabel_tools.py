from openbabel import openbabel

def obabel_fileconverter(input_file: str, output_file:str=None, input_format:str="pdb", output_format:str="mol2") -> None:
    '''converts pdbfile to mol2-file.'''
    # Create an Open Babel molecule object
    mol = openbabel.OBMol()

    # Read the PDB file
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(input_format)
    obConversion.ReadFile(mol, input_file)

    # Convert the molecule to the desired output format (Mol2 file)
    obConversion.SetOutFormat(output_format)
    obConversion.WriteFile(mol, output_file or input_file)
    return output_file or input_file