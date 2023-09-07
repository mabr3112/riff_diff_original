from openbabel import openbabel

def obabel_fileconverter(input_file: str, output_file:str=None, input_format:str="pdb", output_format:str="mol2") -> None:
    '''
    Converts a PDB file to a Mol2 file using the Open Babel library

    Parameters:
    - input_file (str): The path of the input file to be converted.
    - output_file (str, optional): The path of the output file. If not provided, the output file will be named the same as the input file.
    - input_format (str, optional): The format of the input file. Default is "pdb".
    - output_format (str, optional): The format of the output file. Default is "mol2".

    Returns:
    - str: The path of the output file, if it exists. Otherwise, it returns the path of the input file.
    '''
    
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