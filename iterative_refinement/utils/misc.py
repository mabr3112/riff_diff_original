"""
misc.py - Mischellaneous Functions to be used in iterative_refinement.

This module provides utility functions working with pandas DataFrames for Poses class objects.

Functions:
-----------
- `combine_motifs_in_df(df, col1, col2, operator)`: 
    Combine motifs from two columns in a DataFrame based on a specified operator. 
    Supported operators are "and", "or", "xand", "xor".
    
- `combine_motifs(motif1, motif2, operator)`: 
    Combine two motifs represented as dictionaries based on a specified operator. 
    Supported operators are "and", "or", "xand", "xor".

Example:
--------
>>> import pandas as pd
>>> df = pd.DataFrame({'col1': [{'A': [1, 2]}, {'B': [2, 3]}],
...                    'col2': [{'A': [2, 3]}, {'B': [3, 4]}]})
>>> combine_motifs_in_df(df, 'col1', 'col2', 'and')
[{'A': [2]}, {'B': [3]}]

Note:
-----
This module uses the `pandas` library for DataFrame manipulations.

"""

import pandas as pd

def combine_motifs_in_df(df:pd.DataFrame, col1:str=None, col2:str=None, operator:str="and") -> list[dict]:
    """
    Combine motifs from two columns in a DataFrame based on a specified operator.

    This function iterates over the rows of the DataFrame, taking motifs from `col1` and `col2` 
    and combining them using the `combine_motifs` function based on the specified `operator`.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the motifs to combine.
    col1 : str, optional
        The name of the first column containing motifs. Defaults to None.
    col2 : str, optional
        The name of the second column containing motifs. Defaults to None.
    operator : str, optional
        The operator to use for combining motifs. Supported operators are "and", "or", "xand", "xor". 
        Defaults to "and".

    Returns:
    --------
    list[dict]
        A list of dictionaries, each representing a combined motif.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'col1': [{'A': [1, 2]}, {'B': [2, 3]}],
    ...                    'col2': [{'A': [2, 3]}, {'B': [3, 4]}]})
    >>> combine_motifs_in_df(df, 'col1', 'col2', 'and')
    [{'A': [2]}, {'B': [3]}]

    Note:
    -----
    This function assumes that the motifs in `col1` and `col2` are dictionaries.
    """
    return [combine_motifs(m1, m2, operator=operator) for m1, m2 in zip(df[col1].to_list(), df[col2].to_list())]

def combine_motifs(motif1:dict=None, motif2:dict=None, operator:str="and") -> dict:
    """
    Combine two motifs represented as dictionaries based on a specified operator.

    This function takes two motifs, represented as dictionaries, and combines them 
    based on the specified `operator`. The keys in the dictionaries are assumed to be
    motif labels, and the values are lists of positions.

    Parameters:
    -----------
    motif1 : dict, optional
        The first motif to combine, represented as a dictionary. Defaults to an empty dictionary.
    motif2 : dict, optional
        The second motif to combine, represented as a dictionary. Defaults to an empty dictionary.
    operator : str, optional
        The operator to use for combining motifs. Supported operators are "and", "or", "xand", "xor". 
        Defaults to "and".

    Returns:
    --------
    dict
        A dictionary representing the combined motif.

    Raises:
    -------
    NotImplementedError
        If an unsupported operator is specified.

    Example:
    --------
    >>> combine_motifs({'A': [1, 2]}, {'A': [2, 3]}, 'and')
    {'A': [2]}

    >>> combine_motifs({'A': [1, 2]}, {'A': [2, 3]}, 'or')
    {'A': [1, 2, 3]}

    Note:
    -----
    The function performs set operations on the list of positions for each motif label to combine them.
    """

    # Initialize an empty dictionary to store the new motifs
    new_motif = {}
    if motif1 is None: motif1 = dict()
    if motif2 is None: motif2 = dict()

    # Iterate through the keys of the first motif dictionary
    for key in motif1.keys():

        # Check if the same key exists in the second motif dictionary
        if key in motif2:
            # make sure both motifs have int as residue indeces.
            # Convert the lists to sets for set operations
            set1 = set([int(x) for x in motif1[key]])
            set2 = set([int(x) for x in motif2[key]])

            # Perform the operation based on the operator argument
            if operator == "and":
                new_set = set1.intersection(set2)
            elif operator == "or":
                new_set = set1.union(set2)
            elif operator == "xand":
                new_set = set1.symmetric_difference(set2).difference(set1.intersection(set2))
            elif operator == "xor":
                new_set = set1.symmetric_difference(set2)
            else:
                raise NotImplementedError(f"The specified operator '{operator}' is not implemented yet!")

            # Convert the resulting set back to a list and sort it
            new_list = sorted(list(new_set))

            # Add it to the new motif dictionary
            new_motif[key] = new_list

    return new_motif
