#!/home/mabr3112/anaconda3/bin/python3.9
#
#    This code should read in a scorefile from <input_path>, normalize columns of specified <scoreterms>
#    and add them weighted by the specified <weights>. This new score should then be added to the
#    scorefile which is subsequently saved.
#
#######################

import pandas as pd
from sklearn.preprocessing import StandardScaler

def prep_composite_score_inputs(input_data):
    if type(input_data) == str:
        out_list = input_data.split(",")
    elif type(input_data) == list:
        out_list = input_data
    else:
        raise TypeError(f"ERROR: Argument type {type(input_data)} not supported for function calc_composite_score! Change inputs.")
    return out_list


def calc_composite_score(input_data, name: str, scoreterms: list[str], weights: list[float]) -> pd.DataFrame:
    '''
    This code should read in a scorefile from <input_path>, normalize columns of specified <scoreterms>
    and add them weighted by the specified <weights>. This new score should then be added to the
    scorefile which is subsequently saved.

    <input_path>: Can be either a pd.DataFrame, or a path to a .json or .sc File.
    <name>: Name of the composite_score that you want to calculate and add to your DataFrame.
    <scoreterms>: Scoreterms from which to calculate the composite_score.
    <weights>: Weights to give to <scoreterms>.
    <index_col>: column that should be used as index for recombining composite score with original DataFrame.
    '''
    # prepare cols and weights
    cols = prep_composite_score_inputs(scoreterms)
    w = prep_composite_score_inputs(weights)

    # read in scores from file, extract scoreterms and normalize them.
    if type(input_data) == str:
        if input_data.endswith(".sc"):
            print(f"Type of input Data at {input_data} automatically determined to be Rosetta Scorefile.")
            df = pd.read_csv(input_data, delim_whitespace=True, header=[1], na_filter=True)
        else:
            print(f"Type of input Data at {input_data} automatically determined to be .json")
            df = pd.read_json(input_data)
    elif type(input_data) == pd.DataFrame:
        df = input_data
    else:
        raise TypeError(f"ERROR: Unsupported datatype for your input_data argument: {type(input_data)}. Only str (path to .json file) or pd.DataFrame are valid.")
    print(f"Normalizing specified columns.")
    
    # calculate composite score:
    return calc_score_by_weights(df, name=name, cols=cols, weights=weights)

def normalize_df(df):
    '''
    performs normalization with sklearn on the numbered columns of a DataFrame
    '''
    if df.isnull().values.any(): raise ValueError(f"Your DataFrame contains NaN values. Make sure to remove them!!")
    numbered_cols = list(df.select_dtypes('number').columns) # just to be sure

    # Separating out the features and Standardizing the columns
    x = df.loc[:, numbered_cols].values
    x = StandardScaler().fit_transform(x)
    normalized_df = pd.DataFrame(data=x, columns=numbered_cols)

    return normalized_df

def calc_score_by_weights(df: pd.DataFrame, name: str, cols: list, weights: list) -> pd.DataFrame:
    '''
    Add a custom new score of name <name> to your DataFrame.
    This score is calculated from all <cols> scaled by the listed <weights>

    Returns Pandas DataFrame.
    '''
    # Check if cols and weights are the same length:
    if not len(cols) == len(weights): raise ValueError(f"Cols and Weights are not of the same length: {cols}, {weights}")

    # Check if all scoreterms provided in <cols> are in df:
    df_cols = df.columns
    for col in cols:
        if col not in df_cols: raise KeyError(f"ERROR: Column {col} not found in DataFrame. Available columns: {[x for x in df_cols]}")
    df[name] = sum([(df[col] - df[col].median()) / df[col].std() * weight for col, weight in zip(cols, weights)])
    return df

def main(args):
    '''
    '''
    ll = ["confidence", "pLDDT", "plddt"]
    # DAU check
    for c, w in zip(args.scoreterms, args.weights):
        if c in ll:
            if w > 0:
                print(f"Warning: your specified weight for {c} is positive ({w}). This means that you weight lower {c} scores as better.")
    print(f"\n{'#'*50}\nRunning calc_composite_score.py on {args.input_path}\n{'#'*50}\n")
    config = '\n'.join([f"{s}: {w}" for s, w in zip(args.scoreterms, args.weigths)])
    print(f"Configuration:\n{config}\n")
    score_df = calc_composite_score(args.input_path, args.name, args.scoreterms, args.weights)
    print(f"Writing new score {args.name} into {args.input_path}")
    score_df.to_json(args.input_path)

if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", required=True, help="Path to the scorefile in which you want to calculate the new score.")
    argparser.add_argument("--name", required=True, help="Name of the new scoreterm you want to calculate")
    argparser.add_argument("--scoreterms", required=True, help="Comma-separated list of scoreterms. These scoreterms will be taken to calculate the composite score. Example: --scoreterms='pLDDT,bb_ca_rmsd,motif_heavy_rmsd'")
    argparser.add_argument("--weights", required=True, help="Comma-separated list of weights for the specified scoreterms. Composite score will be calculated by summing up the scoreterms scaled by the provided weights: composite_score=[sum(scoreterm_i*weight_i)]. Example: --weights='-1,2,1'. IMPORTANT NOTE: The sign in front of the weight specifies the 'direction' of the score. For AF2 pLDDT, a negative weight has to be provided, because higher values mean higher confidence, while for Rosetta Scores weights should generally be positive, since lower scores generally mean lower (better) Energies.")
    args = argparser.parse_args()

    main(args)
