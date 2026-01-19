import pandas as pd
import numpy as np

def add_phylogeny_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns phylogeny-inspired weights to AMR surveillance samples.
    Lower weight = likely clonal / outbreak
    Higher weight = unique epidemiological signal
    (Logic copied from phylogeny_weighting.py)
    """

    df = df.copy()

    # Group by epidemiological similarity
    group_cols = ["location", "pathogen", "antibiotic", "year"]
    # Handle cases where columns might be missing slightly gracefully? 
    # The app enforces them though.
    
    if not all(col in df.columns for col in group_cols):
        df['phylogeny_weight'] = 1.0
        return df

    group_sizes = df.groupby(group_cols).size().reset_index(name="cluster_size")

    # Merge cluster size back
    df = df.merge(group_sizes, on=group_cols, how="left")

    # Weight logic (inverse cluster density)
    df["phylogeny_weight"] = 1 / (1 + np.log1p(df["cluster_size"]))

    # Normalize to 0-1
    min_w = df["phylogeny_weight"].min()
    max_w = df["phylogeny_weight"].max()
    if max_w > min_w:
        df["phylogeny_weight"] = (df["phylogeny_weight"] - min_w) / (max_w - min_w + 1e-9)
    else:
        df["phylogeny_weight"] = 1.0

    return df
