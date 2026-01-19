import pandas as pd
import numpy as np

def add_phylogeny_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns phylogeny-inspired weights to AMR surveillance samples.
    Lower weight = likely clonal / outbreak
    Higher weight = unique epidemiological signal
    """

    df = df.copy()

    # Group by epidemiological similarity
    group_cols = ["location", "pathogen", "antibiotic", "year"]
    group_sizes = df.groupby(group_cols).size().reset_index(name="cluster_size")

    # Merge cluster size back
    df = df.merge(group_sizes, on=group_cols, how="left")

    # Weight logic (inverse cluster density)
    df["phylogeny_weight"] = 1 / (1 + np.log1p(df["cluster_size"]))

    # Normalize to 0–1
    df["phylogeny_weight"] = (
        (df["phylogeny_weight"] - df["phylogeny_weight"].min())
        / (df["phylogeny_weight"].max() - df["phylogeny_weight"].min() + 1e-9)
    )

    return df


def analyze_weights(df: pd.DataFrame) -> dict:
    """
    Summary statistics for phylogeny weights
    """

    return {
        "mean_weight": float(df["phylogeny_weight"].mean()),
        "std_weight": float(df["phylogeny_weight"].std()),
        "min_weight": float(df["phylogeny_weight"].min()),
        "max_weight": float(df["phylogeny_weight"].max()),
        "outbreak_like_samples": int((df["phylogeny_weight"] < 0.3).sum()),
        "unique_samples": int((df["phylogeny_weight"] > 0.7).sum()),
    }
