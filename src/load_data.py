import random
import numpy as np
import pandas as pd
from datasets import load_dataset

SEED = 42

def load_bodhisetu():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading Bodhisetu enriched dataset...")

    dataset = load_dataset(
        "51Hypers/Bodhisetu",
        "enriched",
        split="train"
    )

    df = dataset.to_pandas()

    images_df = df[
        df["media_type"].astype(str).str.upper() == "IMAGE"
    ].copy()

    print("Total records:", len(df))
    print("Total images:", len(images_df))

    return df, images_df

if __name__ == "__main__":
    df, images_df = load_bodhisetu()
    images_df.to_csv("data/dataset_manifest.csv", index=False)
    print("Manifest saved!")
