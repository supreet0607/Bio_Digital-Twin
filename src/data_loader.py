import os
import pandas as pd

def load_bio_leaching_data():
    # Get project root directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    master_path = os.path.join(
        BASE_DIR, "data", "raw", "biomasterdata.csv"
    )
    meta_path = os.path.join(
        BASE_DIR, "data", "metadata", "bio_metadata.csv"
    )

    master_df = pd.read_csv(master_path)
    meta_df = pd.read_csv(meta_path)

    return master_df, meta_df
