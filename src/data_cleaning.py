import pandas as pd
import numpy as np

def preprocess_bio_leaching(master_df):
    """
    Cleans and structures bio-leaching time-series data
    using process time (hours).
    """

    df = master_df.copy()

    # ---- 1. Sort by Flask and process time ----
    df = df.sort_values(['Flask_ID', 'Time_hr'])

    # ---- 2. Replace placeholders with NaN ----
    df.replace(['NO', 'N/A', 'NAN'], np.nan, inplace=True)

    # ---- 3. Convert numeric columns ----
    numeric_cols = ['pH', 'OD600', 'Temp_C', 'PulpDensity_g/l', 'Time_hr']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---- 4. Create process time in days ----
    df['Time_days'] = df['Time_hr'] / 24.0

    # ---- 5. Interpolate within each flask ----
    df[['pH', 'OD600']] = (
        df.groupby('Flask_ID')[['pH', 'OD600']]
        .apply(lambda x: x.interpolate(method='linear'))
        .reset_index(drop=True)
    )

    return df
