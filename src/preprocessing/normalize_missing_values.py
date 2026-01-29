import numpy as np 
import pandas as pd
GENERIC_MISSING = [None, "", " "]

def normalize_missing_values(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].replace(GENERIC_MISSING, np.nan)
    return df