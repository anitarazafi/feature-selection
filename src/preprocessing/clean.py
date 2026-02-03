import pandas as pd
import numpy as np

VALID_CATEGORIES = {
    "BASIN": {"NA", "EP", "WP", "NI", "SI", "SP", "SA"},
    "SUBBASIN": {"CS", "GM", "CP", "BB", "AS", "WA", "EA"},
    "NATURE": {"DS", "TS", "ET", "SS", "NR", "MX"},
    "WMO_AGENCY": {"bom", "nadi", "wellington", "reunion", "newdelhi", "tokyo", "hurdat_epa", "hurdat_atl", "atcf"},
    "TRACK_TYPE": {"main", "spur", "provisional", "us-provisional", "us-provisional_spur"},
    "USA_AGENCY": {"hurdat_atl", "hurdat_epa", "atcf", "jtwc_wp", "jtwc_io", "jtwc_ep", "jtwc_cp", "jtwc_sh", "cphc"},
    "USA_RECORD": {"C", "G", "I", "L", "P", "R", "S", "T", "W"},
    "USA_STATUS": {"DB", "TD", "TS", "TY", "ST", "TC", "HU", "HR", "SD", "SS", "EX", "PT", "IN", "DS", "LO", "WV", "ET", "MD", "XX"},
    "NEWDELHI_GRADE": {"D", "DD", "CS", "SCS", "VSCS", "SCS"},
    "MLC_CLASS": {"EX", "HU", "LO", "MH", "SD", "SS", "TD", "TS", "TW", "WV"},
    "HKO_CAT": {"LW", "TD", "TS", "STS", "T", "ST", "SuperT"},
    "KMA_CAT": {"TD", "TS", "STS", "TY", "L"}
}

def drop_columns(df, columns):
    """
    Drop specified columns if they exist.
    """
    if not columns:
        return df
        
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)
    

def correct_dtypes(df, cfg):
    """
    Correct column data types
    """
    df = df.copy()
    # datetime columns
    datetime_cols = cfg.get("datetime", [])
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
                utc=True
            )

    # numeric columns
    obj_cols = df.select_dtypes(include="object").columns
    numeric_like = obj_cols[
        df[obj_cols].apply(lambda c: pd.to_numeric(c, errors="coerce").notna().any())
    ]
    for col in numeric_like:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # categorical columns
    for col, valid_set in VALID_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[~df[col].isin(valid_set), col] = np.nan
            df[col] = df[col].astype("category")

    return df


def handle_missing(df, strategy="mean"):
    """
    Handle missing values.
    strategy:
    - "mean"   : numerical mean
    - "median" : numerical median
    - "most_frequent" : mode
    - "drop"   : drop rows with any missing values
    """
    if strategy == "drop":
        return df.dropna()

    df = df.copy()
    if strategy in ["mean", "median"]:
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            value = df[col].mean() if strategy == "mean" else df[col].median()
            df[col] = df[col].fillna(value)
    
    if strategy == "most_frequent":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df
