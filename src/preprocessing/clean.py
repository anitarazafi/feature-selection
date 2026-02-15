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


GENERIC_MISSING = [None, "", " "]


def normalize_missing_values(df):
    """
    Transform missing values to standart nan
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].replace(GENERIC_MISSING, np.nan)
    return df


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
    

def drop_high_missing_columns(df, threshold=0.6):
    """
    Drop columns with missing values above threshold.
    """
    # Calculate missing percentage
    missing_pct = df.isna().sum() / len(df)
    
    # Find columns above threshold
    high_missing = missing_pct[missing_pct > threshold]
    
    if len(high_missing) > 0:
        print(f"\n{'='*60}")
        print(f"DROPPING HIGH MISSING VALUE COLUMNS (> {threshold*100:.0f}%)")
        print(f"{'='*60}")
        print(f"Dropping {len(high_missing)} columns:")
        for col, pct in high_missing.items():
            print(f"  - {col:40s} : {pct*100:.1f}% missing")
        print(f"Columns: {len(df.columns)} -> {len(df.columns) - len(high_missing)}")
        print(f"{'='*60}\n")
    else:
        print(f"\nNo columns with > {threshold*100:.0f}% missing values\n")
    
    # Drop columns
    df_cleaned = df.drop(columns=high_missing.index.tolist())
    dropped_info = [(col, pct) for col, pct in high_missing.items()]
    
    return df_cleaned, dropped_info


def remove_target_leakage(X, y, threshold=0.9):
    """
    Remove features with extremely high correlation with target.
    """
    # Only check numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Calculate correlation with target
    target_corr = X_numeric.corrwith(y).abs()
    
    # Find features above threshold
    leaky_features = target_corr[target_corr > threshold].index.tolist()
    
    if leaky_features:
        print(f"\n{'='*60}")
        print(f"REMOVING TARGET LEAKAGE (correlation > {threshold})")
        print(f"{'='*60}")
        print(f"Removing {len(leaky_features)} features:")
        for feat in leaky_features:
            corr_value = target_corr[feat]
            print(f"  - {feat:30s} : r = {corr_value:.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n✓ No features with correlation > {threshold}\n")
    
    # Remove leaky features from entire dataframe (including non-numeric)
    X_cleaned = X.drop(columns=leaky_features, errors='ignore')
    
    return X_cleaned, leaky_features