from src.preprocessing.normalize_missing_values import normalize_missing_values
from src.preprocessing.clean import correct_dtypes, handle_missing, drop_high_missing_columns, remove_target_leakage

def preprocess(df, cfg):
    df = df.iloc[1:].reset_index(drop=True) # drop duplicated header/metadata row
    df = normalize_missing_values(df)
    df, dropped_missing = drop_high_missing_columns(df, threshold=0.6)
    df = correct_dtypes(df, cfg["features"])
    df = df.sort_values("ISO_TIME")
    df = handle_missing(df, cfg["preprocessing"].get("missing_values"))
    df = df.drop_duplicates()
    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]
    X, removed_features = remove_target_leakage(X, y, threshold=0.9)
    return X, y
