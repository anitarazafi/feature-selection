from src.preprocessing.clean import normalize_missing_values, correct_dtypes, handle_missing, drop_high_missing_columns, remove_target_leakage

def preprocess(df, cfg):
    remove_first_line = cfg["preprocessing"]["remove_first_row"]
    if remove_first_line:
        df = df.iloc[1:].reset_index(drop=True) # drop duplicated header/metadata row

    df = normalize_missing_values(df)

    remove_high_missing_values_columns = cfg["preprocessing"]["remove_high_missing_values_columns"]
    if remove_high_missing_values_columns:
        high_missing_values_threshold = cfg["preprocessing"]["high_missing_values_threshold"]
        df = drop_high_missing_columns(df, high_missing_values_threshold)

    df = correct_dtypes(df, cfg)

    sort_col = cfg.get("features", {}).get("sort_by")
    if sort_col and sort_col in df.columns:
        print(f"\n{'='*60}")
        print(f"= Sorting rows")
        df = df.sort_values(sort_col)

    missing_values_strategy = cfg["preprocessing"].get("missing_values_strategy")
    df = handle_missing(df, missing_values_strategy)

    drop_duplicates = cfg["preprocessing"].get("drop_duplicates")
    if drop_duplicates:
        print(f"\n{'='*60}")
        print(f"= Dropping duplicate values if present")
        df = df.drop_duplicates()

    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]

    X = remove_target_leakage(X, y, threshold=0.9)
    return X, y
