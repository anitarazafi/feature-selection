from src.preprocessing.clean import drop_columns, correct_dtypes, handle_missing

def preprocess(df, cfg):
    df = drop_columns(df, cfg["features"].get("drop"))
    df = correct_dtypes(df, cfg["features"])
    df = handle_missing(df, cfg["preprocessing"].get("missing_values"))
    df = df.drop_duplicates()

    X = df.drop(cfg["target"], axis=1)
    y = df[cfg["target"]]

    return X, y
