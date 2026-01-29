import pandas as pd

def encode(df, categorical_cols, encoding="onehot"):
    df = df.copy()

    if not categorical_cols:
        return df

    if encoding == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    elif encoding == "label":
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

    else:
        raise ValueError(f"Unknown encoding strategy: {encoding}")

    return df
