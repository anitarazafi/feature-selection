import pandas as pd


def encode(df, method="onehot"):
    """
    Encode categorical features.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        "onehot" (default)

    Returns
    -------
    pd.DataFrame
    """
    if method == "onehot":
        cat_cols = df.select_dtypes(include=["category"]).columns
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

    return df