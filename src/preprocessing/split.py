from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, seed=42, stratify=True):
    """
    Split data into train/test.
    """
    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }