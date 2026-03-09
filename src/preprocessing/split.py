def split_data_train_val_test(X, y):
    """Split data temporally (respecting time order). 70% train, 10% val, 20% test."""
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    return {
        "X_train": X.iloc[:train_end],
        "X_val":   X.iloc[train_end:val_end],
        "X_test":  X.iloc[val_end:],
        "y_train": y.iloc[:train_end],
        "y_val":   y.iloc[train_end:val_end],
        "y_test":  y.iloc[val_end:],
    }
