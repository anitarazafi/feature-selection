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

def split_data_train_test(X, y):
    """Split data temporally (respecting time order)."""
    # Assuming X has ISO_TIME column or index
    split_idx = int(len(X) * 0.8)  # 80% train, 20% test
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
