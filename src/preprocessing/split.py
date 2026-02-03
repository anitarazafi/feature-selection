def split_data(X, y):
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
