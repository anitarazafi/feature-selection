import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_train_val_test(X_train, X_val, X_test):
    """
    Fit encoder on train, transform train, val and test.
    Returns: X_train_encoded, X_val_encoded, X_test_encoded, encoder
    """
    categorical_cols = X_train.select_dtypes(include=['category']).columns
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    if len(categorical_cols) == 0:
        return X_train, X_val, X_test, None

    # Fit encoder on TRAIN only
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])

    # Transform all three
    train_encoded = encoder.transform(X_train[categorical_cols])
    val_encoded   = encoder.transform(X_val[categorical_cols])
    test_encoded  = encoder.transform(X_test[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    X_train_encoded = pd.concat([
        X_train[numerical_cols].reset_index(drop=True),
        pd.DataFrame(train_encoded, columns=feature_names)
    ], axis=1)
    X_val_encoded = pd.concat([
        X_val[numerical_cols].reset_index(drop=True),
        pd.DataFrame(val_encoded, columns=feature_names)
    ], axis=1)
    X_test_encoded = pd.concat([
        X_test[numerical_cols].reset_index(drop=True),
        pd.DataFrame(test_encoded, columns=feature_names)
    ], axis=1)

    return X_train_encoded, X_val_encoded, X_test_encoded, encoder
