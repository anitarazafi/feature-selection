import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode(X_train, X_test):
    """
    Fit encoder on train, transform both train and test.
    Returns: X_train_encoded, X_test_encoded, encoder
    """
    categorical_cols = X_train.select_dtypes(include=['category']).columns
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    if len(categorical_cols) == 0:
        return X_train, X_test, None
    
    # Fit encoder on TRAIN only
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    # Transform both
    train_encoded = encoder.transform(X_train[categorical_cols])
    test_encoded = encoder.transform(X_test[categorical_cols])
    # Get feature names
    feature_names = encoder.get_feature_names_out(categorical_cols)
    # Combine with numerical columns
    X_train_encoded = pd.concat([
        X_train[numerical_cols].reset_index(drop=True),
        pd.DataFrame(train_encoded, columns=feature_names)
    ], axis=1)
    
    X_test_encoded = pd.concat([
        X_test[numerical_cols].reset_index(drop=True),
        pd.DataFrame(test_encoded, columns=feature_names)
    ], axis=1)
    
    return X_train_encoded, X_test_encoded, encoder
