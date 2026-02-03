import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    """
    Fit scaler on train, transform both train and test.
    Returns: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train only
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler