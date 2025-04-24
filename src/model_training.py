"""
Model training module for stock price prediction.
Includes functions for data preparation, model training, evaluation, and saving.
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def prepare_data_for_training(df, target_column, features=None, test_size=0.2, scale=True):
    """
    Prepare data for model training.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and target
    target_column : str
        Name of the target column (e.g., 'Pct_Change_1')
    features : list, optional
        List of feature columns to use. If None, uses all non-target columns.
    test_size : float, default=0.2
        Proportion of data to use for testing
    scale : bool, default=True
        Whether to scale the features

    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, scaler, selected_features
    """
    print(f"Preparing data for predicting {target_column}...")

    # Make a copy of the dataframe
    data = df.copy()

    # Remove rows with NaN in the target column
    data = data.dropna(subset=[target_column])

    # Identify target-related columns to exclude from features
    target_patterns = ['Future_', 'Price_Change_', 'Pct_Change_', 'Target_Direction_']
    target_cols = [col for col in data.columns
                   if any(pattern in col for pattern in target_patterns)]

    # Remove Date column if present
    if 'Date' in data.columns:
        data_cols = data.columns.drop('Date')
    else:
        data_cols = data.columns

    # Select features
    if features is None:
        # Use all columns except targets and date
        features = [col for col in data_cols if col not in target_cols]

    print(f"Selected {len(features)} features")

    # Split data chronologically (no shuffling for time series)
    X = data[features]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Scale features if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert back to DataFrames with column names
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = pd.DataFrame(X_test, columns=features)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, features


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest regression model.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of the trees
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    RandomForestRegressor
        Trained model
    """
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame of feature importances
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(importance_df.head(10))

    return model


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
    """
    Train an XGBoost regression model.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.1
        Learning rate
    max_depth : int, default=6
        Maximum depth of the trees
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    xgb.XGBRegressor
        Trained model
    """
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame of feature importances
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(importance_df.head(10))

    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance on test data.

    Parameters:
    -----------
    model : trained model
        The trained machine learning model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_name : str, default="Model"
        Name of the model for printing

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Calculate directional accuracy (for price movement prediction)
    directional_accuracy = np.mean((y_test > 0) == (y_pred > 0))
    print(f"Directional Accuracy: {directional_accuracy:.4f}")

    # Return metrics as a dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }

    return metrics, y_pred


def plot_predictions(y_test, y_pred, model_name, target_column, ticker):
    """
    Plot actual vs. predicted values.

    Parameters:
    -----------
    y_test : pandas.Series
        Actual values
    y_pred : numpy.ndarray
        Predicted values
    model_name : str
        Name of the model
    target_column : str
        Name of the target column
    ticker : str
        Stock ticker symbol
    """
    plt.figure(figsize=(12, 6))

    # Convert y_test to numpy array if it's not
    if isinstance(y_test, pd.Series):
        y_test_values = y_test.values
    else:
        y_test_values = y_test

    # Create an index for the x-axis
    x = np.arange(len(y_test_values))

    # Plot actual and predicted values
    plt.plot(x, y_test_values, label='Actual', marker='o', alpha=0.6)
    plt.plot(x, y_pred, label='Predicted', marker='x', alpha=0.6)

    # Add a line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    # Set title and labels
    plt.title(f'{ticker} - {model_name} Predictions for {target_column}')
    plt.xlabel('Test Sample')
    plt.ylabel(target_column)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True, alpha=0.3)

    # Show plot
    plt.tight_layout()

    # Save plot
    output_path = f'data/{ticker}_{model_name}_{target_column}_predictions.png'
    plt.savefig(output_path)
    print(f"Saved prediction plot to {output_path}")

    plt.show()


def save_model(model, scaler, features, target_column, model_name, ticker):
    """
    Save model, scaler, and metadata to disk.

    Parameters:
    -----------
    model : trained model
        The trained machine learning model
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for the features
    features : list
        List of feature names
    target_column : str
        Name of the target column
    model_name : str
        Name of the model
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    str
        Path to the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Create model file name
    model_filename = f'models/{ticker}_{model_name}_{target_column.replace("/", "_")}.pkl'

    # Create metadata
    metadata = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'target_column': target_column,
        'ticker': ticker,
        'model_name': model_name,
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    # Save the model and metadata
    with open(model_filename, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Model saved to {model_filename}")

    return model_filename


def load_model(model_path):
    """
    Load a saved model and its metadata.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file

    Returns:
    --------
    dict
        Dictionary containing the model and its metadata
    """
    with open(model_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Loaded model for {metadata['ticker']} to predict {metadata['target_column']}")

    return metadata


def make_prediction(model_metadata, new_data):
    """
    Make predictions using a loaded model.

    Parameters:
    -----------
    model_metadata : dict
        Model metadata from load_model
    new_data : pandas.DataFrame
        New data to make predictions on

    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    # Extract model components from metadata
    model = model_metadata['model']
    scaler = model_metadata['scaler']
    features = model_metadata['features']

    # Select required features
    X = new_data[features]

    # Scale the features if a scaler is provided
    if scaler is not None:
        X = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X)

    return predictions


if __name__ == "__main__":
    # Test the model training functions
    import os
    from feature_engineering import generate_features

    # Example usage with a specific ticker
    ticker = 'AAPL'

    # Load features data
    features_csv = f'data/{ticker}_features_' + pd.Timestamp.now().strftime('%Y%m%d') + '.csv'

    if os.path.exists(features_csv):
        print(f"Loading existing features from {features_csv}")
        features_df = pd.read_csv(features_csv)

        # Convert Date column to datetime if it exists
        if 'Date' in features_df.columns:
            features_df['Date'] = pd.to_datetime(features_df['Date'])
    else:
        print(f"Features file not found: {features_csv}")
        print("Please run feature_engineering.py first to generate features")
        exit(1)

    # Define target column (what we want to predict)
    # Options:
    # - 'Pct_Change_1': Percentage change for next day
    # - 'Target_Direction_1': Binary classification (up or down) for next day
    # - 'Price_Change_1': Absolute price change for next day
    target_column = 'Pct_Change_1'

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, selected_features = prepare_data_for_training(
        features_df, target_column, test_size=0.2
    )

    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=10)

    # Evaluate Random Forest model
    rf_metrics, rf_predictions = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Train XGBoost model
    xgb_model = train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=6)

    # Evaluate XGBoost model
    xgb_metrics, xgb_predictions = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Plot predictions
    plot_predictions(y_test, rf_predictions, "RandomForest", target_column, ticker)
    plot_predictions(y_test, xgb_predictions, "XGBoost", target_column, ticker)

    # Save models
    rf_model_path = save_model(rf_model, scaler, selected_features, target_column, "RandomForest", ticker)
    xgb_model_path = save_model(xgb_model, scaler, selected_features, target_column, "XGBoost", ticker)

    # Test loading and prediction
    loaded_model = load_model(rf_model_path)

    # Get the most recent data point
    recent_data = features_df.tail(1)

    # Make a prediction
    prediction = make_prediction(loaded_model, recent_data)

    print(f"\nPrediction for next day {target_column}: {prediction[0]:.4f}")

    if target_column.startswith('Pct_Change'):
        current_price = recent_data['Close'].values[0]
        predicted_price = current_price * (1 + prediction[0] / 100)
        print(f"Current price: ${current_price:.2f}")
        print(f"Predicted next price: ${predicted_price:.2f}")