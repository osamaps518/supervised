import numpy as np
from models.LinearRegressionModel import LinearRegressionModel


def main():
    """
    Implementation of LRM1: predicting Customer Value using all independent attributes
    """
    # Load the prepared data
    print("Loading data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
        # Diagnostic check 1: Check for NaN values in loaded data
    print("\nDiagnostic Check 1 - NaN Count:")
    print(f"X_train NaN count: {np.isnan(X_train).sum()}")
    print(f"X_test NaN count: {np.isnan(X_test).sum()}")
    print(f"y_train NaN count: {np.isnan(y_train).sum()}")
    print(f"y_test NaN count: {np.isnan(y_test).sum()}")
    
    # Create and train LRM1
    print("\nTraining LRM1 model...")
    lrm1 = LinearRegressionModel(model_name="LRM1")
    lrm1.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating LRM1 performance...")
    metrics, predictions = lrm1.evaluate(X_test, y_test)
    
    # Print results
    print("\nLRM1 Performance Metrics:")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")

    # Plot predictions
    lrm1.plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main()