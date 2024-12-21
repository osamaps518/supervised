import numpy as np
import pandas as pd
from models.LinearRegressionModel import LinearRegressionModel


def main():
    """
    Implementation of LRM1: predicting Customer Value using all independent attributes
    """
    # Load the prepared data
    print("Loading data...")
    train_df = pd.read_csv('../data/train.csv')
    test_df  = pd.read_csv('../data/test.csv')
   
    independent_features = ['Age', 'Plan_pre-paid', 'Call Failure', 
                          'Complains', 'Distinct Called Numbers', 'Charge Amount', 
                          'Freq. of use', 'Status']
    
    X_train = train_df[independent_features]
    y_train = train_df['Customer Value']
    X_test = test_df[independent_features]
    y_test = test_df['Customer Value']

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