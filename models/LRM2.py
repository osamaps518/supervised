import pandas as pd
from models.LinearRegressionModel import LinearRegressionModel


def main():
    """
    Implementation of LRM2: predicting Customer Value using all independent attributes
    """
    # Load the prepared data
    print("Loading data...")
    train_df = pd.read_csv('../data/train.csv')
    test_df  = pd.read_csv('../data/test.csv')
    
    # These features show strong correlation with Customer Value
    X_train = train_df[['Freq. of use', 'Distinct Called Numbers']]
    y_train = train_df['Customer Value']
    X_test = test_df[['Freq. of use', 'Distinct Called Numbers']]
    y_test = test_df['Customer Value']
    
    # Create and train LRM1
    print("\nTraining LRM2 model...")
    lrm2 = LinearRegressionModel(model_name="LRM2")
    lrm2.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating LRM2 performance...")
    metrics, predictions = lrm2.evaluate(X_test, y_test)
    
    # Print results
    print("\nLRM2 Performance Metrics:")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")

    # Plot predictions
    lrm2.plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main()