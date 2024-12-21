import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class LinearRegressionModel:
    """
    A base class for linear regression models. All of the models will use this 
    base class.
    """
    def __init__(self, model_name):
        self.model = LinearRegression()
        self.model_name = model_name
    
    def fit(self, X_train, y_train):
        """Train the model with given features"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance using standard metrics.
        Returns both metrics and predictions for flexibility.
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics, y_pred

    def plot_predictions(self, y_true, y_pred):
        """
        Create scatter plot of predicted vs actual values.
        
        Args:
            y_true (numpy.ndarray): True target values
            y_pred (numpy.ndarray): Predicted target values
        """
        plt.figure(figsize=(10, 6))
        # plot the true and predicted values, make them semi-tranparent if two points 
        # overlaps
        plt.scatter(y_true, y_pred, alpha=0.5)

        # this is the perfect prediction line
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2)

        plt.xlabel('Actual Customer Value')
        plt.ylabel('Predicted Customer Value')
        plt.title(f'Actual vs Predicted Values - {self.model_name}')
        plt.tight_layout()
        plt.show()