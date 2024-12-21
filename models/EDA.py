import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """
    Load data and perform initial preprocessing steps.
    
    Args:
        filepath (str): Path to the CSV file
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(filepath)
    
    # Encode categorical variables
    df["Churn"] = df["Churn"].map({'no': 0, 'yes': 1})
    df["Complains"] = df["Complains"].map({'no': 0, 'yes': 1})
    df["Status"] = df["Status"].map({'not-active': 0, 'active': 1})
   
    # Get dummies and convert boolean to int64
    df = pd.get_dummies(df, columns=['Plan'])
    df['Plan_post-paid'] = df['Plan_post-paid'].astype('int64')
    df['Plan_pre-paid'] = df['Plan_pre-paid'].astype('int64')

    print("\n=== Missing Values ===")
    print(df.isnull().sum())


    return df

def perform_initial_analysis(df):
    """
    Perform initial data analysis and print summary statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
    """
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Summary Statistics ===")
    print(df.describe().round(2))
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def visualize_churn_distribution(df):
    """
    Create visualization for churn distribution.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing churn data
    """
    fig, ax1 = plt.subplots(1, figsize=(15, 5))
    sns.countplot(data=df, x='Churn', ax=ax1, palette=['#ff9999', '#66b3ff'])
    ax1.set_title('Distribution of Customer Churn')
    ax1.set_xlabel('Churn Status')
    ax1.set_ylabel('Count')
    for i in ax1.containers:
        ax1.bar_label(i)
    plt.show()

def visualize_age_group_churn(df):
    """
    Create visualization for churn distribution across age groups.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing age and churn data
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df,
                x='Age Group',
                hue='Churn',
                multiple="dodge")
    plt.title('Distribution of Churn by Age Group', fontsize=12, pad=15)
    plt.xlabel('Age Group')
    plt.ylabel('Count of Customers')
    plt.show()

def visualize_charge_amount_churn(df):
    """
    Create visualization for churn distribution across charge amounts.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing charge amount and churn data
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df,
                x='Charge Amount',
                hue='Churn',
                multiple='dodge')
    plt.title("Distribution of Churn by Charge amount", fontsize=12, pad=15)
    plt.xlabel("Charge Amount")
    plt.ylabel("Churn")
    plt.show()

def analyze_charge_amounts(df):
    """
    Analyze and print charge amount statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing charge amount data
    """
    print("Unique Charge Amounts in the dataset:")
    print(sorted(df['Charge Amount'].unique()))
    print("\nFrequency of each Charge Amount:")
    print(df['Charge Amount'].value_counts().sort_index())

def plot_correlation_heatmap(data_frame):
    """
    Creates and displays a correlation heatmap for all numeric features in the dataset.
    
    Args:
        data_frame (pd.DataFrame): The dataset to analyze, with ID column to be excluded
    
    The function will:
        1. Calculate correlations between all numeric columns (excluding ID)
        2. Create a masked triangular heatmap for better readability
        3. Display correlation values with 2 decimal places
    """
    # Remove ID column and calculate correlation matrix
    correlation_matrix = data_frame.drop('ID', axis=1).corr()
    
    # Set the visual style for better appearance
    sns.set_theme(style="white")
    
    # Create a mask for the upper triangle
    # We do this because correlation matrices are symmetrical,
    # so we only need to show half to avoid redundancy
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Create the figure with a reasonable size
    plt.figure(figsize=(11, 9))
    
    # Generate a blue-red color palette centered at 0
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create and customize the heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,                  # Apply the triangular mask
                cmap=cmap,                  # Use our custom colormap
                vmax=1,                     # Set maximum correlation value
                center=0,                   # Center the colormap at 0
                square=True,                # Make cells square
                linewidth=.5,               # Add thin lines between cells
                cbar_kws={'shrink': .5},   # Customize the colorbar
                annot=True,                 # Show correlation values
                fmt='.2f')                  # Format to 2 decimal places
    
    # Add title and adjust layout
    plt.title('Correlation Matrix of Customer Churn Features', pad=20)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def prepare_train_test_split(df):
    """
    Prepare features and target, perform train-test split.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process
    
    Returns:
        tuple: (modified_df, X_train, X_test, y_train, y_test)
    """
    # Drop unnecessary columns and create new DataFrame
    df_modified = df.drop(['Age Group', 'Plan_post-paid', 'Freq. of SMS'], axis=1)
    
    X = df_modified.drop('Churn', axis=1)
    y = df_modified['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
    # return df_modified, X_train, X_test, y_train, y_test
    return df_modified, X_train, X_test, y_train, y_test

def main():
    """
    Main execution function that orchestrates the EDA process.
    """
    # Load and preprocess data
    df = load_and_preprocess_data('data/Customer_Churn.csv')
    
    # Perform initial EDA
    perform_initial_analysis(df)
    
    # Create visualizations
    visualize_churn_distribution(df)
    visualize_age_group_churn(df)
    visualize_charge_amount_churn(df)
    analyze_charge_amounts(df)
    
    # Prepare train-test split and get modified DataFrame
    df, X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Create correlation heatmap (final version with dropped columns)
    plot_correlation_heatmap(df)
    
    # Save splits for later use
    print("\nSaving train-test splits to files...")
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    print("Train-test splits saved successfully!")

if __name__ == "__main__":
    main()