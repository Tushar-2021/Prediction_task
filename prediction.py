import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Data Validation and Cleaning
def read_csv(file_path):
    """Read the CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print("Error reading CSV file:", e)

def clean_data(df):
    """Handle missing values, outliers, and inconsistencies."""
    #
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values detected. Handling missing values...")
        df.fillna(method='ffill', inplace=True)  

    

    return df

# Step 2: Stock Analysis
def analyze_stock(df):
    """Calculate statistics and visualize historical trends."""
    # 
    df['Date'] = pd.to_datetime(df['Date'])

    
    basic_stats = df.describe()

    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
    plt.title('Historical Trends of Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    

    
    correlations = df.corr()

    return basic_stats, correlations

# Step 3: Predictive Modeling
def build_model(df):
    """Build a predictive model for the stock."""
    #
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return model, rmse, mae

# Step 4: Deployment and Monitoring (not implemented for brevity)


def main():
    
    file_path = 'NSEI.csv'  
    df = read_csv(file_path)
    if df is not None:
        clean_df = clean_data(df)

        
        basic_stats, correlations = analyze_stock(clean_df)
        print("Basic Statistics:")
        print(basic_stats)
        print("\nCorrelation Matrix:")
        print(correlations)

        
        model, rmse, mae = build_model(clean_df)
        print("\nModel Evaluation:")
        print("RMSE:", rmse)
        print("MAE:", mae)

       
    else:
        print("Error: Unable to read CSV file.")

if __name__ == "__main__":
    main()
