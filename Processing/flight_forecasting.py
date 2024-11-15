# flight_forecasting.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ARIMA
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# For Linear Regression
from sklearn.linear_model import LinearRegression

# For Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# For Prophet
from prophet import Prophet

# For Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')



# ---------------------------
# Configuration
# ---------------------------

# Define the input directory containing the CSV files for origin airports
INPUT_DIR = 'D:/GitHub Repos/630-SOUTHWEST-SCHEDULING/Data/'

# Define the output directory to save forecasts and visualizations
OUTPUT_DIR = 'D:/GitHub Repos/630-SOUTHWEST-SCHEDULING/Forecasts/'

# List of CSV filenames corresponding to different origin airports
CSV_FILES = [
    'Origin Airport Baltimore, MD BaltimoreWashington International Thurgood Marshall (BWI).csv',
    'Origin Airport Chicago, IL Chicago Midway International (MDW).csv',
    'Origin Airport Dallas, TX Dallas Love Field (DAL).csv',
    'Origin Airport Denver, CO Denver International (DEN).csv',
    'Origin Airport Las Vegas, NV Harry Reid International (LAS).csv'
]

# Specify the month for which to perform forecasting (June)
FORECAST_MONTH = 6

# ---------------------------
# Function Definitions
# ---------------------------

def load_data(filepath):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath} successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {filepath}")
        return None

def clean_and_prepare_data(df, month=6):
    """
    Cleans the DataFrame by handling missing values, filtering for a specific month,
    and extracting relevant date components. Additionally, retains only specified
    destination airports.

    Parameters:
        df (pd.DataFrame): Raw DataFrame.
        month (int): Month to filter data for (default is June).

    Returns:
        pd.DataFrame: Cleaned DataFrame with specified destination airports.
    """
    # Convert 'Date (MM/DD/YYYY)' to datetime, coercing errors to NaT
    df['Date'] = pd.to_datetime(df['Date (MM/DD/YYYY)'], format='%m/%d/%Y', errors='coerce')
    
    # Drop rows with invalid dates (NaT)
    df = df.dropna(subset=['Date']).copy()
    
    # Extract Month and Year from the 'Date' column
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Filter the DataFrame for the specified month (e.g., June)
    df_month = df[df['Month'] == month].copy()
    
    # Drop rows where 'Destination Airport' is missing
    df_clean = df_month.dropna(subset=['Destination Airport']).copy()
    
    # Define the list of allowed destination airport codes
    allowed_destinations = ['SAN', 'HNL', 'OKC', 'LGA', 'MIA', 'LAX', 'SEA', 'CUN', 'DCA', 'CHS']
    
    # Filter the DataFrame to retain only the specified destination airports
    df_clean = df_clean[df_clean['Destination Airport'].isin(allowed_destinations)].copy()
    
    # (Optional) Reset index after filtering for cleanliness
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def aggregate_flights(df):
    """
    Aggregates the number of flights per year.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: Aggregated flights per year.
    """
    flights_per_year = df.groupby('Year').size().reset_index(name='Number_of_Flights')
    return flights_per_year

def initialize_results_dataframe():
    """
    Initializes an empty DataFrame to store forecasting results.

    Returns:
        pd.DataFrame: Empty results DataFrame.
    """
    results = pd.DataFrame({
        'Year': [],
        'Model': [],
        'Predicted': [],
        'Actual': [],
        'MAE': [],
        'RMSE': []
    })
    return results

def forecast_naive(train_data):
    """
    Implements the Naïve Forecast method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.

    Returns:
        int: Predicted number of flights.
    """
    predicted = train_data['Number_of_Flights'].iloc[-1]
    return predicted

def forecast_linear_regression(train_data, test_year):
    """
    Implements the Linear Regression forecasting method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        int: Predicted number of flights.
    """
    X_train = train_data[['Year']]
    y_train = train_data['Number_of_Flights']
    X_test = pd.DataFrame({'Year': [test_year]})
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predicted = int(lr_model.predict(X_test)[0])
    return predicted

def forecast_arima(train_data, test_year):
    """
    Implements the ARIMA(1,1,1) forecasting method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        int or np.nan: Predicted number of flights or NaN if model fails.
    """
    ts = train_data.set_index('Year')['Number_of_Flights']
    ts.index = pd.to_datetime(ts.index.astype(str) + '-06-01')
    
    try:
        model = ARIMA(ts, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted = int(round(forecast.iloc[0]))
        return predicted
    except Exception as e:
        print(f"ARIMA model failed for year {test_year}: {e}")
        return np.nan

def forecast_exponential_smoothing(train_data, test_year):
    """
    Implements the Exponential Smoothing forecasting method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        int or np.nan: Predicted number of flights or NaN if model fails.
    """
    ts = train_data.set_index('Year')['Number_of_Flights']
    ts.index = pd.to_datetime(ts.index.astype(str) + '-06-01')
    
    try:
        model = ExponentialSmoothing(ts, trend='add', seasonal=None)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted = int(round(forecast.iloc[0]))
        return predicted
    except Exception as e:
        print(f"Exponential Smoothing failed for year {test_year}: {e}")
        return np.nan

def forecast_prophet(train_data, test_year):
    """
    Implements the Facebook Prophet forecasting method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        int or np.nan: Predicted number of flights or NaN if model fails.
    """
    try:
        prophet_train = train_data.rename(columns={'Year': 'ds', 'Number_of_Flights': 'y'})
        prophet_train['ds'] = pd.to_datetime(prophet_train['ds'].astype(str) + '-06-01')
        
        model = Prophet(yearly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_train)
        
        future = pd.DataFrame({'ds': [pd.to_datetime(f'{test_year}-06-01')]})
        forecast = model.predict(future)
        predicted = int(round(forecast['yhat'].iloc[0]))
        return predicted
    except Exception as e:
        print(f"Prophet model failed for year {test_year}: {e}")
        return np.nan

def evaluate_and_append(results_df, model_name, predicted, actual):
    """
    Calculates evaluation metrics and appends the results to the results DataFrame.

    Parameters:
        results_df (pd.DataFrame): DataFrame to store results.
        model_name (str): Name of the forecasting model.
        predicted (int): Predicted number of flights.
        actual (int): Actual number of flights.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    if np.isnan(predicted):
        mae = np.nan
        rmse = np.nan
    else:
        mae = mean_absolute_error([actual], [predicted])
        rmse = mean_squared_error([actual], [predicted], squared=False)
    
    results_df = results_df.append({
        'Model': model_name,
        'Predicted': predicted,
        'Actual': actual,
        'MAE': mae,
        'RMSE': rmse
    }, ignore_index=True)
    
    return results_df

def process_forecasting_models(train_data, test_year):
    """
    Runs all forecasting models for a given test year and collects their predictions.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        pd.DataFrame: Results DataFrame for the test year.
    """
    actual_flights = train_data['Number_of_Flights'].iloc[-1]  # Adjusted to use the correct actual
    # Note: The actual flights should come from the test set, but this function assumes it's provided externally.
    # To fix, pass the actual_flights as a parameter or adjust accordingly.
    # Here, we'll skip and handle in walk_forward_validation
    
    # This function is now deprecated as individual forecasting functions handle evaluation and appending.
    pass

def walk_forward_validation(flights_per_year):
    """
    Performs walk-forward validation across all test years and collects forecasting results.

    Parameters:
        flights_per_year (pd.DataFrame): Aggregated flights per year DataFrame.

    Returns:
        pd.DataFrame: Results DataFrame containing predictions and evaluation metrics for all models.
    """
    results = initialize_results_dataframe()
    
    test_years = flights_per_year['Year'][(flights_per_year['Year'] >= 2001) & (flights_per_year['Year'] <= 2024)]
    
    for test_year in test_years:
        print(f"\nProcessing Year: {test_year}")
        
        # Split into training and testing sets
        train_data = flights_per_year[flights_per_year['Year'] < test_year]
        test_data = flights_per_year[flights_per_year['Year'] == test_year]
        actual_flights = test_data['Number_of_Flights'].values[0]
        
        # ---------------------------
        # 1. Naïve Forecast
        # ---------------------------
        predicted_naive = forecast_naive(train_data)
        results = results.append({
            'Year': test_year,
            'Model': 'Naïve Forecast',
            'Predicted': predicted_naive,
            'Actual': actual_flights,
            'MAE': mean_absolute_error([actual_flights], [predicted_naive]),
            'RMSE': mean_squared_error([actual_flights], [predicted_naive], squared=False)
        }, ignore_index=True)
        
        # ---------------------------
        # 2. Linear Regression
        # ---------------------------
        predicted_lr = forecast_linear_regression(train_data, test_year)
        mae_lr = mean_absolute_error([actual_flights], [predicted_lr])
        rmse_lr = mean_squared_error([actual_flights], [predicted_lr], squared=False)
        results = results.append({
            'Year': test_year,
            'Model': 'Linear Regression',
            'Predicted': predicted_lr,
            'Actual': actual_flights,
            'MAE': mae_lr,
            'RMSE': rmse_lr
        }, ignore_index=True)
        
        # ---------------------------
        # 3. ARIMA
        # ---------------------------
        predicted_arima = forecast_arima(train_data, test_year)
        if not np.isnan(predicted_arima):
            mae_arima = mean_absolute_error([actual_flights], [predicted_arima])
            rmse_arima = mean_squared_error([actual_flights], [predicted_arima], squared=False)
        else:
            mae_arima = np.nan
            rmse_arima = np.nan
        results = results.append({
            'Year': test_year,
            'Model': 'ARIMA(1,1,1)',
            'Predicted': predicted_arima,
            'Actual': actual_flights,
            'MAE': mae_arima,
            'RMSE': rmse_arima
        }, ignore_index=True)
        
        # ---------------------------
        # 4. Exponential Smoothing
        # ---------------------------
        predicted_hw = forecast_exponential_smoothing(train_data, test_year)
        if not np.isnan(predicted_hw):
            mae_hw = mean_absolute_error([actual_flights], [predicted_hw])
            rmse_hw = mean_squared_error([actual_flights], [predicted_hw], squared=False)
        else:
            mae_hw = np.nan
            rmse_hw = np.nan
        results = results.append({
            'Year': test_year,
            'Model': 'Exponential Smoothing',
            'Predicted': predicted_hw,
            'Actual': actual_flights,
            'MAE': mae_hw,
            'RMSE': rmse_hw
        }, ignore_index=True)
        
        # ---------------------------
        # 5. Facebook Prophet
        # ---------------------------
        predicted_prophet = forecast_prophet(train_data, test_year)
        if not np.isnan(predicted_prophet):
            mae_prophet = mean_absolute_error([actual_flights], [predicted_prophet])
            rmse_prophet = mean_squared_error([actual_flights], [predicted_prophet], squared=False)
        else:
            mae_prophet = np.nan
            rmse_prophet = np.nan
        results = results.append({
            'Year': test_year,
            'Model': 'Prophet',
            'Predicted': predicted_prophet,
            'Actual': actual_flights,
            'MAE': mae_prophet,
            'RMSE': rmse_prophet
        }, ignore_index=True)
        
    return results

def save_forecasts(results_df, output_dir, airport_name):
    """
    Saves the forecasting results to CSV files.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing forecasting results.
        output_dir (str): Directory to save the CSV files.
        airport_name (str): Name of the origin airport.

    Returns:
        None
    """
    airport_output_dir = os.path.join(output_dir, airport_name)
    os.makedirs(airport_output_dir, exist_ok=True)
    
    # Save the results DataFrame
    results_path = os.path.join(airport_output_dir, f'{airport_name}_Forecast_Results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Forecast results saved to {results_path}")

def visualize_performance(performance_summary, airport_name, output_dir):
    """
    Visualizes the performance metrics (MAE and RMSE) of all models.

    Parameters:
        performance_summary (pd.DataFrame): Summary DataFrame with average MAE and RMSE per model.
        airport_name (str): Name of the origin airport.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    airport_output_dir = os.path.join(output_dir, airport_name)
    
    # Visualization of RMSE
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='RMSE', data=performance_summary, palette='viridis')
    plt.title(f'Average RMSE of Forecasting Models (2001-2024) for {airport_name}')
    plt.xlabel('Forecasting Model')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    rmse_plot_path = os.path.join(airport_output_dir, f'{airport_name}_Average_RMSE.png')
    plt.savefig(rmse_plot_path)
    plt.close()
    print(f"RMSE plot saved to {rmse_plot_path}")
    
    # Visualization of MAE
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='MAE', data=performance_summary, palette='magma')
    plt.title(f'Average MAE of Forecasting Models (2001-2024) for {airport_name}')
    plt.xlabel('Forecasting Model')
    plt.ylabel('Average MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    mae_plot_path = os.path.join(airport_output_dir, f'{airport_name}_Average_MAE.png')
    plt.savefig(mae_plot_path)
    plt.close()
    print(f"MAE plot saved to {mae_plot_path}")

def visualize_predictions(results_df, flights_per_year, airport_name, output_dir):
    """
    Visualizes the predicted vs. actual number of flights for each model across all test years.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing forecasting results.
        flights_per_year (pd.DataFrame): Aggregated flights per year DataFrame.
        airport_name (str): Name of the origin airport.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    airport_output_dir = os.path.join(output_dir, airport_name)
    
    # Pivot the results for easier plotting
    pivot_results = results_df.pivot(index='Year', columns='Model', values='Predicted')
    pivot_results['Actual'] = flights_per_year.set_index('Year').loc[pivot_results.index, 'Number_of_Flights']
    
    # Plotting
    plt.figure(figsize=(14,8))
    for model in pivot_results.columns:
        if model != 'Actual':
            sns.lineplot(x=pivot_results.index, y=pivot_results[model], marker='o', label=model)
    
    # Plot Actuals
    sns.lineplot(x=pivot_results.index, y=pivot_results['Actual'], marker='o', color='black', label='Actual')
    
    plt.title(f'Forecasted vs Actual Number of Flights from BWI in June (2001-2024) for {airport_name}')
    plt.xlabel('Year')
    plt.ylabel('Number of Flights')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    predictions_plot_path = os.path.join(airport_output_dir, f'{airport_name}_Forecast_vs_Actual.png')
    plt.savefig(predictions_plot_path)
    plt.close()
    print(f"Predictions vs Actual plot saved to {predictions_plot_path}")

# ---------------------------
# Main Execution Block
# ---------------------------

def main():
    """
    Main function to execute the forecasting process for multiple origin airports.
    """
    for csv_file in CSV_FILES:
        # Extract airport name from the filename
        airport_name = csv_file.split('.csv')[0].replace('Origin Airport ', '').strip()
        
        print(f"\n{'='*80}\nProcessing Airport: {airport_name}\n{'='*80}")
        
        # Load data
        filepath = os.path.join(INPUT_DIR, csv_file)
        df_raw = load_data(filepath)
        if df_raw is None:
            continue  # Skip to the next file if loading failed
        
        # Clean and prepare data
        df_clean = clean_and_prepare_data(df_raw, month=FORECAST_MONTH)
        
        # Aggregate flights per year
        flights_per_year = aggregate_flights(df_clean)
        
        # Ensure complete years from 2000 to 2024
        all_years = pd.DataFrame({'Year': range(2000, 2025)})
        flights_per_year = pd.merge(all_years, flights_per_year, on='Year', how='left')
        flights_per_year['Number_of_Flights'] = flights_per_year['Number_of_Flights'].fillna(0).astype(int)
        
        # Perform walk-forward validation
        results = walk_forward_validation(flights_per_year)
        
        # Calculate performance summary
        results_clean = results.dropna()
        performance_summary = results_clean.groupby('Model').agg({
            'MAE': 'mean',
            'RMSE': 'mean'
        }).reset_index()
        
        print("\nModel Performance Summary:")
        print(performance_summary)
        
        # Save forecasting results
        save_forecasts(results, OUTPUT_DIR, airport_name)
        
        # Save performance summary
        airport_output_dir = os.path.join(OUTPUT_DIR, airport_name)
        performance_summary_path = os.path.join(airport_output_dir, f'{airport_name}_Performance_Summary.csv')
        performance_summary.to_csv(performance_summary_path, index=False)
        print(f"Performance summary saved to {performance_summary_path}")
        
        # Visualize performance metrics
        visualize_performance(performance_summary, airport_name, OUTPUT_DIR)
        
        # Visualize predictions vs actuals
        visualize_predictions(results_clean, flights_per_year, airport_name, OUTPUT_DIR)
        
    print("\nAll forecasting processes completed successfully!")

if __name__ == "__main__":
    main()
