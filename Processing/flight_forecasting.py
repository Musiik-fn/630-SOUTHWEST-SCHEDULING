import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Added r2_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Configuration
# ---------------------------

# Define base directory (parent of 'Processing')
BASE_DIR = Path(__file__).resolve().parent.parent

# Define input and output directories
INPUT_DIR = BASE_DIR / 'Data'
OUTPUT_DIR = BASE_DIR / 'Forecasts'

# Debugging: Print the directories to verify correctness
print(f"BASE_DIR: {BASE_DIR}")
print(f"INPUT_DIR: {INPUT_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of CSV filenames
CSV_FILES = [
    'Origin Airport Baltimore, MD BaltimoreWashington International Thurgood Marshall (BWI).csv',
    'Origin Airport Chicago, IL Chicago Midway International (MDW).csv',
    'Origin Airport Dallas, TX Dallas Love Field (DAL).csv',
    'Origin Airport Denver, CO Denver International (DEN).csv',
    'Origin Airport Las Vegas, NV Harry Reid International (LAS).csv'
]

# Full paths to CSV files
csv_paths = [INPUT_DIR / filename for filename in CSV_FILES]

# Debugging: Check existence of each file
for csv_path in csv_paths:
    print(f"Checking file: {csv_path}")
    if not csv_path.exists():
        print(f"File does NOT exist: {csv_path}")
    else:
        print(f"File exists: {csv_path}")

# Specify the month for which to perform forecasting (June)
FORECAST_MONTH = 6

# ---------------------------
# Function Definitions
# ---------------------------

def load_data(filepath):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
        filepath (Path): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath} successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def clean_and_prepare_data(df, month=6):
    """
    Cleans the DataFrame by handling missing values, filtering for a specific month,
    and retaining only specified destination airports.

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

    # Ensure 'Destination Airport' codes are uppercase
    df_clean['Destination Airport'] = df_clean['Destination Airport'].str.upper()

    # Filter the DataFrame to retain only the specified destination airports
    df_clean = df_clean[df_clean['Destination Airport'].isin(allowed_destinations)].copy()

    # Handle missing or anomalous 'Departure delay (Minutes)'
    # For simplicity, replace negative delays (early departures) with zero
    df_clean['Departure delay (Minutes)'] = df_clean['Departure delay (Minutes)'].apply(lambda x: x if x >= 0 else 0)

    # Fill missing departure delays with the median delay of the route
    df_clean['Departure delay (Minutes)'] = df_clean.groupby(['Origin Airport', 'Destination Airport'])['Departure delay (Minutes)'].transform(lambda x: x.fillna(x.median()))

    # (Optional) Reset index after filtering for cleanliness
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean


def aggregate_flights(master_df, group_by='Destination Airport'):
    """
    Aggregates the number of flights per year for each specified group.

    Parameters:
        master_df (pd.DataFrame): Master DataFrame containing all flight data.
        group_by (str): Column name to group by ('Destination Airport' or 'Origin Airport').

    Returns:
        pd.DataFrame: Aggregated flights per year per group.
    """
    flights_per_year = master_df.groupby(['Year', group_by]).size().reset_index(name='Number_of_Flights')
    return flights_per_year


def aggregate_departure_delays(master_df):
    """
    Aggregates the average departure delay per year for each route (origin-destination pair).

    Parameters:
        master_df (pd.DataFrame): Master DataFrame containing all flight data.

    Returns:
        pd.DataFrame: Aggregated average departure delays per year per route.
    """
    # Group by Origin, Destination, and Year, then calculate the mean departure delay
    avg_delays = master_df.groupby(['Origin Airport', 'Destination Airport', 'Year'])['Departure delay (Minutes)'].mean().reset_index()

    # Rename the column for clarity
    avg_delays.rename(columns={'Departure delay (Minutes)': 'Avg_Departure_Delay'}, inplace=True)

    return avg_delays


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
        'RMSE': []
    })
    return results


def forecast_naive(train_data):
    """
    Implements the Naïve Forecast method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.

    Returns:
        float: Predicted value.
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
        float: Predicted value.
    """
    X_train = train_data[['Year']]
    y_train = train_data['Number_of_Flights']
    X_test = pd.DataFrame({'Year': [test_year]})

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predicted = lr_model.predict(X_test)[0]
    return predicted


def forecast_arima(train_data, test_year):
    """
    Implements the ARIMA(1,1,1) forecasting method.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame.
        test_year (int): Year to forecast.

    Returns:
        float or np.nan: Predicted value or NaN if model fails.
    """
    ts = train_data.set_index('Year')['Number_of_Flights']
    ts.index = pd.to_datetime(ts.index.astype(str) + '-06-01')

    try:
        model = ARIMA(ts, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted = forecast.iloc[0]
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
        float or np.nan: Predicted value or NaN if model fails.
    """
    ts = train_data.set_index('Year')['Number_of_Flights']
    ts.index = pd.to_datetime(ts.index.astype(str) + '-06-01')

    try:
        model = ExponentialSmoothing(ts, trend='add', seasonal=None)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted = forecast.iloc[0]
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
        float or np.nan: Predicted value or NaN if model fails.
    """
    try:
        prophet_train = train_data.rename(columns={'Year': 'ds', 'Number_of_Flights': 'y'})
        prophet_train['ds'] = pd.to_datetime(prophet_train['ds'].astype(str) + '-06-01')

        model = Prophet(yearly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_train)

        future = pd.DataFrame({'ds': [pd.to_datetime(f'{test_year}-06-01')]})
        forecast = model.predict(future)
        predicted = forecast['yhat'].iloc[0]
        return predicted
    except Exception as e:
        print(f"Prophet model failed for year {test_year}: {e}")
        return np.nan


def compute_performance_metrics(results_clean):
    """
    Computes , RMSE, R2, and Adjusted R2 for each model.

    Parameters:
        results_clean (pd.DataFrame): Cleaned results DataFrame after dropping NaNs.

    Returns:
        pd.DataFrame: Performance summary with metrics.
    """
    performance_summary = results_clean.groupby('Model').agg({
        'RMSE': 'mean'
    }).reset_index()

    # Compute R² and Adjusted R² per model
    performance_summary['R2'] = np.nan
    performance_summary['Adjusted_R2'] = np.nan

    for idx, row in performance_summary.iterrows():
        model = row['Model']
        model_results = results_clean[results_clean['Model'] == model]
        y_true = model_results['Actual']
        y_pred = model_results['Predicted']

        r2 = r2_score(y_true, y_pred)
        performance_summary.at[idx, 'R2'] = r2

        # Adjusted R² only for models with predictors
        if model == 'Linear Regression':
            p = 1  # Number of predictors
            n = len(y_true)
            if n > p + 1:
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                performance_summary.at[idx, 'Adjusted_R2'] = adjusted_r2
        else:
            performance_summary.at[idx, 'Adjusted_R2'] = np.nan

        print(f"Model: {model}, R²: {r2}, Adjusted R²: {performance_summary.at[idx, 'Adjusted_R2']}")

    return performance_summary


def walk_forward_validation(flights_per_year):
    """
    Performs walk-forward validation across all test years and collects forecasting results.

    Parameters:
        flights_per_year (pd.DataFrame): Aggregated flights or delays per year per group.

    Returns:
        pd.DataFrame: Results DataFrame containing predictions and evaluation metrics for all models.
    """
    results_list = []

    # Define test years (assuming 2001-2024 based on data availability)
    test_years = sorted(flights_per_year['Year'][(flights_per_year['Year'] >= 2001) & (flights_per_year['Year'] <= 2024)].unique())

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
        rmse_naive = mean_squared_error([actual_flights], [predicted_naive], squared=False)
        results_list.append({
            'Year': test_year,
            'Model': 'Naïve Forecast',
            'Predicted': predicted_naive,
            'Actual': actual_flights,
            'RMSE': rmse_naive
        })

        # ---------------------------
        # 2. Linear Regression
        # ---------------------------
        predicted_lr = forecast_linear_regression(train_data, test_year)
        rmse_lr = mean_squared_error([actual_flights], [predicted_lr], squared=False)
        results_list.append({
            'Year': test_year,
            'Model': 'Linear Regression',
            'Predicted': predicted_lr,
            'Actual': actual_flights,
            'RMSE': rmse_lr
        })

        # ---------------------------
        # 3. ARIMA
        # ---------------------------
        predicted_arima = forecast_arima(train_data, test_year)
        if not np.isnan(predicted_arima):
            rmse_arima = mean_squared_error([actual_flights], [predicted_arima], squared=False)
        else:
            rmse_arima = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'ARIMA(1,1,1)',
            'Predicted': predicted_arima,
            'Actual': actual_flights,
            'RMSE': rmse_arima
        })

        # ---------------------------
        # 4. Exponential Smoothing
        # ---------------------------
        predicted_hw = forecast_exponential_smoothing(train_data, test_year)
        if not np.isnan(predicted_hw):
            rmse_hw = mean_squared_error([actual_flights], [predicted_hw], squared=False)
        else:
            rmse_hw = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Exponential Smoothing',
            'Predicted': predicted_hw,
            'Actual': actual_flights,
            'RMSE': rmse_hw
        })

        # ---------------------------
        # 5. Facebook Prophet
        # ---------------------------
        predicted_prophet = forecast_prophet(train_data, test_year)
        if not np.isnan(predicted_prophet):
            rmse_prophet = mean_squared_error([actual_flights], [predicted_prophet], squared=False)
        else:
            rmse_prophet = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Prophet',
            'Predicted': predicted_prophet,
            'Actual': actual_flights,
            'RMSE': rmse_prophet
        })

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df


def walk_forward_validation_delays(flights_per_year):
    """
    Performs walk-forward validation for departure delays across all test years for a specific group.

    Parameters:
        flights_per_year (pd.DataFrame): Aggregated flights or delays per year per group.

    Returns:
        pd.DataFrame: Results DataFrame containing predictions and evaluation metrics for all models.
    """
    results_list = []

    # Define test years (assuming 2001-2024 based on data availability)
    test_years = sorted(flights_per_year['Year'][(flights_per_year['Year'] >= 2001) & (flights_per_year['Year'] <= 2024)].unique())

    for test_year in test_years:
        print(f"\nProcessing Year: {test_year}")

        # Split into training and testing sets
        train_data = flights_per_year[flights_per_year['Year'] < test_year]
        test_data = flights_per_year[flights_per_year['Year'] == test_year]
        actual_delay = test_data['Avg_Departure_Delay'].values[0]

        # ---------------------------
        # 1. Naïve Forecast
        # ---------------------------
        predicted_naive = forecast_departure_delay(train_data, test_year, model_type='Naïve')
        rmse_naive = mean_squared_error([actual_delay], [predicted_naive], squared=False) if not np.isnan(predicted_naive) else np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Naïve Forecast',
            'Predicted': predicted_naive,
            'Actual': actual_delay,
            'RMSE': rmse_naive
        })

        # ---------------------------
        # 2. Linear Regression
        # ---------------------------
        predicted_lr = forecast_departure_delay(train_data, test_year, model_type='Linear Regression')
        rmse_lr = mean_squared_error([actual_delay], [predicted_lr], squared=False) if not np.isnan(predicted_lr) else np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Linear Regression',
            'Predicted': predicted_lr,
            'Actual': actual_delay,
            'RMSE': rmse_lr
        })

        # ---------------------------
        # 3. ARIMA
        # ---------------------------
        predicted_arima = forecast_departure_delay(train_data, test_year, model_type='ARIMA')
        if not np.isnan(predicted_arima):
            rmse_arima = mean_squared_error([actual_delay], [predicted_arima], squared=False)
        else:
            rmse_arima = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'ARIMA(1,1,1)',
            'Predicted': predicted_arima,
            'Actual': actual_delay,
            'RMSE': rmse_arima
        })

        # ---------------------------
        # 4. Exponential Smoothing
        # ---------------------------
        predicted_hw = forecast_departure_delay(train_data, test_year, model_type='Exponential Smoothing')
        if not np.isnan(predicted_hw):
            rmse_hw = mean_squared_error([actual_delay], [predicted_hw], squared=False)
        else:
            rmse_hw = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Exponential Smoothing',
            'Predicted': predicted_hw,
            'Actual': actual_delay,
            'RMSE': rmse_hw
        })

        # ---------------------------
        # 5. Facebook Prophet
        # ---------------------------
        predicted_prophet = forecast_departure_delay(train_data, test_year, model_type='Prophet')
        if not np.isnan(predicted_prophet):
            rmse_prophet = mean_squared_error([actual_delay], [predicted_prophet], squared=False)
        else:
            rmse_prophet = np.nan
        results_list.append({
            'Year': test_year,
            'Model': 'Prophet',
            'Predicted': predicted_prophet,
            'Actual': actual_delay,
            'RMSE': rmse_prophet
        })

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df


def forecast_departure_delay(train_data, test_year, model_type='Prophet'):
    """
    Forecasts the average departure delay for a specific route using the chosen model.

    Parameters:
        train_data (pd.DataFrame): Training DataFrame with 'Year' and 'Avg_Departure_Delay'.
        test_year (int): Year to forecast.
        model_type (str): Forecasting model to use ('Naïve', 'Linear Regression', 'ARIMA', 'Exponential Smoothing', 'Prophet').

    Returns:
        float or np.nan: Predicted average departure delay for the test year or NaN if forecasting fails.
    """
    # Rename columns to match forecasting functions
    renamed_train = train_data.rename(columns={'Avg_Departure_Delay': 'Number_of_Flights'})

    if model_type == 'Naïve':
        return forecast_naive(renamed_train)
    elif model_type == 'Linear Regression':
        return forecast_linear_regression(renamed_train, test_year)
    elif model_type == 'ARIMA':
        return forecast_arima(renamed_train, test_year)
    elif model_type == 'Exponential Smoothing':
        return forecast_exponential_smoothing(renamed_train, test_year)
    elif model_type == 'Prophet':
        return forecast_prophet(renamed_train, test_year)
    else:
        print(f"Unknown model type: {model_type}")
        return np.nan


def save_forecasts(results_df, output_dir, group_name):
    """
    Saves the forecasting results to CSV files.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing forecasting results.
        output_dir (Path): Directory to save the CSV files.
        group_name (str): Name of the group (Origin or Destination Airport).

    Returns:
        None
    """
    group_output_dir = output_dir / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the results DataFrame
    results_path = group_output_dir / f'{group_name}_Forecast_Results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Forecast results saved to {results_path}")


def visualize_performance(performance_summary, group_name, output_dir):
    """
    Visualizes the performance metrics (, RMSE, R2, Adjusted R2) of all models.

    Parameters:
        performance_summary (pd.DataFrame): Summary DataFrame with average , RMSE, R2, and Adjusted R2 per model.
        group_name (str): Name of the group (Origin or Destination Airport).
        output_dir (Path): Directory to save the plots.

    Returns:
        None
    """
    group_output_dir = output_dir / group_name

    # Visualization of RMSE
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='RMSE', data=performance_summary, palette='viridis')
    plt.title(f'Average RMSE of Forecasting Models (2001-2024) for {group_name}')
    plt.xlabel('Forecasting Model')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    rmse_plot_path = group_output_dir / f'{group_name}_Average_RMSE.png'
    plt.savefig(rmse_plot_path)
    plt.close()
    print(f"RMSE plot saved to {rmse_plot_path}")

    # Visualization of R2
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='R2', data=performance_summary, palette='coolwarm')
    plt.title(f'R² of Forecasting Models (2001-2024) for {group_name}')
    plt.xlabel('Forecasting Model')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    r2_plot_path = group_output_dir / f'{group_name}_R2.png'
    plt.savefig(r2_plot_path)
    plt.close()
    print(f"R² plot saved to {r2_plot_path}")

    # Visualization of Adjusted R2
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='Adjusted_R2', data=performance_summary, palette='coolwarm')
    plt.title(f'Adjusted R² of Forecasting Models (2001-2024) for {group_name}')
    plt.xlabel('Forecasting Model')
    plt.ylabel('Adjusted R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    adj_r2_plot_path = group_output_dir / f'{group_name}_Adjusted_R2.png'
    plt.savefig(adj_r2_plot_path)
    plt.close()
    print(f"Adjusted R² plot saved to {adj_r2_plot_path}")


def visualize_predictions(results_df, flights_per_year, group_name, output_dir):
    """
    Visualizes the predicted vs. actual number of flights or departure delays for each model across all test years.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing forecasting results.
        flights_per_year (pd.DataFrame): Aggregated flights or delays per year DataFrame.
        group_name (str): Name of the group (Origin or Destination Airport).
        output_dir (Path): Directory to save the plots.

    Returns:
        None
    """
    group_output_dir = output_dir / group_name

    # Pivot the results for easier plotting
    pivot_results = results_df.pivot(index='Year', columns='Model', values='Predicted')
    pivot_results['Actual'] = flights_per_year.set_index('Year').loc[pivot_results.index, 'Number_of_Flights']

    # Plotting
    plt.figure(figsize=(14,8))

    # Plot Actuals
    plt.plot(pivot_results.index, pivot_results['Actual'], label='Actual', marker='o', color='black', linewidth=2)

    # Plot Predictions for each model
    models = [col for col in pivot_results.columns if col != 'Actual']
    for model in models:
        plt.plot(pivot_results.index, pivot_results[model], label=model, marker='o', linewidth=1)

    plt.title(f'Forecasted vs Actual Number of Flights in June for {group_name}')
    plt.xlabel('Year')
    plt.ylabel('Number of Flights')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    predictions_plot_path = group_output_dir / f'{group_name}_Forecast_vs_Actual.png'
    plt.savefig(predictions_plot_path)
    plt.close()
    print(f"Forecast vs Actual plot saved to {predictions_plot_path}")


def main():
    """
    Main function to execute the forecasting process for both Origin and Destination airports,
    including flight volumes and departure delays.
    """
    # Initialize a master DataFrame to hold all flight data
    master_df = pd.DataFrame()

    # Iterate through each CSV file (origin airport) and perform data loading and cleaning
    for csv_file in CSV_FILES:
        # Extract origin airport name from the filename
        origin_airport = csv_file.split('.csv')[0].replace('Origin Airport ', '').strip()

        print(f"\n{'='*80}\nProcessing Origin Airport: {origin_airport}\n{'='*80}")

        # Load data
        filepath = INPUT_DIR / csv_file
        df_raw = load_data(filepath)
        if df_raw is None:
            print(f"Skipping {origin_airport} due to loading issues.")
            continue  # Skip to the next file if loading failed

        # **Assign 'Origin Airport' before cleaning**
        df_raw['Origin Airport'] = origin_airport

        # Clean and prepare data
        df_clean = clean_and_prepare_data(df_raw, month=FORECAST_MONTH)

        # Append to master DataFrame
        master_df = pd.concat([master_df, df_clean], ignore_index=True)

        # Optional: Display first few rows
        print("\nFirst few rows of the cleaned DataFrame:")
        print(df_clean.head())

    # ---------------------------
    # Forecasting for Destination Airports (Flight Volumes)
    # ---------------------------
    print(f"\n{'='*80}\nForecasting for Destination Airports (Flight Volumes)\n{'='*80}")

    # Aggregate flights per Destination Airport per Year across all origins
    aggregated_dest_df = aggregate_flights(master_df, group_by='Destination Airport')

    # Define complete years
    all_years = pd.DataFrame({'Year': range(2000, 2025)})

    # Get unique destination airports
    destinations = aggregated_dest_df['Destination Airport'].unique()

    # Initialize summary list for destinations
    destination_summary = []

    # Iterate over each destination airport for forecasting flight volumes
    for destination in destinations:
        print(f"\n{'='*80}\nForecasting Flight Volumes for Destination Airport: {destination}\n{'='*80}")

        # Filter data for the current destination
        df_dest = aggregated_dest_df[aggregated_dest_df['Destination Airport'] == destination].copy()

        # Merge with all years to ensure continuity
        df_dest = pd.merge(all_years, df_dest, on='Year', how='left')
        df_dest['Number_of_Flights'] = df_dest['Number_of_Flights'].fillna(0).astype(int)

        print("\nFlights Per Year:")
        print(df_dest)

        # Perform walk-forward validation for flight volumes
        results = walk_forward_validation(df_dest)

        # Display forecasting results
        print("\nForecasting Results:")
        print(results)

        # Calculate performance summary
        results_clean = results.dropna()
        performance_summary = compute_performance_metrics(results_clean)

        print("\nModel Performance Summary:")
        print(performance_summary)

        # Identify the best model based on RMSE
        if not performance_summary.empty:
            best_model_row = performance_summary.loc[performance_summary['RMSE'].idxmin()]
            best_model = best_model_row['Model']
            best_rmse = best_model_row['RMSE']

            print(f"\nBest Model for {destination}: {best_model} with RMSE = {best_rmse:.2f}")

            # Forecasting for 2025 Using the Best Model
            print(f"\nForecasting 2025 Flight Volumes for {destination} using {best_model}...\n")

            # Prepare data for forecasting 2025
            train_data_2025 = df_dest.copy()

            # Depending on the model, fit and forecast accordingly
            if best_model == 'Naïve Forecast':
                # For Naïve Forecast, prediction is the last known value
                prediction_2025 = forecast_naive(train_data_2025)
            elif best_model == 'Linear Regression':
                prediction_2025 = forecast_linear_regression(train_data_2025, test_year=2025)
            elif best_model == 'ARIMA(1,1,1)':
                prediction_2025 = forecast_arima(train_data_2025, test_year=2025)
            elif best_model == 'Exponential Smoothing':
                prediction_2025 = forecast_exponential_smoothing(train_data_2025, test_year=2025)
            elif best_model == 'Prophet':
                prediction_2025 = forecast_prophet(train_data_2025, test_year=2025)
            else:
                print(f"Unknown model: {best_model}. Cannot forecast 2025.")
                prediction_2025 = np.nan

            # Handle cases where prediction could not be made
            if np.isnan(prediction_2025):
                print(f"Prediction for 2025 could not be made using {best_model}.")
                prediction_2025_display = "Prediction Failed"
            else:
                prediction_2025_display = int(round(prediction_2025))

            # Append the prediction to the summary table
            destination_summary.append({
                'Destination Airport': destination,
                '2025 Predicted Flights': prediction_2025_display,
                'Model Used': best_model
            })

            print(f"2025 Predicted Flights for {destination}: {prediction_2025_display}")

            # Save forecasting results
            save_forecasts(results, OUTPUT_DIR, destination)

            # Save performance summary
            performance_summary_path = OUTPUT_DIR / destination / f'{destination}_Performance_Summary.csv'
            performance_summary.to_csv(performance_summary_path, index=False)
            print(f"Performance summary saved to {performance_summary_path}")

            # Visualize performance metrics
            visualize_performance(performance_summary, destination, OUTPUT_DIR)

            # Visualize predictions vs actuals
            visualize_predictions(results_clean, df_dest, destination, OUTPUT_DIR)
        else:
            print(f"No valid performance metrics available for {destination}. Skipping forecasting.")

        print("\n" + "="*80 + "\n")

    # ---------------------------
    # Forecasting for Origin Airports (Flight Volumes)
    # ---------------------------
    print(f"\n{'='*80}\nForecasting for Origin Airports (Flight Volumes)\n{'='*80}")

    # Aggregate flights per Origin Airport per Year across all destinations
    aggregated_origin_df = aggregate_flights(master_df, group_by='Origin Airport')

    # Get unique origin airports
    origins = aggregated_origin_df['Origin Airport'].unique()

    # Initialize summary list for origins
    origin_summary = []

    # Iterate over each origin airport for forecasting flight volumes
    for origin in origins:
        print(f"\n{'='*80}\nForecasting Flight Volumes for Origin Airport: {origin}\n{'='*80}")

        # Filter data for the current origin
        df_origin = aggregated_origin_df[aggregated_origin_df['Origin Airport'] == origin].copy()

        # Merge with all years to ensure continuity
        df_origin = pd.merge(all_years, df_origin, on='Year', how='left')
        df_origin['Number_of_Flights'] = df_origin['Number_of_Flights'].fillna(0).astype(int)

        print("\nFlights Per Year:")
        print(df_origin)

        # Perform walk-forward validation for flight volumes
        results = walk_forward_validation(df_origin)

        # Display forecasting results
        print("\nForecasting Results:")
        print(results)

        # Calculate performance summary
        results_clean = results.dropna()
        performance_summary = compute_performance_metrics(results_clean)

        print("\nModel Performance Summary:")
        print(performance_summary)

        # Identify the best model based on RMSE
        if not performance_summary.empty:
            best_model_row = performance_summary.loc[performance_summary['RMSE'].idxmin()]
            best_model = best_model_row['Model']
            best_rmse = best_model_row['RMSE']

            print(f"\nBest Model for {origin}: {best_model} with RMSE = {best_rmse:.2f}")

            # Forecasting for 2025 Using the Best Model
            print(f"\nForecasting 2025 Flight Volumes for {origin} using {best_model}...\n")

            # Prepare data for forecasting 2025
            train_data_2025 = df_origin.copy()

            # Depending on the model, fit and forecast accordingly
            if best_model == 'Naïve Forecast':
                # For Naïve Forecast, prediction is the last known value
                prediction_2025 = forecast_naive(train_data_2025)
            elif best_model == 'Linear Regression':
                prediction_2025 = forecast_linear_regression(train_data_2025, test_year=2025)
            elif best_model == 'ARIMA(1,1,1)':
                prediction_2025 = forecast_arima(train_data_2025, test_year=2025)
            elif best_model == 'Exponential Smoothing':
                prediction_2025 = forecast_exponential_smoothing(train_data_2025, test_year=2025)
            elif best_model == 'Prophet':
                prediction_2025 = forecast_prophet(train_data_2025, test_year=2025)
            else:
                print(f"Unknown model: {best_model}. Cannot forecast 2025.")
                prediction_2025 = np.nan

            # Handle cases where prediction could not be made
            if np.isnan(prediction_2025):
                print(f"Prediction for 2025 could not be made using {best_model}.")
                prediction_2025_display = "Prediction Failed"
            else:
                prediction_2025_display = int(round(prediction_2025))

            # Append the prediction to the summary table
            origin_summary.append({
                'Origin Airport': origin,
                '2025 Predicted Flights': prediction_2025_display,
                'Model Used': best_model
            })

            print(f"2025 Predicted Flights for {origin}: {prediction_2025_display}")

            # Save forecasting results
            save_forecasts(results, OUTPUT_DIR, origin)

            # Save performance summary
            performance_summary_path = OUTPUT_DIR / origin / f'{origin}_Performance_Summary.csv'
            performance_summary.to_csv(performance_summary_path, index=False)
            print(f"Performance summary saved to {performance_summary_path}")

            # Visualize performance metrics
            visualize_performance(performance_summary, origin, OUTPUT_DIR)

            # Visualize predictions vs actuals
            visualize_predictions(results_clean, df_origin, origin, OUTPUT_DIR)
        else:
            print(f"No valid performance metrics available for {origin}. Skipping forecasting.")

        print("\n" + "="*80 + "\n")

    # ---------------------------
    # Forecasting for Departure Delays
    # ---------------------------
    print(f"\n{'='*80}\nForecasting Departure Delays for Each Route\n{'='*80}")

    # Aggregate average departure delays per route per year
    aggregated_delays_df = aggregate_departure_delays(master_df)

    # Get unique routes
    routes = aggregated_delays_df[['Origin Airport', 'Destination Airport']].drop_duplicates()

    # Initialize summary list for departure delays
    delays_summary = []

    # Iterate over each route for forecasting departure delays
    for index, route in routes.iterrows():
        origin = route['Origin Airport']
        destination = route['Destination Airport']
        print(f"\n{'='*80}\nForecasting Departure Delay for Route: {origin} -> {destination}\n{'='*80}")

        # Filter data for the current route
        df_route = aggregated_delays_df[
            (aggregated_delays_df['Origin Airport'] == origin) & 
            (aggregated_delays_df['Destination Airport'] == destination)
        ].copy()

        # Merge with all years to ensure continuity
        df_route = pd.merge(all_years, df_route, on='Year', how='left')
        df_route['Avg_Departure_Delay'] = df_route['Avg_Departure_Delay'].fillna(method='ffill').fillna(method='bfill')

        print("\nAverage Departure Delay Per Year:")
        print(df_route)

        # Perform walk-forward validation for departure delays
        results = walk_forward_validation_delays(df_route)

        # Display forecasting results
        print("\nForecasting Results:")
        print(results)

        # Calculate performance summary
        results_clean = results.dropna()
        if results_clean.empty:
            print("No valid predictions available for performance summary.")
            continue
        performance_summary = compute_performance_metrics(results_clean)

        print("\nModel Performance Summary:")
        print(performance_summary)

        # Identify the best model based on RMSE
        if not performance_summary.empty:
            best_model_row = performance_summary.loc[performance_summary['RMSE'].idxmin()]
            best_model = best_model_row['Model']
            best_rmse = best_model_row['RMSE']

            print(f"\nBest Model for Route {origin} -> {destination}: {best_model} with RMSE = {best_rmse:.2f}")

            # Forecasting for 2025 Using the Best Model
            print(f"\nForecasting 2025 Departure Delay for Route {origin} -> {destination} using {best_model}...\n")

            # Prepare data for forecasting 2025
            train_data_2025 = df_route.copy()

            # Depending on the model, fit and forecast accordingly
            if best_model == 'Naïve Forecast':
                # For Naïve Forecast, prediction is the last known value
                prediction_2025 = forecast_departure_delay(train_data_2025, test_year=2025, model_type='Naïve')
            elif best_model == 'Linear Regression':
                prediction_2025 = forecast_departure_delay(train_data_2025, test_year=2025, model_type='Linear Regression')
            elif best_model == 'ARIMA(1,1,1)':
                prediction_2025 = forecast_departure_delay(train_data_2025, test_year=2025, model_type='ARIMA')
            elif best_model == 'Exponential Smoothing':
                prediction_2025 = forecast_departure_delay(train_data_2025, test_year=2025, model_type='Exponential Smoothing')
            elif best_model == 'Prophet':
                prediction_2025 = forecast_departure_delay(train_data_2025, test_year=2025, model_type='Prophet')
            else:
                print(f"Unknown model: {best_model}. Cannot forecast 2025.")
                prediction_2025 = np.nan

            # Handle cases where prediction could not be made
            if np.isnan(prediction_2025):
                print(f"Prediction for 2025 could not be made using {best_model}.")
                prediction_2025_display = "Prediction Failed"
            else:
                prediction_2025_display = round(prediction_2025, 2)  # Rounded to 2 decimal places

            # Append the prediction to the summary table
            delays_summary.append({
                'Origin Airport': origin,
                'Destination Airport': destination,
                '2025 Predicted Avg Departure Delay (Minutes)': prediction_2025_display,
                'Model Used': best_model
            })

            print(f"2025 Predicted Average Departure Delay for Route {origin} -> {destination}: {prediction_2025_display} Minutes")

            # Save forecasting results
            # Uncomment the lines below if you want to save departure delay forecasts
            # save_forecasts(results, OUTPUT_DIR, f"{origin}_{destination}_Delay")

            # Save performance summary
            # Uncomment the lines below if you want to save departure delay performance summaries
            # performance_summary_path = OUTPUT_DIR / f"{origin}_{destination}_Performance_Summary.csv"
            # performance_summary.to_csv(performance_summary_path, index=False)
            # print(f"Performance summary saved to {performance_summary_path}")

            # Visualize predictions vs actuals
            # Uncomment the lines below if you want to visualize departure delay forecasts
            # visualize_predictions(results_clean, df_route, f"{origin}_{destination}_Delay", OUTPUT_DIR)
        else:
            print(f"No valid performance metrics available for Route {origin} -> {destination}. Skipping forecasting.")

        print("\n" + "="*80 + "\n")

    # ---------------------------
    # Save 2025 Prediction Summary Tables
    # ---------------------------

    # After both forecasting loops, save the summary tables
    # Convert summary lists to DataFrames
    destination_summary_df = pd.DataFrame(destination_summary)
    origin_summary_df = pd.DataFrame(origin_summary)
    delays_summary_df = pd.DataFrame(delays_summary)

    # Save Destination 2025 Prediction Table
    destination_summary_path = OUTPUT_DIR / 'Destination_Airports_2025_Predictions.csv'
    destination_summary_df.to_csv(destination_summary_path, index=False)
    print(f"Destination 2025 Prediction Table saved to {destination_summary_path}")

    # Save Origin 2025 Prediction Table
    origin_summary_path = OUTPUT_DIR / 'Origin_Airports_2025_Predictions.csv'
    origin_summary_df.to_csv(origin_summary_path, index=False)
    print(f"Origin 2025 Prediction Table saved to {origin_summary_path}")

    # Save Departure Delays 2025 Prediction Table
    delays_summary_path = OUTPUT_DIR / 'Route_Departure_Delays_2025_Predictions.csv'
    delays_summary_df.to_csv(delays_summary_path, index=False)
    print(f"Route Departure Delays 2025 Prediction Table saved to {delays_summary_path}")


if __name__ == "__main__":
    main()
