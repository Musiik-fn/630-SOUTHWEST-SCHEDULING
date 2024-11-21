# Function Documentation

## `load_data`

**Description:**
Loads a flight data CSV file into a pandas DataFrame.

**Parameters:**
- `filepath` (`Path`): Path to the flight data CSV file.

**Returns:**
- `pd.DataFrame` or `None`: 
  - Returns the loaded DataFrame if successful.
  - Returns `None` if the file is not found or an error occurs during loading.

---

## `clean_and_prepare_data`

**Description:**
Cleans the flight data DataFrame by handling missing values, filtering for a specific month (default is June), and retaining only specified destination airports. It also processes departure delays by replacing negative delays with zero and filling missing values with the median delay for each route.

**Parameters:**
- `df` (`pd.DataFrame`): Raw flight data DataFrame.
- `month` (`int`, optional): Month to filter data for (default is June, i.e., `6`).

**Returns:**
- `pd.DataFrame`: Cleaned DataFrame with specified destination airports and processed departure delays.

---

## `aggregate_flights`

**Description:**
Aggregates the number of flights per year for each specified group, either Destination Airport or Origin Airport.

**Parameters:**
- `master_df` (`pd.DataFrame`): Master DataFrame containing all cleaned flight data.
- `group_by` (`str`, optional): Column name to group by (`'Destination Airport'` or `'Origin Airport'`). Defaults to `'Destination Airport'`.

**Returns:**
- `pd.DataFrame`: Aggregated DataFrame showing the number of flights per year per group.

---

## `aggregate_departure_delays`

**Description:**
Aggregates the average departure delay per year for each route, defined by Origin and Destination Airport pairs.

**Parameters:**
- `master_df` (`pd.DataFrame`): Master DataFrame containing all cleaned flight data.

**Returns:**
- `pd.DataFrame`: Aggregated DataFrame with average departure delays per year per route.

---

## `initialize_results_dataframe`

**Description:**
Initializes an empty DataFrame to store forecasting results for various models across different years.

**Parameters:**
- None

**Returns:**
- `pd.DataFrame`: Empty DataFrame with predefined columns for storing results (`Year`, `Model`, `Predicted`, `Actual`, `MAE`, `RMSE`).

---

## `forecast_naive`

**Description:**
Implements the Naïve Forecast method by using the last known number of flights as the prediction for the next year.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame containing historical flight data up to the year before the forecast year.

**Returns:**
- `float`: Predicted number of flights for the forecast year.

---

## `forecast_linear_regression`

**Description:**
Implements the Linear Regression forecasting method to predict future flight volumes based on the year.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame containing historical flight data up to the year before the forecast year.
- `test_year` (`int`): Year for which to make the forecast.

**Returns:**
- `float`: Predicted number of flights for the specified forecast year.

---

## `forecast_arima`

**Description:**
Implements the ARIMA(1,1,1) forecasting method to predict future flight volumes. Converts the 'Year' column to a datetime index assuming June 1st for each year.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame containing historical flight data up to the year before the forecast year.
- `test_year` (`int`): Year for which to make the forecast.

**Returns:**
- `float` or `np.nan`: Predicted number of flights for the specified forecast year. Returns `np.nan` if the model fails.

---

## `forecast_exponential_smoothing`

**Description:**
Implements the Exponential Smoothing forecasting method to predict future flight volumes. Converts the 'Year' column to a datetime index assuming June 1st for each year.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame containing historical flight data up to the year before the forecast year.
- `test_year` (`int`): Year for which to make the forecast.

**Returns:**
- `float` or `np.nan`: Predicted number of flights for the specified forecast year. Returns `np.nan` if the model fails.

---

## `forecast_prophet`

**Description:**
Implements the Facebook Prophet forecasting method to predict future flight volumes. Prepares the DataFrame by renaming columns to 'ds' and 'y', and converts the 'Year' column to a datetime format assuming June 1st for each year.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame containing historical flight data up to the year before the forecast year.
- `test_year` (`int`): Year for which to make the forecast.

**Returns:**
- `float` or `np.nan`: Predicted number of flights for the specified forecast year. Returns `np.nan` if the model fails.

---

## `compute_performance_metrics`

**Description:**
Computes performance metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination (R²), and Adjusted R² for each forecasting model based on the cleaned results DataFrame.

**Parameters:**
- `results_clean` (`pd.DataFrame`): Cleaned results DataFrame after dropping `NaN` values.

**Returns:**
- `pd.DataFrame`: Performance summary DataFrame with average MAE, RMSE, R², and Adjusted R² for each model.

---

## `walk_forward_validation`

**Description:**
Performs walk-forward validation across all test years (2001-2024) to evaluate forecasting models for flight volumes. It iteratively trains on all data before the test year and makes predictions for the test year using different forecasting models. It then collects the predictions and evaluates them against the actual values.

**Parameters:**
- `flights_per_year` (`pd.DataFrame`): Aggregated flights per year DataFrame.

**Returns:**
- `pd.DataFrame`: DataFrame containing predictions and evaluation metrics (MAE and RMSE) for all models across all test years.

---

## `walk_forward_validation_delays`

**Description:**
Performs walk-forward validation for departure delays across all test years (2001-2024) for a specific route (Origin -> Destination). It iteratively trains on all data before the test year and makes predictions for the test year using different forecasting models. It then collects the predictions and evaluates them against the actual average departure delays.

**Parameters:**
- `flights_per_year` (`pd.DataFrame`): Aggregated departure delays per year DataFrame.

**Returns:**
- `pd.DataFrame`: DataFrame containing predictions and evaluation metrics (MAE and RMSE) for all models across all test years.

---

## `forecast_departure_delay`

**Description:**
Forecasts the average departure delay for a specific route using the chosen forecasting model. Depending on the `model_type` parameter, it calls the corresponding forecasting function.

**Parameters:**
- `train_data` (`pd.DataFrame`): Training DataFrame with historical average departure delays up to the year before the forecast year.
- `test_year` (`int`): Year for which to make the forecast.
- `model_type` (`str`, optional): Forecasting model to use (`'Naïve'`, `'Linear Regression'`, `'ARIMA'`, `'Exponential Smoothing'`, `'Prophet'`). Defaults to `'Prophet'`.

**Returns:**
- `float` or `np.nan`: Predicted average departure delay (in minutes) for the specified year. Returns `np.nan` if forecasting fails.

---

## `save_forecasts`

**Description:**
Saves the forecasting results to CSV files in the specified output directory, organized by the group name (Origin or Destination Airport).

**Parameters:**
- `results_df` (`pd.DataFrame`): DataFrame containing forecasting results.
- `output_dir` (`Path`): Directory where the CSV files will be saved.
- `group_name` (`str`): Name of the group (e.g., Origin or Destination Airport).

**Returns:**
- `None`

---

## `visualize_performance`

**Description:**
Visualizes performance metrics (MAE, RMSE, R², Adjusted R²) of all forecasting models using bar plots. Saves the plots in the specified output directory under the group name.

**Parameters:**
- `performance_summary` (`pd.DataFrame`): Summary DataFrame with average MAE, RMSE, R², and Adjusted R² per model.
- `group_name` (`str`): Name of the group (e.g., Origin or Destination Airport).
- `output_dir` (`Path`): Directory where the plots will be saved.

**Returns:**
- `None`

---

## `visualize_predictions`

**Description:**
Visualizes the predicted versus actual number of flights or departure delays for each model across all test years. It creates line plots comparing the forecasts of different models against the actual values and saves the plots in the specified output directory under the group name.

**Parameters:**
- `results_df` (`pd.DataFrame`): DataFrame containing forecasting results.
- `flights_per_year` (`pd.DataFrame`): Aggregated flights or delays per year DataFrame.
- `group_name` (`str`): Name of the group (e.g., Origin or Destination Airport).
- `output_dir` (`Path`): Directory where the plots will be saved.

**Returns:**
- `None`

---

## `main`

**Description:**
Main function to execute the entire forecasting process for both Origin and Destination airports, including flight volumes and departure delays. It orchestrates data loading, cleaning, aggregation, forecasting using various models, performance evaluation, and result visualization. It also handles saving of prediction summaries for the year 2025.

**Parameters:**
- None

**Returns:**
- `None`

---
