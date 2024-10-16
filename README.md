# Ex.No: 6               HOLT WINTERS METHOD
# Developed by:Subashini S
# Reg no:212222240106
### Date: 



### AIM:
To create and implement Holt Winter's Method Model using python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import math

# Step 2: Load the dataset (weather data you uploaded)
file_path = '/content/weather_classification_data.csv'
data = pd.read_csv(file_path)

# Step 3: Add a synthetic 'date' column (assuming daily data)
# Creating a date range that matches the length of the dataset
data['date'] = pd.date_range(start='2022-01-01', periods=len(data), freq='D')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# For time series analysis, let's take the 'Temperature' column as the target (similar to sales)
data['target'] = data['Temperature']

# Group the data by date and resample it to a monthly frequency (beginning of the month)
monthly_data = data['target'].resample('MS').mean()  # 'MS' means month start

# Step 4: Plot the monthly data
plt.figure(figsize=(10,6))
plt.plot(monthly_data, label='Monthly Data')
plt.title('Monthly Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Step 5: Time series decomposition to visualize the trend, seasonality, and residuals
decomposition = seasonal_decompose(monthly_data, model='additive')
decomposition.plot()
plt.show()

# Step 6: Split the data into training and testing sets for model validation
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data[:train_size], monthly_data[train_size:]

# Step 7: Fit a Holt-Winters model to the training data and make predictions for the test period
holt_winters_model = ExponentialSmoothing(train, seasonal='additive', trend='additive', seasonal_periods=12).fit()

# Test predictions
test_predictions = holt_winters_model.forecast(len(test))

# Step 8: Calculate RMSE for the test set
rmse = math.sqrt(mean_squared_error(test, test_predictions))
print(f"Test RMSE: {rmse:.2f}")

# Final predictions: Forecast for future periods (let's say 12 months after the test set)
forecast_periods = 12  # Predict for 12 future months beyond the test set
final_predictions = holt_winters_model.forecast(len(test) + forecast_periods)[-forecast_periods:]

# Step 9: Plot the original data, test predictions, and final predictions
plt.figure(figsize=(10,6))
plt.plot(monthly_data, label='Original Data')
plt.plot(test_predictions.index, test_predictions, label='Test Predictions', color='orange')
plt.plot(final_predictions.index, final_predictions, label='Final Predictions', color='red')
plt.title('Holt-Winters Test and Final Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Display Test Predictions and Final Predictions
print("\nTest Predictions:")
print(test_predictions)

print("\nFinal Predictions (Future 12 months):")
print(final_predictions)

# Step 10: Calculate the mean and standard deviation of the entire dataset
mean_value = np.mean(monthly_data)
std_value = np.std(monthly_data)
print(f"Mean: {mean_value:.2f}, Standard Deviation: {std_value:.2f}")

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/8077f1a7-89ae-4aae-87c4-9bfdd18a1ea7)
![image](https://github.com/user-attachments/assets/a666358b-c6b6-44e2-9b5d-e2940c115944)





### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
