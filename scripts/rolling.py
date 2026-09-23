# Define the features we want to average
features_to_roll = ['wind_speed_10m', 'pm10']

# Define our window sizes (e.g., 6 hours and 12 hours)
windows = [6, 12]

for window in windows:
    for feature in features_to_roll:
        # Create a descriptive column name
        mean_col = f"{feature}_rolling_mean_{window}h"

        # Shift by 1 hour FIRST to prevent leakage, then calculate the rolling mean
        final_dataset[mean_col] = final_dataset[feature].shift(1).rolling(window=window).mean()

        # Capture volatility by adding a rolling standard deviation
        std_col = f"{feature}_rolling_std_{window}h"
        final_dataset[std_col] = final_dataset[feature].shift(1).rolling(window=window).std()
  # Drop the newly created NaN values
final_dataset = final_dataset.dropna()

print(final_dataset.head())
