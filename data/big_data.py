# The features we want the model to look back at
features_to_lag = ['wind_speed_10m', 'wind_direction_10m', 'relative_humidity_2m', 'temperature_2m']

# How many hours into the past we want to look
lag_hours = [1, 2, 3]

for hour in lag_hours:
    for feature in features_to_lag:
        # Create a descriptive column name, e.g., 'wind_speed_10m_lag_3'
        col_name = f"{feature}_lag_{hour}"

        # Shift the data down by the specified number of hours
        final_dataset[col_name] = final_dataset[feature].shift(hour)
aqi_features = ['pm10', 'pm2_5']

for hour in lag_hours:
    for feature in aqi_features:
        col_name = f"{feature}_lag_{hour}"
        final_dataset[col_name] = final_dataset[feature].shift(hour)
# Drop the rows at the very top that are now missing data
final_dataset = final_dataset.dropna()

# Take a look at the massive new dataset we've built!
print(final_dataset.head())
