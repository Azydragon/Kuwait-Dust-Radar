# Extract month (1-12)
final_dataset['month'] = final_dataset['time'].dt.month

# Extract hour of the day (0-23)
final_dataset['hour'] = final_dataset['time'].dt.hour

# Extract the day of the year (1-365) to capture broad seasonal trends
final_dataset['day_of_year'] = final_dataset['time'].dt.dayofyear
# Drop the 'time' column (axis=1 means we are dropping a column, not a row)
final_dataset = final_dataset.drop('time', axis=1)
print(final_dataset.head())
