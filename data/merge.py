# Open-Meteo Air Quality API URL
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Same location and time period as our weather data
aqi_params = {
    "latitude": 29.3759,
    "longitude": 47.9774,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "hourly": "pm10,pm2_5",
    "timezone": "Asia/Riyadh"
}

# Fetch and parse the data
aqi_response = requests.get(aqi_url, params=aqi_params)
aqi_data = aqi_response.json()
aqi_df = pd.DataFrame(aqi_data["hourly"])

# Convert time strings to actual datetime objects
aqi_df["time"] = pd.to_datetime(aqi_df["time"])
# 'inner' merge means we only keep rows where both tables have data for that exact hour
final_dataset = pd.merge(df, aqi_df, on="time", how="inner")

# Check the final merged table
print(final_dataset.head())
