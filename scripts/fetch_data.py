# Open-Meteo Historical Weather API URL
url = "https://archive-api.open-meteo.com/v1/archive"

# Set parameters for Kuwait
params = {
    "latitude": 29.3759,
    "longitude": 47.9774,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31", # Let's pull one full year
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
    "timezone": "Asia/Riyadh" # Open-Meteo uses Riyadh for the GMT+3 timezone
}
# Fetch the data from the internet
response = requests.get(url, params=params)
data = response.json()
# Extract just the hourly data
hourly_data = data["hourly"]

# Convert it into a clean table (DataFrame)
df = pd.DataFrame(hourly_data)

# Convert the time column into actual timestamp objects
df["time"] = pd.to_datetime(df["time"])

# Let's see the first 5 rows!
print(df.head())
