import numpy as np

# 1. Redefine 'y' from the latest, fully cleaned dataset
y = final_dataset['pm10']

# 2. Identify and drop the pollution columns to create X
pollution_columns = [col for col in final_dataset.columns if 'pm' in col]
X_weather_only = final_dataset.drop(columns=pollution_columns)

# 3. Verify they are now exactly the same length
print(f"X rows: {len(X_weather_only)} | y rows: {len(y)}")

# 4. Re-split the data
X_train_w, X_test_w, y_train, y_test = train_test_split(
    X_weather_only, y, test_size=0.2, shuffle=False
)

# 5. Re-create the sample weights to match the new y_train length
sample_weights = np.where(y_train > 300, 10, 1)

# 6. Train the Pure Weather Model
weather_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    random_state=42
)

print("Training Pure Weather Model...")
weather_model.fit(
    X_train_w,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train_w, y_train), (X_test_w, y_test)],
    verbose=100
)

# 7. Save the model for your Streamlit app!
weather_model.save_model('kuwait_dust_radar.json')
print("Model successfully saved as 'kuwait_dust_radar.json'.")
