import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 1. Initialize the model
model = xgb.XGBRegressor(
    n_estimators=1000,           # Maximum number of trees to build
    learning_rate=0.1,           # How much each tree contributes
    max_depth=5,                 # Maximum complexity of each tree
    subsample=0.8,               # Use 80% of data per tree to prevent overfitting
    early_stopping_rounds=50,    # Stop if no improvement for 50 rounds
    objective='reg:squarederror',# Standard loss function for continuous regression
    random_state=42              # Lock the random seed for reproducibility
)

# 2. Train the model
print("Training XGBoost Regressor...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)], # Monitor both train and test scores
    verbose=100                                      # Print progress every 100 trees
)

# 3. Evaluate the performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nFinal Test RMSE: {rmse:.2f} µg/m³")
print(f"Final Test MAE: {mae:.2f} µg/m³")
