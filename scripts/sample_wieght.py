import numpy as np

# Apply a heavy multiplier (10x) to any row where PM10 is greater than 300 µg/m³
# Otherwise, keep the standard weight of 1
sample_weights = np.where(y_train > 300, 10, 1)

print(f"Number of heavily weighted storm hours: {len(sample_weights[sample_weights == 10])}")
# Train the model with the sample_weight parameter
print("Training XGBoost Regressor with heavy dust storm weights...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,                    # <--- The critical addition
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)
