# The target: what we are trying to predict
y = final_dataset['pm2_5']

# The features: drop the current hour's air quality readings
X = final_dataset.drop(['pm2_5', 'pm10'], axis=1)
# We already imported train_test_split in our very first step, but here is how we use it:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training data size: {X_train.shape[0]} hours")
print(f"Testing data size: {X_test.shape[0]} hours")
