# The new target is PM10
y = final_dataset['pm10']
# Drop BOTH current hour readings, just like before
X = final_dataset.drop(['pm2_5', 'pm10'], axis=1)
# Re-split the data with the new y target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the exact same model instance
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
