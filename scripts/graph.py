# Ask the updated model to predict the test set
y_pred_weighted = model.predict(X_test)
import matplotlib.pyplot as plt

# Create the canvas
plt.figure(figsize=(14, 6))

# Plot the actual historical values in blue
plt.plot(y_test.values, label='Actual PM10', color='blue', alpha=0.6)

# Plot the NEW weighted predictions in red
plt.plot(y_pred_weighted, label='Predicted PM10 (Weighted)', color='red', alpha=0.8, linestyle='--')

# Add labels and a title reflecting the new architecture
plt.title('Updated XGBoost PM10 Forecast (Weighted & Rolling Features)')
plt.xlabel('Hours in the Test Set (Chronological)')
plt.ylabel('PM10 Concentration (µg/m³)')

# Display the legend
plt.legend()

# Render the graph
plt.show()
