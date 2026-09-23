# .isnull() finds the blanks, .sum() adds them up per column
print(final_dataset.isnull().sum())
# Forward fill the missing values
final_dataset = final_dataset.ffill()
# Drop any remaining rows with NaN values
final_dataset = final_dataset.dropna()

# Run the check one last time to confirm we are at zero
print("Missing values after cleaning:")
print(final_dataset.isnull().sum())
