import pandas as pd

# Load the CSV file
df = pd.read_csv(r"..\data\gesture_landmarks_both_hands.csv")

# Get unique labels
unique_labels = df['label'].unique()
label_count = df['label'].nunique()

# Print the count and each label on a new line
print(f"Total Unique Labels: {label_count}")
print("Labels:")
for label in unique_labels:
    print(f"- {label}")
