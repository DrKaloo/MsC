import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load metadata
df = pd.read_csv('/Users/kaloyantodorov/Desktop/Python/MSc Dissertation/MRI data/metadata.csv')

# Keep only first scan per patient
df = df.sort_values('scan_filename').groupby('patient_id').first().reset_index()

print(f"Total patients: {len(df)}")
print(f"Class distribution:\n{df['diagnosis'].value_counts()}\n")

# Split WITHOUT stratification (few samples per class)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create folder and save splits
os.makedirs('data_splits', exist_ok=True)
train_df.to_csv('data_splits/train.csv', index=False)
val_df.to_csv('data_splits/val.csv', index=False)
test_df.to_csv('data_splits/test.csv', index=False)

print("Splits created:")
print(f"Train: {len(train_df)} patients")
print(train_df['diagnosis'].value_counts())
print(f"\nValidation: {len(val_df)} patients")
print(val_df['diagnosis'].value_counts())
print(f"\nTest: {len(test_df)} patients")
print(test_df['diagnosis'].value_counts())

print("\n Splits saved to data_splits/")
print("\n Warning: Small dataset - results will be preliminary!")