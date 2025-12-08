import os
import pandas as pd
from glob import glob

main_folder = '/Users/kaloyantodorov/Desktop/Python/MSc Dissertation/MRI data'
processed_folder = os.path.join(main_folder, 'ALL_PROCESSED')

# Find all patient folders
patient_folders = glob(f'{main_folder}/OAS1_*_MR1')

print(f"Found {len(patient_folders)} patients\n")
print("="*60)

metadata = []
skipped = []

for patient_folder in sorted(patient_folders):
    patient_id = os.path.basename(patient_folder)
    txt_file = f'{patient_folder}/{patient_id}.txt'
    
    if not os.path.exists(txt_file):
        print(f"No .txt file for {patient_id}")
        skipped.append(patient_id)
        continue
    
    # Read the .txt file
    with open(txt_file, 'r') as f:
        content = f.read()
    
    # Extract CDR, AGE, SEX
    cdr = None
    age = None
    sex = None
    
    for line in content.split('\n'):
        if 'CDR:' in line:
            try:
                cdr_str = line.split('CDR:')[1].strip()
                if cdr_str:  # Check if not empty
                    cdr = float(cdr_str)
            except (ValueError, IndexError):
                pass
        
        if 'AGE:' in line:
            try:
                age_str = line.split('AGE:')[1].strip()
                if age_str:
                    age = int(age_str)
            except (ValueError, IndexError):
                pass
        
        if 'M/F:' in line:
            try:
                sex = line.split('M/F:')[1].strip()
            except IndexError:
                pass
    
    # Skip if CDR is missing (critical for labels)
    if cdr is None:
        print(f"{patient_id}: Missing CDR - SKIPPED")
        skipped.append(patient_id)
        continue
    
    # Map CDR to diagnosis
    if cdr == 0:
        diagnosis = 'CN'
        label = 0
    elif cdr == 0.5:
        diagnosis = 'MCI'  # Very mild dementia
        label = 1
    elif cdr >= 1:
        diagnosis = 'AD'   # Dementia
        label = 2
    else:
        diagnosis = 'Unknown'
        label = -1
    
    # Find all processed scans for this patient
    processed_scans = glob(f'{processed_folder}/{patient_id}_*.nii.gz')
    
    if len(processed_scans) == 0:
        print(f"{patient_id}: No processed scans found")
        skipped.append(patient_id)
        continue
    
    for scan_path in processed_scans:
        scan_filename = os.path.basename(scan_path)
        
        metadata.append({
            'patient_id': patient_id,
            'scan_filename': scan_filename,
            'scan_path': scan_path,
            'cdr': cdr,
            'age': age,
            'sex': sex,
            'diagnosis': diagnosis,
            'label': label
        })
    
    print(f"✓ {patient_id}: CDR={cdr} → {diagnosis} ({len(processed_scans)} scans)")

# Create DataFrame
df = pd.DataFrame(metadata)

# Save to CSV
metadata_file = os.path.join(main_folder, 'metadata.csv')
df.to_csv(metadata_file, index=False)

print("\n" + "="*60)
print(f"Metadata saved to: {metadata_file}")
print(f"\nProcessed patients: {len(df['patient_id'].unique())}")
print(f"Skipped patients: {len(skipped)}")
if skipped:
    print(f"Skipped IDs: {', '.join(skipped)}")

print(f"\nTotal scans: {len(df)}")
print(f"\nClass distribution:")
print(df['diagnosis'].value_counts())
print(f"\nPatients per class:")
print(df.groupby('diagnosis')['patient_id'].nunique())