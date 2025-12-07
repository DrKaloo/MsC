import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from glob import glob
import os

def preprocess_scan(input_path, output_path):
    """
    Takes raw .hdr file → outputs 96x96x96 normalized .nii.gz
    """
    
    # 1. Load scan
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Handle 4D data (squeeze out extra dimension)
    if data.ndim == 4:
        data = data.squeeze()
    
    # 2. Basic brain mask (threshold to remove background)
    brain_mask = data > (data.mean() * 0.1)
    
    # 3. Normalize intensity (z-score within brain)
    brain_voxels = data[brain_mask]
    mean_val = brain_voxels.mean()
    std_val = brain_voxels.std()
    data_normalized = (data - mean_val) / (std_val + 1e-8)
    data_normalized[~brain_mask] = 0  # Zero out background
    
    # 4. Crop to brain bounding box
    coords = np.array(np.where(brain_mask))
    x_min, y_min, z_min = coords.min(axis=1)
    x_max, y_max, z_max = coords.max(axis=1)
    
    data_cropped = data_normalized[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # 5. Resize to 96x96x96 for model
    target_shape = (96, 96, 96)
    zoom_factors = [target_shape[i] / data_cropped.shape[i] for i in range(3)]
    data_resized = zoom(data_cropped, zoom_factors, order=1)
    
    # 6. Save as .nii.gz (modern format)
    output_img = nib.Nifti1Image(data_resized, affine=np.eye(4))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(output_img, output_path)
    
    return data_resized.shape


# Main folder with all patient folders
main_folder = '/Users/kaloyantodorov/Desktop/Python/MSc Dissertation/MRI data'
processed_folder = os.path.join(main_folder, 'ALL_PROCESSED')

# Find all .hdr files in all RAW subfolders
# This searches: MRI data/OAS1_XXXX_MR1/RAW/*.hdr
hdr_files = glob(f'{main_folder}/**/RAW/*.hdr', recursive=True)

print(f"Found {len(hdr_files)} scans across all patients\n")
print("="*60)

success_count = 0
fail_count = 0

for i, hdr_path in enumerate(hdr_files, 1):
    # Extract patient ID and scan name
    # Example path: .../OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr
    parts = hdr_path.split('/')
    patient_id = parts[-3]  # OAS1_0001_MR1
    filename = os.path.basename(hdr_path).replace('.hdr', '')
    
    output_path = f'{processed_folder}/{patient_id}_{filename}.nii.gz'
    
    try:
        preprocess_scan(hdr_path, output_path)
        success_count += 1
        print(f"✓ [{i}/{len(hdr_files)}] {patient_id} - {filename}")
    except Exception as e:
        fail_count += 1
        print(f"[{i}/{len(hdr_files)}] {patient_id} - {filename}: {e}")

print("\n" + "="*60)
print(f" Successfully processed: {success_count} scans")
print(f" Failed: {fail_count} scans")
print(f"\n All processed files saved to: {processed_folder}")