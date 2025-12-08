import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import os

raw_folder = '/Users/kaloyantodorov/Desktop/Python/MSC Dissertation/MRI data/OAS1_0001_MR1/RAW'

# Find ALL .hdr files
hdr_files = glob(f'{raw_folder}/**/*.hdr', recursive=True)

print(f"Found {len(hdr_files)} .hdr files:\n")

for i, f in enumerate(hdr_files):
    filename = os.path.basename(f)
    print(f"  {i+1}. {filename}")

print("\n" + "="*50)
print("Loading first scan to test...")
print("="*50 + "\n")




# Load the first one
if len(hdr_files) > 0:
    test_file = hdr_files[0]
    print(f"Loading: {os.path.basename(test_file)}\n")
    
    # Load it
    img = nib.load(test_file)
    data = img.get_fdata()
    
    # REMOVE extra dimension if it exists
    if data.ndim == 4:
        data = data.squeeze()  # Removes dimensions of size 1
    
    print(f"Shape: {data.shape}")
    print(f"Voxel size: {img.header.get_zooms()[:3]}")  # Only first 3 values
    print(f"Data range: {data.min():.2f} to {data.max():.2f}")
    
    # Visualise 3 views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (horizontal slice - top view)
    axes[0].imshow(data[:, :, data.shape[2]//2], cmap='gray')
    axes[0].set_title('Axial (top view)')
    axes[0].axis('off')
    
    # Sagittal (vertical slice - side view)
    axes[1].imshow(data[data.shape[0]//2, :, :].T, cmap='gray', origin='lower')
    axes[1].set_title('Sagittal (side view)')
    axes[1].axis('off')
    
    # Coronal (vertical slice - front view)
    axes[2].imshow(data[:, data.shape[1]//2, :].T, cmap='gray', origin='lower')
    axes[2].set_title('Coronal (front view)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('brain_scan_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualization saved as 'brain_scan_preview.png'")
    print("\nYou should see brain structure in the images!")
    
else:
    print("No .hdr files found!")