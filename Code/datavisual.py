import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

# -> processed scan
processed_folder = '/Users/kaloyantodorov/Desktop/Python/MSc Dissertation/MRI data/ALL_PROCESSED'
scans = glob(f'{processed_folder}/*.nii.gz')

print(f"Found {len(scans)} processed scans")
print(f"\nFirst scan: {scans[0]}")

# Load the first one
scan = nib.load(scans[0])
data = scan.get_fdata()

print(f"Processed scan shape: {data.shape}")

# Visualise
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(data[48, :, :], cmap='grey')
axes[0].set_title('Sagittal')
axes[0].axis('off')

axes[1].imshow(data[:, 48, :], cmap='grey')
axes[1].set_title('Coronal')
axes[1].axis('off')

axes[2].imshow(data[:, :, 48], cmap='grey')
axes[2].set_title('Axial')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('brain_visualization_meeting.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Saved as 'brain_visualization_meeting.png'")


#demographics. stat split, control vs alzh  BALANCED DATA