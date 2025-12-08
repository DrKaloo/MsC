#firsttest1.py
  #quick raw sanity check
  #searches a hardcoded raw folder for .hdr, and loads first scan
  #prints basic info and shows/saves mid-slice preview

#testcode.py
  #main bulk processing
  #converts raw .hdr to processed .nii.gz
  #brain mask + z-score normalise + crop to brain + resize to 96x96x96
  #writes into ALL_PROCESSED, logs successes/failures

#datavisual.py
  #quality control of processed files
  #loads first processed and checks shape
  #plots 3 orthogonal slices and saves the image

#cdrrates.py
  #builds labels and demographics
  #reads each patient text file, extracts CDR/age/sec       #discrepancies in education??
  #

#createsplits.py
  #generates patient-level splits from saved .csv info
  #keeps one scan per patient to reduce leakage
  #produces train.csv, val.csv, test.csv with 60/20/20

#dataset.py
  #pytorch dataset for loading .nii.gz using split CSVs
  #returns tensors shaped in [1, 96, 96, 96] and a label
  #includes random 3D flip augmentation 

#model.py
  #3D residual network 18 layers style classifier
  #designed for 3-class CN/MCI/AD output                     #0 to 1 per 0.1 decimal instead of 3 outputs only!??!!?
  #includes a parameter sanity check 

#train.py
  #Training loop entrypoint
  #Loads splits, builds dataloaders, trains ResNet3D_18
  #Uses CrossEntropy + Adam + LR scheduler



