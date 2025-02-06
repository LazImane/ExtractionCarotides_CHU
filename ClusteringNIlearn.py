import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import nibabel as nib 
from scipy.optimize import curve_fit 
import time

import numpy as np
from matplotlib import patches, ticker

from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations


data_path = "FICHIERSDONNEES/" ## A CHANGER
filename_img_4D =  os.path.join(data_path, "w_PET_4D.nii")

# Charger le fichier NIfTI 4D
nifti_img = nib.load(filename_img_4D)

# Obtenir les données sous forme de tableau NumPy
#img_4D = nifti_img.get_fdata()  # Dimensions typiques : (X, Y, Z, T) ici pas necessaire puisqu'on ward nifti_img(l'image pas le data)

start = time.time()

ward = Parcellations(
    method="ward",
    n_parcels=10, #creating 10 regions of interest ROI
    standardize=False,
    smoothing_fwhm=2.0, #smoothing data to reduce noise
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
ward.fit(nifti_img)
print(f"Ward agglomeration 10 clusters: {time.time() - start:.2f}s")
ward_labels_img = ward.labels_img_

# Now, ward_labels_img are Nifti1Image object, it can be saved to file
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_data_driven_parcellations"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
ward_labels_img.to_filename(output_dir / "ward_parcellation.nii.gz")


first_plot = plotting.plot_roi(
    ward_labels_img, title="Ward parcellation", display_mode="xz"
)

cut_coords = first_plot.cut_coords

