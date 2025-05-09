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
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_roi
from nilearn.image import index_img
from scipy.cluster.hierarchy import dendrogram,linkage
import random
from nilearn.masking import compute_brain_mask


data_path = r"FICHIERSDONNEES/"
filename_img_4D =  os.path.join(data_path, "p04.nii")


########## CHARGER UN FICHIER .NII
nifti_img = nib.load(filename_img_4D)

########## ACCEDER A LA MATRICE 4D
img_4D = nifti_img.get_fdata() 



#selectionner un timeframe intéressant 
nifti_img_3D = index_img(nifti_img,2) #on choisit le temps !!A MODIFIER POUR CHAQUE PATIENT
#ici j'utilise temporairement le patient 3 timeframe 2 (?)


#extraire données cerveau a partir de la bibliothéque nilearn: nilearn.masking.compute_brain_mask
brain_mask = compute_brain_mask(nifti_img_3D, threshold=0.0)
plot_roi(brain_mask, bg_img=nifti_img_3D, title='Brain Mask over Anatomical Image')
plt.show()


