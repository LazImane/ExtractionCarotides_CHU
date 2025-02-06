import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.optimize import curve_fit
from nilearn import image
from nilearn.image import new_img_like
from nilearn.image import math_img
from nilearn import plotting
from nilearn.decoding import Decoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

data_path = r"C:/Users/SCD-UM/Documents/GitHub/projetS6/Données_Patients" # à changer
nom = input("Enter one of these patients (P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11)")
nom += ".nii"
filename_img_4D =  os.path.join(data_path, nom)
nifti_img = nib.load(filename_img_4D)
nilearn_img = new_img_like(nifti_img, nifti_img.get_fdata(), affine=nifti_img.affine)
volume = image.index_img(nilearn_img,6)
volume_data = volume.get_fdata() 
threshold = 40
mask_data = np.where(volume_data > threshold, 1, 0).astype(np.float32) #had an error where this wasn't compatible with 64int
z_limit = 4
mask_data[:, :, z_limit:] = 0
mask_img = nib.Nifti1Image(mask_data, nifti_img.affine)
nib.save(mask_img, "carotid_mask_limited.nii")
mask_filename = "carotid_mask_limited.nii"
bg_img = image.smooth_img(image.index_img(nifti_img, 26),fwhm="fast")
mean_intensities = []

num_volumes = nifti_img.shape[-1]
for i in range(num_volumes):
    volume = image.index_img(nifti_img, i)
    volume_data = volume.get_fdata()
    
    masked_data = volume_data[mask_data > 0]
    mean_intensity = masked_data.mean()
    mean_intensities.append(mean_intensity)

threshold = np.mean(mean_intensities) 
labels = [1 if intensity > threshold else 0 for intensity in mean_intensities]

volume_data_list = []
num_voxels = np.sum(mask_data > 0)

for i in range(nifti_img.shape[3]):
    volume = nifti_img.slicer[..., i]
    volume_data = volume.get_fdata()
    
    masked_data = volume_data[mask_data > 0]
    mean_intensity = masked_data.mean() if masked_data.size > 0 else 0
    volume_data_list.append(mean_intensity)
volume_data_list = [0] + volume_data_list

img_4D = nifti_img.get_fdata()
t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
print(len(t))
t_min = t/60
delta_i = np.diff(t_min) # DurÃ©e de chaque frame en min
# Extraire la courbe temporelle pour le voxel
tac = img_4D[18, 52, 3, :]  # Données sur la dimension temporelle
'''
# Visualiser la TAC
time_points = range(len(tac))  # Index temporels (ou utilisez une échelle réelle si disponible)
plt.plot(time_points, tac, marker='o')
plt.title(f'TAC pour le voxel de carotide droite')
plt.xlabel('Temps')
plt.ylabel('Intensité')
plt.grid(True)
plt.show()
'''

########## FAIRE LE FIT A LA PARTIE DECROISSANTE AVEC LE MODELE DE FENG 
def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
    """
    Compute a gamma variate model for the given parameters over time.

    This function calculates the concentration of a tracer using a gamma variate model
    based on the provided amplitudes and decay rates. It is particularly used to model 
    the exponential decay observed in tracer kinetics.

    Parameters:
    t : array-like
        Time points at which the model is evaluated.
    A1, A2, A3 : float
        Amplitudes of the exponential components.
    lambda1, lambda2, lambda3 : float
        Decay rates of the exponential components.

    Returns:
    Cpmodel : array-like
        The computed concentration values at each time point.
    """
    Cpmodel = (A1 - A2 - A3) * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + A3 * np.exp(-lambda3 * t)
    return Cpmodel

t_data = t_min
y_data = volume_data_list

# Trouver l'index de la valeur maximale de yy
idx_max = np.argmax(y_data)

# Extraire la partie dÃ©croissante des donnÃ©es
xx_dec = t_data[idx_max:]
yy_dec = y_data[idx_max:]
#added this line because there was a mismatch 
if len(xx_dec) != len(yy_dec):
    min_len = min(len(xx_dec), len(yy_dec))
    xx_dec = xx_dec[:min_len]
    yy_dec = yy_dec[:min_len]
# Extraire les valeurs de t qui ont été supprimÃ©es
xx_removed = t_data[:idx_max]
yy_removed = [0] * len(xx_removed)

# Ajouter ces valeurs au début de xx_dec et yy_dec
xx_full = np.concatenate((xx_removed, xx_dec))
yy_full = np.concatenate((yy_removed, yy_dec))

params_dec, covariance_dec = curve_fit(gammavariate_model_dec, xx_dec, yy_dec, maxfev=100000000)

IDIF_fit = np.concatenate((yy_removed, gammavariate_model_dec(xx_dec, *params_dec)))

########## PLOT DES RESULTATS
plt.figure(figsize=(10, 6))

# TracÃ© des donnÃ©es expÃ©rimentales
plt.plot(t_min, volume_data_list, 'bo-', label="Données expérimentales", markersize=4)

# TracÃ© du modÃ¨le ajustÃ©
plt.plot(t_min, IDIF_fit, 'r--', label="Modèle ajusté")

plt.xlabel("Temps (minutes)")
plt.ylabel("Concentration (kBq/cc)")
plt.title("Ajustement de la TAC par le modèle gamma-variée")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()