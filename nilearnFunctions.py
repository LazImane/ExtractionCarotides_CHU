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
#import pour récupérer un model de machine et le "former"/train si on veut pas utiliser decoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report




data_path = r"./"
filename_img_4D =  os.path.join(data_path, "w_PET_4D.nii")
nifti_img = nib.load(filename_img_4D)

nilearn_img = new_img_like(nifti_img, nifti_img.get_fdata(), affine=nifti_img.affine)#nilearn est incompatible avec l'image
#donc on la récupère avec nibabel et on la convertit à un objet compatible avec nilearn
print(nilearn_img.shape)
first_img = image.index_img(nilearn_img,0)
#print(first_img)#extraction d'un volume 3D
#il nous donne aussi des méta données et d'une matrice affine
#data = first_img.get_fdata()
#print(f"Data shape: {data.shape}")
#print(f"Data type: {data.dtype}")
#print(f"Min value: {np.min(data)}")#0
#print(f"Max value: {np.max(data)}")#0
#plotting.plot_stat_map(first_img)#the first image is empty as there is no activity frame=0
#number of volumes equals number of frames
#all volumes are empty untill frame = 4
fifth_img = image.index_img(nilearn_img,4)
#plotting.plot_stat_map(fifth_img, threshold=None)
#plt.show()
'''
for img in image.iter_img(nilearn_img):
    # img is now an in-memory 3D img
    plotting.plot_stat_map(
        img, threshold=3,
          display_mode="z", cut_coords=1, the plot shows only one slice (1 coordinate per graph that is z which represents slices)
            colorbar=False
    )
plt.show()
'''
#also if you get Z=negative number it's because the z-axis in voxels runs from the bottom (negative) to the top (positive)
#of the brain
#so what you're seeing is is an actual real world coordinate(millimeters probably) 
#and represents the position of the slice in physical space, rather than simple voxel index numbers
#image.index_img(rsn, slice(3, 5)) getting volumes from 3 to 5

'''
Je pense pour trouver la carotide c'est mieux de ce fixer dans l'axe transversale au cours de temps puisque
avec une tranche assez basse on peut detecter que l'activité de ces artères même dans différents mesure de temps (sauf s'il est trop haute)
'''

#plotting.plot_stat_map(image.smooth_img(image.index_img(nilearn_img,10),fwhm=3)) 
#plt.show()

#DECODING WHICH IS THE PART THAT INTERESTS US
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
print(mask_img.shape)
print(np.unique(mask_data))
print(np.count_nonzero(mask_data))
'''
plotting.plot_roi(
    mask_img,
    bg_img=bg_img,
    cmap="jet",  
    display_mode="ortho",  # Display all three planes (sagittal, coronal, axial)
    alpha=0.5,  
    vmin=0, 
    vmax=1,
    output_file="output_image.png",
    title="Carotid Artery Mask 2" 
)
plotting.show()
'''
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

print("Mean intensities:", mean_intensities)
print("Generated labels:", labels)

behavioral_df = pd.DataFrame({"timepoint": list(range(num_volumes)), "label": labels})
behavioral_df.to_csv("simulated_labels.csv", index=False)

print("Simulated labels saved to 'simulated_labels.csv'")

behavioral_df = pd.read_csv("simulated_labels.csv")
conditions = behavioral_df["label"].to_numpy()

decoder = Decoder(
    estimator="svc", 
    mask=mask_filename, 
    standardize="zscore_sample",
    screening_percentile=10   # added to avoid warnings
)

fmri_niimgs = image.index_img(nifti_img, range(nifti_img.shape[3]))#ici on a changé le nifti à un format compatible avec le decoder
decoder.fit(fmri_niimgs, conditions)
predictions = decoder.predict(fmri_niimgs)
print("Predictions:", predictions)

accuracy = accuracy_score(conditions, predictions)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(conditions, predictions))

#output :
'''
 UserWarning: Brain mask is smaller than .5% of the size of the standard human brain. This object is probably not tuned to be used on such data.
  selector = check_feature_screening(
 UserWarning: After clustering and screening, the decoding model will be trained only on 29 features. Consider raising clustering_percentile or screening_percentile parameters.
  warnings.warn(
Predictions: [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0]
Accuracy: 1.0
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        18 
           1       1.00      1.00      1.00        11

    accuracy                           1.00        29
   macro avg       1.00      1.00      1.00        29
weighted avg       1.00      1.00      1.00        29
'''
#From these results we can see that the mask is too small
#--->because we prioritised finding voxels for the carotide rather than training a model for now
#so we'll make the mask broader in the future
#also the accuracy rate of 1.0 or 100% is meaningless as the model is testng the data he was trained on

# Cross-validation with KFold
def k_fold_cross_validation(fmri_niimgs, conditions, mask_filename, n_splits=5):
    #cv = KFold(n_splits=n_splits) I got warnings
    cv = StratifiedKFold(n_splits=n_splits)
    accuracies = []

    for fold, (train, test) in enumerate(cv.split(conditions,conditions), start=1):
        decoder = Decoder(
            estimator="svc", mask=mask_filename, standardize="zscore_sample"
        )
        decoder.fit(image.index_img(fmri_niimgs, train), conditions[train])
        prediction = decoder.predict(image.index_img(fmri_niimgs, test))
        fold_accuracy = (prediction == conditions[test]).sum() / len(conditions[test])
        accuracies.append(fold_accuracy)
        
        print(f"CV Fold {fold:01d} | Prediction Accuracy: {fold_accuracy:.3f}")

    mean_accuracy = np.mean(accuracies)
    print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.3f}")
    return mean_accuracy

mean_accuracy = k_fold_cross_validation(fmri_niimgs, conditions, mask_filename, n_splits=5)

#graph showing intensity for the carotide artery over time(frames)
'''
plt.figure(figsize=(10, 5))
plt.plot(range(len(mean_intensities)), mean_intensities, label="Mean Intensity")
plt.axhline(threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.2f})")
plt.xlabel("Timepoint")
plt.ylabel("Mean Intensity")
plt.title("Mean Intensity Over Time")
plt.legend()
plt.grid()
plt.show()
'''

''' manière manuelle si on veut pas transformer nifti en fmri
#preparation de la matrice
feature_matrix = []
#extraction des voxels en utilisant le mask
for i in range(num_volumes):
    volume = image.index_img(nifti_img, i)
    volume_data = volume.get_fdata()
    masked_voxels = volume_data[mask_data > 0]
    feature_matrix.append(masked_voxels)

feature_matrix = np.array(feature_matrix)

print("Feature matrix shape:", feature_matrix.shape)

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, labels, test_size=0.2, random_state=42
)

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
'''