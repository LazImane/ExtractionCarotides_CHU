import os
import numpy as np
import nibabel as nib
from nilearn import plotting 
from nilearn.masking import compute_background_mask
from scipy.ndimage import binary_erosion, binary_closing, label
from skimage import filters, measure
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_data(path, patient_id):
    filename = os.path.join(path, patient_id+".nii")
    nifti = nib.load(filename)
    return nifti.get_fdata(), nifti.affine

def get_brain_mask(data, affine):
    mean_data = np.mean(data, axis=-1) if data.ndim == 4 else data
    mask = compute_background_mask(
        nib.Nifti1Image(mean_data, affine),
        connected=False,
        opening=1, 
    ).get_fdata().astype(bool)
    
    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = binary_erosion(mask, iterations=1)
    
    labeled_mask, num_labels = label(mask)
    if num_labels > 0:
        largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
        mask = largest_component
    
    return mask

def extract_carotids_simple(mean_volume, brain_mask, affine):
    # Find lower brain region
    z_min = np.where(np.any(brain_mask, axis=(0, 1)))[0][-1] 
    z_search_start = z_min + 1
    z_search_end = z_min + 30  
    carotid_region = mean_volume[..., z_search_start:z_search_end]

    # Threshold to find bright regions
    thresh = filters.threshold_li(carotid_region)
    bright_mask = carotid_region > thresh

    # Find connected components
    labeled = measure.label(bright_mask)
    regions = measure.regionprops(labeled, intensity_image=carotid_region)
    
    # Filter components by size
    carotid_candidates = [r for r in regions if 50 < r.area < 1000]

    # Select top 2 brightest candidates (for left/right carotids)
    carotid_candidates.sort(key=lambda x: x.max_intensity, reverse=True)
    final_mask = np.isin(labeled, [r.label for r in carotid_candidates[:2]])

    # Create full volume mask
    full_mask = np.zeros_like(mean_volume)
    full_mask[..., z_search_start:z_search_end] = final_mask

    return full_mask

def generate_graph(volume_data_list, patient):
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,
                  360,420,480,540,600,900,1200,1500,1800,2100,2400])
    t_min = t/60
    
    # Gamma variate model
    def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
        return (A1 - A2 - A3)*np.exp(-lambda1*t) + A2*np.exp(-lambda2*t) + A3*np.exp(-lambda3*t)
    
    # Find washout phase
    idx_max = np.argmax(volume_data_list)
    xx_dec = t_min[idx_max:]
    yy_dec = volume_data_list[idx_max:]
    
    # Handle potential mismatches
    min_len = min(len(xx_dec), len(yy_dec))
    xx_dec = xx_dec[:min_len]
    yy_dec = yy_dec[:min_len]
    
    # Fit model to washout phase
    try:
        params_dec, _ = curve_fit(gammavariate_model_dec, xx_dec, yy_dec, maxfev=1000000)
        fit_curve = gammavariate_model_dec(xx_dec, *params_dec)
    except RuntimeError:
        print("Curve fitting failed - showing raw data only")
        fit_curve = None

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t_min, volume_data_list, 'bo-', label="Données expérimentales", markersize=4)
    if fit_curve is not None:
        plt.plot(xx_dec, fit_curve, 'r--', label="Modèle ajusté")
    plt.xlabel("Temps (minutes)")
    plt.ylabel("Concentration (kBq/cc)")
    plt.title(f"Ajustement de la TAC pour le patient {patient}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    path = "./patients"
    patient_id = "P04"
    
    # Load and process data
    data, affine = get_data(path, patient_id)
    brain_mask = get_brain_mask(data, affine)
    mean_brain = np.mean(data, axis=-1) 
    
    # Extract carotids and compute mean TAC
    carotid_mask = extract_carotids_simple(mean_brain, brain_mask, affine)
    
    masked_data = data * carotid_mask[..., np.newaxis]
    sum_intensity = np.sum(masked_data, axis=(0, 1, 2))
    sum_voxels = np.sum(carotid_mask)
    
    mean_tac = np.zeros_like(sum_intensity)
    valid_mask = (sum_voxels > 0)
    mean_tac[valid_mask] = sum_intensity[valid_mask] / sum_voxels
    


    plotting.plot_roi(
        nib.Nifti1Image(carotid_mask, affine),
        bg_img=nib.Nifti1Image(mean_brain, affine),
        title="Carotid Mask",
        alpha=0.5
    )
    plotting.show()

if __name__ == "__main__":
    main()