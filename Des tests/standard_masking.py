import os
import numpy as np
import nibabel as nib
from nilearn import image,masking
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nilearn.image import new_img_like

#Load images

def load_NifTI_image(path,patient_id):
    """Load the NIfTI and NileArn images"""
    filename = os.path.join(path, patient_id+".nii")
    nifti = nib.load(filename)
    return nifti

def load_NileArn_image(nifti):
    return new_img_like(nifti, nifti.get_fdata(dtype=np.float32), affine=nifti.affine)

#masks and extracting data

def apply_mask(volume_data, threshold=23, z_limit=8):
    """Apply a mask to the NIfTI image."""
    mask_data = np.where(volume_data > threshold, 1, 0).astype(np.uint8)
    mask_data[:, :, z_limit:] = 0
    return mask_data

def calculate_mean_intensities(nifti_img,mask_data):
    """Calculate mean intensities for one pre-defined mask."""
    mean_intensities = []
    intensities_per_volume = []
    num_volumes = nifti_img.shape[-1]
    nilearn = load_NileArn_image(nifti_img)
    for i in range(num_volumes):
        volume = image.index_img(nilearn, i)
        volume_data = volume.get_fdata(dtype=np.float32)
        masked_data = volume_data[mask_data > 0]
        mean_intensity = masked_data.mean() if masked_data.size > 0 else 0
        mean_intensities.append(mean_intensity)
        intensities_per_volume.append(mean_intensity)
    return mean_intensities


def calculate_mean_intensities_per_frame(nifti_img):
    """Calculate mean intensities for each volume."""
    intensities_per_volume = []
    num_volumes = nifti_img.shape[-1]
    nilearn = load_NileArn_image(nifti_img)

    for i in range(num_volumes):
        volume = image.index_img(nilearn, i)
        volume_data = volume.get_fdata(dtype=np.float32)
        mask_data = apply_mask(volume_data)
        masked_data = volume_data[mask_data > 0]
        mean_intensity = masked_data.mean() if masked_data.size > 0 else 0
        intensities_per_volume.append(mean_intensity)
    return intensities_per_volume

def find_best_volume(intensities_per_volume):
    """Find the volume with the highest intensity."""
    max_int = np.max(intensities_per_volume)
    entry_seuil = 0.8 * max_int
    best_vol = next((i for i, val in enumerate(intensities_per_volume) if val >= entry_seuil), np.argmax(intensities_per_volume))
    return best_vol

def realigned_images(nft_img,mask_img,std_nft_img) :
    if not np.array_equal(nft_img.affine, std_nft_img.affine):
        nft_img = image.resample_to_img(nft_img, std_nft_img, interpolation='nearest',force_resample=True,copy_header=True)
    if not np.array_equal(mask_img.affine, nft_img.affine):
        mask_img = image.resample_to_img(mask_img, nft_img, interpolation='nearest',force_resample=True,copy_header=True)
    #this part of code is dedicated to my computer because he can't allocate space memory no more
    nft_img = image.new_img_like(nft_img, nft_img.get_fdata(dtype=np.float32))
    mask_img = image.new_img_like(mask_img, mask_img.get_fdata().astype(np.int8))
    return nft_img,mask_img

def find_max_vx_coords(masked_values,mask_data) :
    max_i = np.argmax(masked_values)
    volume_i,voxel_i = np.unravel_index(max_i,masked_values.shape)
    x,y,z = np.where(mask_data>0)
    x_max,y_max,z_max = x[voxel_i] , y [voxel_i] ,z[voxel_i]
    return x_max,y_max,z_max

def generate_graph(nft_img,volume_data_list,patient):
    img_4D = nft_img.get_fdata()
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
    print(len(t))
    t_min = t/60
    delta_i = np.diff(t_min) # DurÃ©e de chaque frame en min
	# Extraire la courbe temporelle pour le voxel
    tac = img_4D[18, 52, 3, :]  # Données sur la dimension temporelle


########## FAIRE LE FIT A LA PARTIE DECROISSANTE AVEC LE MODELE DE FENG 
    def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
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
    plt.title("Ajustement de la TAC par le modèle gamma-variée pour le patient "+ patient)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plot_filename = os.path.join("./graphs/masque_standard", f"{patient}_TAC_fit.png")
    # plt.savefig(plot_filename, dpi=300)
    # plt.close()
    plt.show()
    #print(f"Graphique enregistré : {plot_filename}")
    
def one_voxel():
    path = "./patients"
    
    #Récupérer un mask standard
    
    std_nft_img = load_NifTI_image(path,"w_PET_4D")
    std_nlrn_img = load_NileArn_image(std_nft_img)
    int_per_volume = calculate_mean_intensities_per_frame(std_nft_img)
    std_bst_volume_index = find_best_volume(int_per_volume)
    print("index de meilleur volume : ",std_bst_volume_index)
    std_bst_volume = image.index_img(std_nlrn_img,std_bst_volume_index)
    mask_data = apply_mask(std_bst_volume.get_fdata(dtype=np.float32))
    mask_img = nib.Nifti1Image(mask_data, std_nft_img.affine)
    print(np.count_nonzero(mask_data))#nb de voxels séléctionnées par le masque
    
    #Appliquer le masque aux autres patients
    patients = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09', 'P10', 'P11']
    for patient in patients :
        nft_img = load_NifTI_image(path,patient)
        nft_img,mask_img = realigned_images(nft_img,mask_img,std_nft_img)
        masked_values = masking.apply_mask(nft_img,mask_img)
        print("no problem for now for "+patient)
        x,y,z = find_max_vx_coords(masked_values,mask_data)
        vx= nft_img.get_fdata()[x,y,z,:]
        volume_data_list = [0] + list(vx)
        
        generate_graph(nft_img,volume_data_list,patient)
    

def group_of_voxels():
    path = "./patients"
    
    #Récupérer un mask standard
    
    std_nft_img = load_NifTI_image(path,"w_PET_4D")
    std_nlrn_img = load_NileArn_image(std_nft_img)
    int_per_volume = calculate_mean_intensities_per_frame(std_nft_img)
    std_bst_volume_index = find_best_volume(int_per_volume)
    print("index de meilleur volume : ",std_bst_volume_index)
    std_bst_volume = image.index_img(std_nlrn_img,std_bst_volume_index)
    mask_data = apply_mask(std_bst_volume.get_fdata(dtype=np.float32))
    mask_img = nib.Nifti1Image(mask_data, std_nft_img.affine)
    print(np.count_nonzero(mask_data))#nb de voxels séléctionnées par le masque
    
    #Appliquer le masque aux autres patients
    patients = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09', 'P10', 'P11']
    for patient in patients :
        nft_img = load_NifTI_image(path,patient)
        nft_img,mask_img = realigned_images(nft_img,mask_img,std_nft_img)
        masked_values = masking.apply_mask(nft_img,mask_img)
        print("no problem for now for "+patient)
        x,y,z = find_max_vx_coords(masked_values,mask_data)
        print(f"les coordonnées de la voxels avec intens max et inclut dans le masque ({x},{y},{z})")
        group_size = 2
        
        # Calculer les limites du groupe
        x_min, x_max = max(0, x - group_size), min(nft_img.get_fdata().shape[0], x + group_size + 1)
        y_min, y_max = max(0, y - group_size), min(nft_img.get_fdata().shape[1], y + group_size + 1)
        z_min, z_max = max(0, z - group_size), min(nft_img.get_fdata().shape[2], z + group_size + 1)
        group_tac = nft_img.get_fdata()[x_min:x_max, y_min:y_max, z_min:z_max, :]
        volume_data_list = np.mean(group_tac, axis = (0, 1, 2))
            
        volume_data_list = [0] + list(volume_data_list )
        
        print(f"la liste pour le patient {patient} est de taille {len(volume_data_list)}")
        
        generate_graph(nft_img,volume_data_list,patient)
        
def main():
    choice = input("Press 1 to get the graph for 1 voxel anything else for a group of voxels")
    if choice == "1":
        one_voxel()
    else:
        group_of_voxels()

if __name__ == "__main__":
    main()