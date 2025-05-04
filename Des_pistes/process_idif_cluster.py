import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def clusterbelongstobrain(voxel_coordinates, brain_mask_coords):
    """
    Checks whether all voxel coordinates are within the brain mask boundaries along the X-axis.
    """
    x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    for coord in voxel_coordinates:
        if coord[0] < x_min or coord[0] > x_max:
            return False
    return True

def process_cluster(voxel_coordinates, brain_mask_coords, nifti_img_3D, nifti_img4D, output_prefix="carotide"):
    """
    Process a cluster of voxels to:
    - Separate into left and right carotid candidates
    - Check if clusters belong to brain region
    - Save masks as NIfTI
    - Extract TACs from 4D data

    Returns:
    --------
    TAC_gauche : np.array
        Time-activity curve for left carotid candidate
    TAC_droite : np.array
        Time-activity curve for right carotid candidate
    """
    x_center = nifti_img_3D.shape[0] // 2
    voxels_gauche = [v for v in voxel_coordinates if v[0] < x_center]
    voxels_droite = [v for v in voxel_coordinates if v[0] >= x_center]

    # Anatomical check
    print("== Vérification de l'appartenance au cerveau ==")
    if clusterbelongstobrain(np.array(voxels_gauche), brain_mask_coords):
        print(" Carotide gauche détectée")
    else:
        print(" Carotide gauche douteuse (hors cerveau)")

    if clusterbelongstobrain(np.array(voxels_droite), brain_mask_coords):
        print(" Carotide droite détectée")
    else:
        print(" Carotide droite douteuse (hors cerveau)")

    # Create and save masks
    shape = nifti_img_3D.shape
    mask_g = np.zeros(shape, dtype=np.uint8)
    mask_d = np.zeros(shape, dtype=np.uint8)

    for x, y, z in voxels_gauche:
        mask_g[x, y, z] = 1
    for x, y, z in voxels_droite:
        mask_d[x, y, z] = 1

    nib.save(nib.Nifti1Image(mask_g, nifti_img_3D.affine), f"{output_prefix}_gauche.nii")
    nib.save(nib.Nifti1Image(mask_d, nifti_img_3D.affine), f"{output_prefix}_droite.nii")
    print(" Masques sauvegardés.")

    # Extract TACs
    data4D = nifti_img4D.get_fdata()

    TAC_gauche = np.mean([data4D[x, y, z, :] for x, y, z in voxels_gauche], axis=0)
    TAC_droite = np.mean([data4D[x, y, z, :] for x, y, z in voxels_droite], axis=0)

    return TAC_gauche, TAC_droite
