#pour tester : pytest test_pipeline.py

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from idif_pipeline_pet import get_data, get_brain_mask, get_IDIF_FIT  # adapte le nom du module si besoin

### ---------- Test 1 : get_data avec un fichier manquant ----------

def test_get_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_data("fake_path", "fake_patient")


### ---------- Test 2 : get_brain_mask retourne bien un masque binaire ----------

def test_get_brain_mask_returns_binary():
    # Création d'une image 4D factice
    fake_data = np.random.rand(10, 10, 10, 5)
    affine = np.eye(4)

    mask = get_brain_mask(fake_data, affine, method="brain")  # ou "background"
    assert mask.dtype == bool
    assert mask.shape == (10, 10, 10)
    assert set(np.unique(mask)).issubset({False, True})


### ---------- Test 3 : get_IDIF_FIT avec une courbe simple ----------

def test_get_IDIF_FIT_simple():
    # Une courbe décroissante simulée (forme exp)
    tac = [0] + list(5 * np.exp(-0.1 * np.array(range(1, 30))))
    dummy_img = nib.Nifti1Image(np.zeros((1,1,1,30)), np.eye(4))

    t_min, xx, fitted = get_IDIF_FIT(dummy_img, tac)

    assert len(fitted) == len(xx)
    assert len(t_min) == 30
    assert all(isinstance(v, float) for v in fitted)
