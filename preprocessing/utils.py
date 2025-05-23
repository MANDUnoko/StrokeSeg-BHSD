# utils.py
import numpy as np
import nibabel as nib
import torch.nn.functional as F

def load_nifti_as_array(path):
    return nib.load(path).get_fdata().astype(np.float32)

def save_nifti(data, path, spacing=(5.0, 1.0, 1.0)):
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    try:
        print("🧪 save_nifti() START")
        print("  ↳ path:", path)
        print("  ↳ shape:", data.shape)
        print("  ↳ dtype:", data.dtype)
        print("  ↳ min/max:", data.min(), data.max())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = data.astype(np.float32)
        affine = np.diag([spacing[2], spacing[1], spacing[0], 1])
        img = nib.Nifti1Image(data, affine)
        nib.save(img, path)
        print("✅ NIfTI saved to:", path)
    except Exception as e:
        print("❌ NIfTI SAVE ERROR:", e)

def crop_or_pad(volume, target_shape):
    """
    Crop or zero-pad volume to match target_shape
    volume: shape (C, D, H, W) or (D, H, W)
    """
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]  # (1, D, H, W)
    c, d, h, w = volume.shape
    td, th, tw = target_shape
    out = np.zeros((c, td, th, tw), dtype=volume.dtype)

    dz = min(d, td)
    dy = min(h, th)
    dx = min(w, tw)

    out[:, :dz, :dy, :dx] = volume[:, :dz, :dy, :dx]
    return out if out.shape[0] > 1 else out[0]

def calculate_snr_cnr(volume, mask):
    lesion = volume[0][mask > 0]
    background = volume[0][mask == 0]

    # 디버그 출력
    print("Lesion mean/std:", lesion.mean(), lesion.std())
    print("Background mean/std:", background.mean(), background.std())
    print("Lesion voxel count:", lesion.size)
    print("Background voxel count:", background.size)
    print("mask unique values:", np.unique(mask))
    print("mask dtype:", mask.dtype)
    print("volume[0] mean/std before masking:", volume[0].mean(), volume[0].std())


    if lesion.size < 10 or background.size < 1000:
        return 0.0, 0.0

    lesion_mean = lesion.mean()
    lesion_std = lesion.std()
    background_mean = background.mean()
    background_std = background.std()

    snr = lesion_mean / (lesion_std + 1e-8)
    cnr = abs(lesion_mean - background_mean) / np.sqrt(lesion_std**2 + background_std**2 + 1e-8)
    return snr, cnr


