import SimpleITK as sitk
import numpy as np
import torch

def resample_volume(image_np, mask_np, spacing, original_spacing=(5.0, 0.5, 0.5)):
    """
    image_np: (C, D, H, W) numpy array
    mask_np: (D, H, W) numpy array
    spacing: target spacing, e.g., [5.0, 1.0, 1.0]
    """

    resampled_image_channels = []
    for c in range(image_np.shape[0]):
        image_sitk = sitk.GetImageFromArray(image_np[c])
        image_sitk.SetSpacing(original_spacing[::-1])  # spacing: (x, y, z)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(spacing[::-1])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetSize([
            int(round(image_sitk.GetSize()[i] * original_spacing[::-1][i] / spacing[::-1][i]))
            for i in range(3)
        ])
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampled = resampler.Execute(image_sitk)
        resampled_np = sitk.GetArrayFromImage(resampled)
        resampled_image_channels.append(resampled_np)

    resampled_image = np.stack(resampled_image_channels, axis=0)

    # 마스크는 nearest로
    mask_sitk = sitk.GetImageFromArray(mask_np)
    mask_sitk.SetSpacing(original_spacing[::-1])
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask_sitk = resampler.Execute(mask_sitk)
    resampled_mask = sitk.GetArrayFromImage(resampled_mask_sitk).astype(np.uint8)

    return resampled_image, resampled_mask

import torch.nn.functional as F

def resize_to_shape(volume, shape, is_mask=False):
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]
    tensor = torch.tensor(volume[None], dtype=torch.float32)
    mode = 'nearest' if is_mask else 'trilinear'

    if mode == 'nearest':
        out = F.interpolate(tensor, size=shape, mode=mode)
    else:
        out = F.interpolate(tensor, size=shape, mode=mode, align_corners=False)

    out_np = out.numpy().squeeze(0)
    return out_np if out_np.shape[0] > 1 else out_np[0]


