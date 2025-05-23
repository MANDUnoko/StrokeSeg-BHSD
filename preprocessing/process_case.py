from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "bhsd.yaml"

import os
import yaml
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from preprocessing.windowing import apply_window
from preprocessing.resample_utils import resize_to_shape
from preprocessing.normalization import zscore_normalize
from preprocessing.utils import load_nifti_as_array, save_nifti, calculate_snr_cnr

# config 불러오기
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

input_dir = Path(config['input_dir']) / 'train'
output_dir = Path(config['output_dir']) / 'train'
output_dir.mkdir(parents=True, exist_ok=True)

# 샘플 케이스
case_id = 'ID_0b10cbee_ID_f91d6a7cd2'  
img_path = input_dir / 'images' / f'{case_id}.nii'
mask_path = input_dir / 'masks' / f'{case_id}.nii'

# Step 1: Load 이미지 & 마스크
image = load_nifti_as_array(str(img_path))     # shape: (D, H, W)
mask = load_nifti_as_array(str(mask_path))     # shape: (D, H, W)

# Step 2: 윈도우 (다채널 구성)
windowed = []
for clip_min, clip_max in config['window']:
    windowed.append(apply_window(image, clip_min, clip_max))
volume = np.stack(windowed, axis=0)  # shape: (C, D, H, W)

# Step 3: 리샘플 (spacing = [5.0, 1.0, 1.0])
volume = resize_to_shape(volume, config['shape'])
mask = resize_to_shape(mask, config['shape'], is_mask=True)
print("→ after resample: mask unique:", np.unique(mask))
print("→ lesion voxel count:", (mask > 0).sum())

# Step 4: 크기 통일 (crop or pad to shape)
target_shape = tuple(config['shape'])  # (D, H, W)
from preprocessing.utils import crop_or_pad
volume = crop_or_pad(volume, target_shape)
mask = crop_or_pad(mask, target_shape)

# Step 5: 정규화 (Z-score)
volume = zscore_normalize(volume)

# Debug: shape 확인
print("volume shape:", volume.shape)
print("mask shape:", mask.shape)

# Step 6: SNR & CNR 계산
snr, cnr = calculate_snr_cnr(volume, mask)
print(f"[Signal Quality] SNR: {snr:.4f}, CNR: {cnr:.4f}")

# Step 7: 저장 경로 준비
save_name = case_id
pt_path = output_dir / f"{save_name}.pt"
nii_path = output_dir / f"{save_name}_processed.nii.gz"

# Step 8: PyTorch tensor 저장
torch_tensor = torch.tensor(volume, dtype=torch.float32)
torch.save({'volume': torch_tensor, 'mask': torch.tensor(mask)}, pt_path)

# Step 9: NIfTI로 저장 (시각화용)
save_nifti(volume[0], str(nii_path))  # 첫 채널만 저장

# Step 10: 품질 로그 저장 (CSV 추가 or 생성)
import pandas as pd

csv_path = Path("data/logs/quality_metrics.csv")
csv_path.parent.mkdir(exist_ok=True, parents=True)

log = pd.DataFrame([{
    "case_id": case_id,
    "snr": snr,
    "cnr": cnr,
    "quality": "good" if (snr > 1.0 and cnr > 0.5) else "poor"
}])

if csv_path.exists():
    log.to_csv(csv_path, mode='a', index=False, header=False)
else:
    log.to_csv(csv_path, index=False)

print("[STEP 1] Saving .pt...")
torch.save({'volume': torch_tensor, 'mask': torch.tensor(mask)}, pt_path)

print("[STEP 2] Saving NIfTI...")
save_nifti(volume[0], str(nii_path))

print("[STEP 3] Writing CSV log...")
csv_path = Path("data/logs/quality_metrics.csv")
csv_path.parent.mkdir(exist_ok=True, parents=True)

log = pd.DataFrame([{
    "case_id": case_id,
    "snr": snr,
    "cnr": cnr,
    "quality": "good" if (snr > 1.0 and cnr > 0.5) else "poor"
}])

if csv_path.exists():
    print("[STEP 3.1] Appending to CSV...")
    log.to_csv(csv_path, mode='a', index=False, header=False)
else:
    print("[STEP 3.2] Creating CSV...")
    log.to_csv(csv_path, index=False)

print("ALL DONE")
