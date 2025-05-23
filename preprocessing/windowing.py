import numpy as np

def apply_window(image, clip_min, clip_max):
    """HU 윈도우 적용 후 0~1로 정규화"""
    image = np.clip(image, clip_min, clip_max)
    return (image - clip_min) / (clip_max - clip_min + 1e-8)
