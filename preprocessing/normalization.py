import numpy as np

def zscore_normalize(volume):
    """
    volume: (C, D, H, W)
    """
    normed = np.zeros_like(volume)
    for c in range(volume.shape[0]):
        mean = volume[c].mean()
        std = volume[c].std()
        normed[c] = (volume[c] - mean) / (std + 1e-8)
    return normed
