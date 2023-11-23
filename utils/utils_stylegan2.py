import numpy as np


def get_weight_initializer_runtime_coef(shape, gain=1, use_wscale=True, lrmul=1):
    """ get initializer and lr coef for different weights shapes"""
    fan_in = np.prod(shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    return init_std, runtime_coef


def nf(stage, fmap_base=1 << 8, fmap_decay=1.0, fmap_min=1, fmap_max=512):
    return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
