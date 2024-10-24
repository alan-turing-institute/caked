from __future__ import annotations

import math

from ccpem_utils.map.parse_mrcmapobj import MapObjHandle


def mask_from_labelobj(label_mapobj: MapObjHandle):
    """
    Create a mask from a label object, where the mask is a boolean array
    where the values are 1 for the labels and 0 for the background.
    """
    mask_obj = label_mapobj.copy(deep=True)
    arr = mask_obj.data
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    mask_obj.data = arr
    return mask_obj


def divx(x, d=8):
    """Ensure the number is divisible (to an integer) by x (to ensure it can pool
    and concatenate max 3 times (2^3))."""
    if x % d != 0:
        y = math.ceil(x / d)
        x = y * d
    return x
