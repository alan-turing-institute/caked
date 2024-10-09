from __future__ import annotations

import random
from enum import Enum

import numpy as np
from ccpem_utils.map.array_utils import rotate_array
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle

from .base import AugmentBase


class Augments(Enum):
    """ """

    RANDOMROT = "randrot"
    ROT90 = "rot90"


def get_augment(augment: str, random_seed) -> AugmentBase:
    """ """

    if augment == Augments.RANDOMROT.value:
        return RandomRotationAugment(random_seed=random_seed)
    if augment == Augments.ROT90.value:
        return Rotation90Augment(random_seed=random_seed)

    msg = f"Unknown Augmentation: {augment}, please choose from {Augments.__members__}"
    raise ValueError(msg)


class ComposeAugment:
    """
    Compose multiple Augments together.

    :param augments: (list) list of augments to compose

    :return: (np.ndarrry) transformed array
    """

    def __init__(self, augments: list[str], random_seed: int = 42):
        self.random_seed = random_seed
        self.augments = augments

    def __call__(self, data: np.ndarray, **kwargs) -> MapObjHandle:
        for augment in self.augments:
            data, augment_kwargs = get_augment(augment, random_seed=self.random_seed)(
                data, **kwargs
            )

            kwargs.update(augment_kwargs)

        return data, kwargs


class RandomRotationAugment(AugmentBase):
    """
    Random or controlled rotation (if ax and an kwargs provided).

    :param data: (np.ndarray) 3d volume
    :param return_all: (bool) if True, will parameters of the rotation (ax, an)
    :param interp: (bool) if True, will interpolate the rotation
    :param ax: (int) 0 for yaw, 1 for pitch, 2 for roll
    :param an: (int) number of times to rotate, between <1 and 3>

    :return: (np.ndarray) rotated volume or (np.ndarray, int, int) rotated volume and rotation parameters
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)

    def __call__(
        self,
        data: np.ndarray,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, int, int]:
        ax = kwargs.get("ax", None)
        an = kwargs.get("an", None)
        interp = kwargs.get("interp", True)

        if (ax is not None and an is None) or (ax is None and an is not None):
            msg = "When specifying rotation, please use both arguments to specify the axis and angle."
            raise RuntimeError(msg)
        rotations = [(0, 1), (0, 2), (1, 2)]  # yaw, pitch, roll
        if ax is None and an is None:
            axes = random.randint(0, 2)
            set_angles = [30, 60, 90]
            angler = random.randint(0, 2)
            angle = set_angles[angler]
        else:
            axes = ax
            angle = an

        r = rotations[axes]
        data = rotate_array(data, angle, axes=r, interpolate=interp, reshape=False)

        return data, {"ax": axes, "an": angle}


class Rotation90Augment(AugmentBase):
    """
    Rotate the volume by 90 degrees.

    :param data: (np.ndarray) 3d volume
    :param return_all: (bool) if True, will parameters of the rotation (ax, an)
    :param interp: (bool) if True, will interpolate the rotation
    :param ax: (int) 0 for yaw, 1 for pitch, 2 for roll
    :param an: (int) number of times to rotate, between <1 and 3>

    :return: (np.ndarray) rotated volume or (np.ndarray, int, int) rotated volume and rotation parameters
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)

    def __call__(
        self,
        data: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        _ = data
        _ = kwargs
        msg = "Rotation90Augment not implemented yet."
        raise NotImplementedError(msg)
