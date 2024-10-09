from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TransformBase(ABC):
    """
    Base class for transformations.

    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, mapobj, **kwargs):
        msg = "The __call__ method must be implemented in the subclass"
        raise NotImplementedError(msg)


class AugmentBase(ABC):
    """
    Base class for augmentations.
    """

    # This will need to take the hyper parameters for the augmentations

    @abstractmethod
    def __init__(self, random_seed: int = 42):
        self.random_state = np.random.RandomState(random_seed)

    @abstractmethod
    def __call__(self, data, **kwargs):
        msg = "The __call__ method must be implemented in the subclass"
        raise NotImplementedError(msg)
