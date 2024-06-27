"""
Functions for loading the data from disk.
Largely taken from https://github.com/alan-turing-institute/affinity-vae
"""

from __future__ import annotations

import logging
import os
import random
import typing
from pathlib import Path

import mrcfile
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .base import AbstractDataLoader, AbstractDataset

np.random.seed(42)
TRANSFORM_OPTIONS = ["normalise", "gaussianblur", "shiftmin"]


class DiskDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
        transformations: list[str] | None = None,
    ) -> None:
        """
        DataLoader implementation for loading data from disk.

        Args:
            dataset_size (int | None, optional): The maximum number of samples to load from the dataset. If None, load all samples. Default is None.
            save_to_disk (bool, optional): Whether to save the loaded data to disk. Default is False.
            training (bool, optional): Whether the DataLoader is used for training. Default is True.
            classes (list[str] | None, optional): A list of classes to load from the dataset. If None, load all classes. Default is None.
            pipeline (str, optional): The data loading pipeline to use. Default is "disk".
            transformations (str | None, optional): The data transformations to apply. If None, no transformations are applied. Default is None.

        Raises:
            RuntimeError: If not all classes in the list are present in the directory.
            RuntimeError: If no processing is required because no transformations were provided.
            RuntimeError: If split size is not provided for training.
            RuntimeError: If train and validation sets are smaller than 2 samples.

        Attributes:
            dataset_size (int | None): The maximum number of samples to load from the dataset.
            save_to_disk (bool): Whether to save the loaded data to disk.
            training (bool): Whether the DataLoader is used for training.
            classes (list[str]): A list of classes to load from the dataset.
            pipeline (str): The data loading pipeline to use.
            transformations (str | None): The data transformations to apply.
            debug (bool): Whether to enable debug mode.
            dataset (DiskDataset): The loaded dataset.

        Methods:
            load(datapath, datatype): Load the data from the specified path and data type.
            process(paths, datatype): Process the loaded data with the specified transformations.
            get_loader(batch_size, split_size): Get the data loader for training or testing.
        """
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training
        self.pipeline = pipeline
        self.transformations = transformations
        self.debug = False

        if classes is None:
            self.classes = []
        else:
            self.classes = classes

    def load(self, datapath, datatype) -> None:
        """
        Load the data from the specified path and data type.

        Args:
            datapath (str): The path to the directory containing the data.
            datatype (str): The type of data to load.

        Returns:
            None
        """
        paths = [f for f in os.listdir(datapath) if "." + datatype in f]

        if not self.debug:
            random.shuffle(paths)

        # ids right now depend on the data being saved with a certain format (id in the first part of the name, separated by _)
        # TODO: make this more general/document in the README
        ids = np.unique([f.split("_")[0] for f in paths])
        if len(self.classes) == 0:
            self.classes = ids
        else:
            class_check = np.isin(self.classes, ids)
            if not np.all(class_check):
                msg = "Not all classes in the list are present in the directory. Missing classes: {}".format(
                    np.asarray(self.classes)[~class_check]
                )
                raise RuntimeError(msg)
            class_check = np.isin(ids, self.classes)
            if not np.all(class_check):
                logging.basicConfig(format="%(message)s", level=logging.INFO)
                logging.info(
                    "Not all classes in the directory are present in the "
                    "classes list. Missing classes: %s. They will be ignored.",
                    (np.asarray(ids)[~class_check]),
                )

        paths = [
            Path(datapath) / p
            for p in paths
            for c in self.classes
            if c in p.split("_")[0]
        ]
        if self.dataset_size is not None:
            paths = paths[: self.dataset_size]

        if self.transformations is None:
            self.dataset = DiskDataset(paths=paths, datatype=datatype)
        else:
            self.dataset = self.process(paths=paths, datatype=datatype)

    def process(self, paths: list[str], datatype: str):
        """
        Process the loaded data with the specified transformations.

        Args:
            paths (list[str]): List of file paths to the data.
            datatype (str): Type of data being processed.

        Returns:
            DiskDataset: Processed dataset object.

        Raises:
            RuntimeError: If no transformations were provided.
        """
        if self.transformations is None:
            msg = "No processing to do as no transformations were provided."
            raise RuntimeError(msg)
        transforms = list(self.transformations)
        rescale = 0
        normalise = False
        if "normalise" in transforms:
            normalise = True
            transforms.remove("normalise")

        gaussianblur = False
        if "gaussianblur" in transforms:
            gaussianblur = True
            transforms.remove("gaussianblur")

        shiftmin = False
        if "shiftmin" in transforms:
            shiftmin = True
            transforms.remove("shiftmin")

        for i in transforms:
            if i.startswith("rescale"):
                transforms.remove(i)
                rescale = int(float(i.split("=")[-1]))

        if len(transforms) > 0:
            msg = f"The following transformations are not supported: {transforms}"
            raise RuntimeError(msg)

        return DiskDataset(
            paths=paths,
            datatype=datatype,
            rescale=rescale,
            normalise=normalise,
            gaussianblur=gaussianblur,
            shiftmin=shiftmin,
        )

    def get_loader(
        self,
        batch_size: int,
        split_size: float | None = None,
        no_val_drop: bool = False,
    ):
        """
        Retrieve the data loader.

        Args:
            batch_size (int): The batch size for the data loader.
            split_size (float | None, optional): The percentage of data to be used for validation set.
                If None, the entire dataset will be used for training. Defaults to None.
            no_val_drop (bool, optional): If True, the last batch of validation data will not be dropped if it is smaller than batch size. Defaults to False.

        Returns:
            DataLoader or Tuple[DataLoader, DataLoader]: The data loader(s) for testing or training/validation, according to whether training is True or False.

        Raises:
            RuntimeError: If split_size is None and the method is called for training.
            RuntimeError: If the train and validation sets are smaller than 2 samples.

        """
        if self.training:
            if split_size is None:
                msg = "Split size must be provided for training. "
                raise RuntimeError(msg)
            # split into train / val sets
            idx = np.random.permutation(len(self.dataset))
            if split_size < 1:
                split_size = split_size * 100

            s = int(np.ceil(len(self.dataset) * int(split_size) / 100))
            if s < 2:
                msg = "Train and validation sets must be larger than 1 sample, train: {}, val: {}.".format(
                    len(idx[:-s]), len(idx[-s:])
                )
                raise RuntimeError(msg)
            train_data = Subset(self.dataset, indices=idx[:-s])
            val_data = Subset(self.dataset, indices=idx[-s:])

            loader_train = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=True,
            )
            loader_val = DataLoader(
                val_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=(not no_val_drop),
            )
            return loader_train, loader_val

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


class DiskDataset(AbstractDataset):
    """
    A dataset class for loading data from disk.

    Args:
        paths (list[str]): List of file paths.
        datatype (str, optional): Type of data to load. Defaults to "npy".
        rescale (int, optional): Rescale factor for the data. Defaults to 0.
        shiftmin (bool, optional): Whether to shift the minimum value of the data. Defaults to False.
        gaussianblur (bool, optional): Whether to apply Gaussian blur to the data. Defaults to False.
        normalise (bool, optional): Whether to normalise the data. Defaults to False.
        input_transform (typing.Any, optional): Additional input transformation. Defaults to None.
    """

    def __init__(
        self,
        paths: list[str],
        datatype: str = "npy",
        rescale: int = 0,
        shiftmin: bool = False,
        gaussianblur: bool = False,
        normalise: bool = False,
        input_transform: typing.Any = None,
    ) -> None:
        self.paths = paths
        self.rescale = rescale
        self.normalise = normalise
        self.gaussianblur = gaussianblur
        self.transform = input_transform
        self.datatype = datatype
        self.shiftmin = shiftmin

    def __len__(self):
        return len(self.paths)

    def dim(self):
        return len(np.array(self.read(self.paths[0])).shape)

    def __getitem__(self, item):
        filename = self.paths[item]

        data = np.array(self.read(filename))
        x = self.transformation(data)

        # ground truth
        y = Path(filename).name.split("_")[0]

        return x, y

    def read(self, filename):
        """
        Read data from file.

        Args:
            filename (str): File path.

        Returns:
            np.ndarray: Loaded data.

        Raises:
            RuntimeError: If the data type is not supported. Currently supported: .mrc, .npy
        """
        if self.datatype == "npy":
            return np.load(filename)

        if self.datatype == "mrc":
            try:
                with mrcfile.open(filename) as f:
                    return np.array(f.data)
            except ValueError as exc:
                msg = f"File {filename} is corrupted."
                raise ValueError(msg) from exc

        else:
            msg = "Currently we only support mrcfile and numpy arrays."
            raise RuntimeError(msg)

    def transformation(self, x):
        """
        Apply transformations to the input data.

        Args:
            x (np.ndarray): Input data.

        Returns:
            torch.Tensor: Transformed data.
        """
        if self.rescale:
            x = np.asarray(x, dtype=np.float32)
            sh = tuple([self.rescale / s for s in x.shape])
            x = zoom(x, sh)

        # convert numpy to torch tensor
        x = torch.Tensor(x)

        # unsqueeze adds a dimension for batch processing the data
        x = x.unsqueeze(0)

        if self.shiftmin:
            x = (x - x.min()) / (x.max() - x.min())

        if self.gaussianblur:
            T = transforms.GaussianBlur(3, sigma=(0.08, 10.0))
            x = T(x)

        if self.normalise:
            T = transforms.Normalize(0, 1, inplace=False)
            x = T(x)

        if self.transform:
            x = self.transform(x)
        return x

    def augment(self, augment):
        raise NotImplementedError
