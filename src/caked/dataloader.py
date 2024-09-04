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
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle, get_mapobjhandle
from scipy.ndimage import zoom
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms

from caked.base import AbstractDataLoader, AbstractDataset, DatasetConfig
from caked.hdf5 import HDF5DataStore
from caked.Transforms.augments import ComposeAugment
from caked.Transforms.transforms import ComposeTransform, DecomposeToSlices, Transforms
from caked.utils import (
    get_sorted_paths,
    process_datasets,
)

try:
    from ccpem_utils.other.utils import set_gpu
except ImportError:

    def set_gpu():
        pass


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
            class_check = np.in1d(self.classes, ids)
            if not np.all(class_check):
                msg = f"Not all classes in the list are present in the directory. Missing classes: {np.asarray(self.classes)[~class_check]}"
                raise RuntimeError(msg)
            class_check = np.in1d(ids, self.classes)
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
                rescale = int(i.split("=")[-1])

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
                msg = f"Train and validation sets must be larger than 1 sample, train: {len(idx[:-s])}, val: {len(idx[-s:])}."
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


class MapDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
        transformations: list[str] | None = None,
        augmentations: list[str] | None = None,
        decompose: bool = True,
    ) -> None:
        """
        DataLoader implementation for loading map data from disk and saving them to a internal HDF5 store.


        """
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training
        self.pipeline = pipeline
        self.transformations = transformations
        self.augmentations = augmentations
        self.decompose = decompose
        self.debug = False
        self.classes = classes

        if self.classes is None:
            self.classes = []
        if self.transformations is None:
            self.transformations = []
        if self.augmentations is None:
            self.augmentations = []

    def load(
        self,
        datapath: str | Path,
        datatype: str,
        label_path: str | Path | None = None,
        weight_path: str | Path | None = None,
        use_gpu: bool = False,
        num_workers: int = 1,
    ) -> None:
        """
        Load the data from the specified path and data type.

        Args:
            datapath (str | Path): The path to the directory containing the data.
            datatype (str): The type of data to load.
            label_path (str | Path, optional): The path to the directory containing the labels. Defaults to None.
            weight_path (str | Path, optional): The path to the directory containing the weights. Defaults to None.
            multi_process (bool, optional): Whether to use multi-processing. Defaults to False.
            use_gpu (bool, optional): Whether to use the GPU. Defaults to False.

        Returns:
            None
        """
        datasets = []

        if use_gpu and num_workers > 1:
            msg = "Cannot use GPU and multi-process at the same time."
            raise ValueError(msg)
        if use_gpu:
            set_gpu()

        datapath = Path(datapath)
        label_path = Path(label_path) if label_path is not None else None
        weight_path = Path(weight_path) if weight_path is not None else None
        map_hdf5_store = HDF5DataStore(datapath.joinpath("raw_map_data.h5"))

        label_hdf5_store = (
            HDF5DataStore(label_path.joinpath("label_data.h5"))
            if label_path is not None
            else None
        )
        weight_hdf5_store = (
            HDF5DataStore(weight_path.joinpath("weight_data.h5"))
            if weight_path is not None
            else None
        )

        paths = get_sorted_paths(datapath, datatype, self.dataset_size)
        label_paths = get_sorted_paths(label_path, datatype, self.dataset_size)
        weight_paths = get_sorted_paths(weight_path, datatype, self.dataset_size)

        if self.dataset_size is not None:
            paths = paths[: self.dataset_size]
            label_paths = (
                label_paths[: self.dataset_size] if label_paths is not None else None
            )
            weight_paths = (
                weight_paths[: self.dataset_size] if weight_paths is not None else None
            )

        if label_paths is not None and len(label_paths) != len(paths):
            msg = "Label paths and data paths do not match."
            raise RuntimeError(msg)

        if weight_paths is not None and len(weight_paths) != len(paths):
            msg = "Weight paths and data paths do not match."
            raise RuntimeError(msg)

        label_paths = label_paths or [None] * len(paths)
        weight_paths = weight_paths or [None] * len(paths)

        # HDF5 store assumes the data is all in one location

        datasets = process_datasets(
            num_workers,
            paths,
            label_paths,
            weight_paths,
            self.transformations,
            self.augmentations,
            self.decompose,
            map_hdf5_store,
            label_hdf5_store,
            weight_hdf5_store,
        )

        self.dataset = ConcatDataset(datasets)

        # TODO: I think this should be removed in favour of user input for classes
        if not self.classes and label_hdf5_store is not None:
            unique_labels = [
                np.unique(label_data) for label_data in label_hdf5_store.values()
            ]
            self.classes = np.unique(np.concatenate(unique_labels).flatten()).tolist()

    def process(self):
        """ """
        raise NotImplementedError()

    def get_hdf5_store(
        self,
    ) -> tuple[HDF5DataStore, HDF5DataStore | None, HDF5DataStore | None]:
        if self.dataset is None:
            msg = "The dataset has not been loaded yet."
            raise RuntimeError(msg)
        return (
            self.dataset.datasets[0].map_hdf5_store,
            self.dataset.datasets[0].label_hdf5_store,
            self.dataset.datasets[0].weight_hdf5_store,
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
                msg = f"Train and validation sets must be larger than 1 sample, train: {len(idx[:-s])}, val: {len(idx[-s:])}."
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
        raise NotImplementedError()


class MapDataset(AbstractDataset):
    def __init__(
        self,
        path: str | Path,
        **kwargs,
    ) -> None:
        """
        A dataset class for loading map data, alongside the corresponding class labels and weights.
        The map data is loaded from the disk and is decomposed into a set of tiles. These tiles are
        then returned when indexing the dataset.

        Args:
            path (Union[str, Path]): The path to the map data.
            label_path (Optional[Union[str, Path]]): The path to the label data. Defaults to None.
            weight_path (Optional[Union[str, Path]]): The path to the weight data. Defaults to None.
            map_hdf5_store (Optional[HDF5DataStore]): The HDF5 store for the map data. Defaults to None.
            label_hdf5_store (Optional[HDF5DataStore]): The HDF5 store for the label data. Defaults to None.
            transforms (Optional[List[str]]): The transformations to apply to the data.
            augments (Optional[List[str]]): The augmentations to apply to the data.
            decompose (bool): Whether to decompose the data into tiles. Defaults to True.
            decompose_kwargs (Optional[Dict[str, int]]): The decomposition parameters. Defaults to None.
            transform_kwargs (Optional[Dict]): The transformation parameters. Defaults to None.


        Attributes:
            data_shape (Optional[Tuple]): The shape of the map data. Defaults to None.
            mapobj (Optional[MapObjHandle]): The map object handle for the map data. Defaults to None.
            label_mapobj (Optional[MapObjHandle]): The map object handle for the label data. Defaults to None.
            weight_mapobj (Optional[MapObjHandle]): The map object handle for the weight data. Defaults to None.
            slices (Optional[List[Tuple]]): The slices of the data. Defaults to None.
            tiles (Optional): The tiles of the data. Defaults to None.
            tiles_count (int): The number of tiles. Defaults to 0.

        """
        config = DatasetConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.path = Path(path)
        self.id = self.path.stem
        self.label_path = (
            Path(config.label_path) if config.label_path is not None else None
        )
        self.weight_path = (
            Path(config.weight_path) if config.weight_path is not None else None
        )

        self.map_hdf5_store: HDF5DataStore | None = config.map_hdf5_store
        self.label_hdf5_store: HDF5DataStore | None = config.label_hdf5_store
        self.weight_hdf5_store: HDF5DataStore | None = config.weight_hdf5_store
        self.slices: list[tuple] | None = None
        self.tiles = None
        self.tiles_count = config.tiles_count
        self.transforms = config.transforms
        self.augments = config.augments
        self.decompose_kwargs = config.decompose_kwargs
        self.transform_kwargs = config.transform_kwargs
        self.decompose = config.decompose
        self.data_shape: tuple | None = None

        self.mapobj: MapObjHandle | None = None
        self.label_mapobj: MapObjHandle | None = None
        self.weight_mapobj: MapObjHandle | None = None

        if self.decompose_kwargs is None:
            self.decompose_kwargs = {"cshape": 32, "margin": 8}

        if self.transform_kwargs is None:
            self.transform_kwargs = {}

        if self.augments is None:
            self.augments = []

        if self.transforms is None:
            self.transforms = []

        if not self.decompose_kwargs.get("step", False):
            self.decompose_kwargs["step"] = self.decompose_kwargs.get("cshape", 1) - (
                2 * self.decompose_kwargs.get("margin")
            )

    def __len__(self):
        if self.tiles_count == 0 and self.decompose:
            self.generate_tile_indicies()
        elif self.tiles_count == 0:
            self.tiles_count = 1

        return self.tiles_count

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # This needs to be changhed to hold where the data is stored

        if (self.slices is None) or (self.tiles is None):
            self.generate_tile_indicies()

        if self.mapobj is None:
            self.load_map_objects()

        map_slice = self.mapobj.data[self.slices[idx]]
        label_slice = (
            self.label_mapobj.data[self.slices[idx]]
            if self.label_mapobj is not None
            else None
        )
        weight_slice = (
            self.weight_mapobj.data[self.slices[idx]]
            if self.weight_mapobj is not None
            else None
        )

        map_tensor = torch.tensor(map_slice)
        label_tensor = torch.tensor(label_slice) if label_slice is not None else None
        weight_tensor = torch.tensor(weight_slice) if weight_slice is not None else None

        if weight_tensor is None and label_tensor is not None:
            weight_tensor = torch.where(
                label_tensor != 0,
                torch.ones_like(label_tensor),
                torch.zeros_like(label_tensor),
            )

        # Ensure weight_tensor has the same shape as map_tensor
        if weight_tensor is not None and weight_tensor.shape == map_tensor.shape:
            # Add weight values to the first dimension of the map tensor
            map_tensor = torch.cat(
                (weight_tensor.unsqueeze(0), map_tensor.unsqueeze(0)), dim=0
            )

        # if the weight tensor is None then I want to create weights tesnor using the label tensor

        self.close_map_objects(self.mapobj, self.label_mapobj, self.weight_mapobj)

        return tuple(
            tensor for tensor in (map_tensor, label_tensor) if tensor is not None
        )

    def load_map_objects(
        self,
    ) -> None:
        self.mapobj = get_mapobjhandle(self.path)
        self.mapobj.all_transforms = True
        if self.label_path is not None:
            if not self.label_path.exists():
                msg = f"Label file {self.label_path} not found."
                raise FileNotFoundError(msg)
            self.label_mapobj = get_mapobjhandle(self.label_path)
            self.label_mapobj.all_transforms = False
        if self.weight_path is not None:
            if not self.weight_path.exists():
                msg = f"Weight file {self.weight_path} not found."
                raise FileNotFoundError(msg)
            self.weight_mapobj = get_mapobjhandle(self.weight_path)
            self.weight_mapobj.all_transforms = False

    def close_map_objects(self, *args):
        for arg in args:
            if arg is not None:
                arg.close()

    def augment(self, close_map_objects) -> None:
        augment_kwargs = self._augment_keywords_builder()
        if len(self.augments) == 0:
            return {}

        self.mapobj, extra_kwargs = ComposeAugment(self.augments)(
            self.mapobj, **augment_kwargs
        )
        augment_kwargs.update(
            extra_kwargs
        )  # update the kwargs with the returned values

        self.label_mapobj = ComposeAugment(self.augments)(
            self.label_mapobj, **augment_kwargs
        )
        self.weight_mapobj = ComposeAugment(self.augments)(
            self.weight_mapobj, **augment_kwargs
        )

        if close_map_objects:
            self.close_map_objects(self.mapobj, self.label_mapobj, self.weight_mapobj)

        return augment_kwargs

    def transform(self, close_map_objects: bool = True):
        """
        Perform the transformations on the map data.

        Note: The final map shape is calculated here,

        Args:
            close_map_objects (bool, optional): Whether to close the map objects after transformation. Defaults to True.

        """
        # TODO: Need to see if same transforms are applied to all map objects, maybe just voxel space normalisation
        if self.mapobj is None:
            self.load_map_objects()
        transform_kwargs = self._transform_keywords_builder()
        if len(self.transforms) == 0:
            self.transform_kwargs = transform_kwargs

        self.transform_kwargs = ComposeTransform(self.transforms)(
            self.mapobj, self.label_mapobj, self.weight_mapobj, **transform_kwargs
        )
        self.get_data_shape(close_map_objects=False)

        if close_map_objects:
            self.close_map_objects(self.mapobj, self.label_mapobj, self.weight_mapobj)

    def get_data_shape(self, close_map_objects: bool = True):
        if self.data_shape is not None:
            return

        if (self.mapobj is None) or (self.mapobj.data) is None:
            self.load_map_objects()
        self.data_shape = self.mapobj.data.shape
        if self.label_mapobj is not None:
            assert (
                self.label_mapobj.data.shape == self.data_shape
            ), f"Map and label shapes do not match for {self.id}."
        if self.weight_mapobj is not None:
            assert (
                self.weight_mapobj.data.shape == self.data_shape
            ), f"Map and weight shapes do not match for {self.id}."

        if close_map_objects:
            self.close_map_objects(self.mapobj, self.label_mapobj, self.weight_mapobj)

    def generate_tile_indicies(self):
        if self.data_shape is None:
            self.get_data_shape()

        decompose = DecomposeToSlices(
            self.data_shape,
            step=self.decompose_kwargs.get("step"),
            cshape=self.decompose_kwargs.get("cshape"),
            margin=self.decompose_kwargs.get("margin"),
        )

        self.slices = decompose.slices
        self.tiles = decompose.tiles
        self.tiles_count = len(self.tiles)

    def _transform_keywords_builder(self):
        keywords = {}
        keywords.update(self.decompose_kwargs)

        for transform in self.transforms:
            if transform == Transforms.MASKCROP.value:
                keywords["mask"] = self.label_mapobj

            if transform == Transforms.NORM.value:
                keywords["ext_dim"] = (0, 0, 0)
                keywords["fill_padding"] = (0, 0, 0)

            if transform == Transforms.VOXNORM.value:
                keywords["vox"] = self.decompose_kwargs.get("vox", 1.0)
                keywords["vox_lim"] = self.decompose_kwargs.get("vox_lim", (0.95, 1.05))

        return keywords

    def _augment_keywords_builder(self):
        keywords = {}
        for augment in self.augments:
            if augment.__class__.__name__ == "RandomRotationAugment":
                keywords["ax"] = self.ax
                keywords["an"] = self.an

        return keywords


class ArrayDataset(AbstractDataset):
    """Class to handle loading of data from hdf5 files, to be handled by a DataLoader"""

    # need to add their own and update the dataset id
    def __init__(
        self,
        dataset_id: str,
        data_array: np.ndarray,
        label_array: np.ndarray | None = None,
        weight_array: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        config = DatasetConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.id = dataset_id
        self.data_array = data_array
        self.label_array = label_array
        self.weight_array = weight_array

        self.slices = config.slices
        self.tiles = config.tiles
        self.tiles_count = config.tiles_count
        self.augments = config.augments
        self.decompose = config.decompose
        self.data_shape: tuple | None = None
        self.decompose_kwargs = config.decompose_kwargs
        self.map_hdf5_store = config.map_hdf5_store
        self.label_hdf5_store = config.label_hdf5_store
        self.weight_hdf5_store = config.weight_hdf5_store
        if self.decompose_kwargs is None:
            self.decompose_kwargs = {"cshape": 64, "margin": 8}

        if not self.decompose_kwargs.get("step", False):
            self.decompose_kwargs["step"] = self.decompose_kwargs.get("cshape", 1) - (
                2 * self.decompose_kwargs.get("margin")
            )

        if self.augments is None:
            self.augments = []

        # create an instance of the map dataset so I can use it's functions using composition
        self.__mapdataset = MapDataset(
            path=self.id,
            # use the attributes from the config object
            **config.__dict__,
        )

    def __len__(self):
        if self.tiles_count == 0 and self.decompose:
            self.generate_tile_indicies()
        elif self.tiles_count == 0:
            self.tiles_count = 1

        return self.tiles_count

    def __getitem__(self, idx):
        if (self.slices is None) or (self.tiles is None):
            self.generate_tile_indicies()

        if self.data_array is None:
            self.get_data()

        data_slice = self.data_array[self.slices[idx]]

        label_slice = (
            self.label_array[self.slices[idx]] if self.label_array is not None else None
        )
        weight_slice = (
            self.weight_array[self.slices[idx]]
            if self.weight_array is not None
            else None
        )
        data_tensor = torch.tensor(data_slice)
        label_tensor = torch.tensor(label_slice) if label_slice is not None else None
        weight_tensor = torch.tensor(weight_slice) if weight_slice is not None else None

        if weight_tensor is None and label_tensor is not None:
            weight_tensor = torch.where(
                label_tensor != 0,
                torch.ones_like(label_tensor),
                torch.zeros_like(label_tensor),
            )

        if weight_tensor is not None and weight_tensor.shape == data_tensor.shape:
            # Add weight values to the first dimension of the map tensor
            data_tensor = torch.cat(
                (weight_tensor.unsqueeze(0), data_tensor.unsqueeze(0)), dim=0
            )

        self.close_data()

        return tuple(
            tensor for tensor in (data_tensor, label_tensor) if tensor is not None
        )

    def get_data(self):
        self.data_array = self.map_hdf5_store.get(self.id + "_map")
        if self.label_hdf5_store is not None:
            self.label_array = self.label_hdf5_store.get(self.id + "_label")
        if self.weight_hdf5_store is not None:
            self.weight_array = self.weight_hdf5_store.get(self.id + "_weight")

    def close_data(self):
        self.data_array = None
        self.label_array = None
        self.weight_array = None

    def _augment_keywords_builder(self):
        return self.__mapdataset._augment_keywords_builder()

    def _transform_keywords_builder(self):
        return self.__mapdataset._transform_keywords_builder()

    # need to do augment
    def transform(self) -> None:
        msg = "Transforms are not supported for ArrayDataset."
        raise NotImplementedError(msg)

    def augment(self) -> None:
        augment_kwargs = self._augment_keywords_builder()
        if len(self.augments) == 0:
            return {}

        self.data_array, extra_kwargs = ComposeAugment(self.augments)(
            self.data_array, **augment_kwargs
        )

        augment_kwargs.update(
            extra_kwargs
        )  # update the kwargs with the returned values
        if self.label_array is not None:
            self.label_array, _ = ComposeAugment(self.augments)(
                self.label_array, **augment_kwargs
            )
        if self.weight_array is not None:
            self.weight_array, _ = ComposeAugment(self.augments)(
                self.weight_array, **augment_kwargs
            )

        return augment_kwargs

    def get_data_shape(self, close_data: bool = True):
        if self.data_shape is not None:
            return

        if self.data_array is None:
            self.get_data()
        self.data_shape = self.data_array.shape
        if self.label_array is not None:
            assert (
                self.label_array.shape == self.data_shape
            ), "Map and label shapes do not match."
        if self.weight_array is not None:
            assert (
                self.weight_array.shape == self.data_shape
            ), "Map and weight shapes do not match."

        if close_data:
            self.close_data()

    def generate_tile_indicies(self):
        if self.data_shape is None:
            self.get_data_shape()

        decompose = DecomposeToSlices(
            self.data_shape,
            step=self.decompose_kwargs.get("step"),
            cshape=self.decompose_kwargs.get("cshape"),
            margin=self.decompose_kwargs.get("margin"),
        )

        self.slices = decompose.slices
        self.tiles = decompose.tiles
        self.tiles_count = len(self.tiles)

    def save_to_store(self, close_data: bool = True):
        self.id = self.map_hdf5_store.add_array(
            self.data_array,
            self.id + "_map",
        )
        self.id = self.id.replace("_map", "")

        if self.label_array is not None:
            self.label_hdf5_store.add_array(
                self.label_array,
                self.id + "_label",
            )
        if self.weight_array is not None:
            self.weight_hdf5_store.add_array(
                self.weight_array,
                self.id + "_weight",
            )
        if close_data:
            self.close_data()
