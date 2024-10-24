from __future__ import annotations

from pathlib import Path

import torch

from caked.dataloader import MapDataLoader, MapDataset
from caked.hdf5 import HDF5DataStore, LRUCache
from caked.utils import add_dataset_to_HDF5, duplicate_and_augment_from_hdf5

ORIG_DIR = Path.cwd()


DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"
VOXNORM = "voxnorm"
NORM = "norm"
MASKCROP = "maskcrop"
PADDING = "padding"
ROTATION = "randrot"
TRANSFORM_ALL = [VOXNORM, NORM, PADDING]
AUGMENT_ALL = [ROTATION]


def test_map_dataloader():
    test_loader = MapDataLoader()

    assert test_loader is not None
    assert isinstance(test_loader, MapDataLoader)


def test_map_dataset(test_data_single_mrc_dir):
    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}"))
    )
    assert test_map_dataset is not None
    assert isinstance(test_map_dataset, MapDataset)


def test_slices(test_data_single_mrc_dir):
    hdf5_store = HDF5DataStore("test.hdf5", cache_size=1)

    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}")),
        transforms=[],
        augments=[],
        map_hdf5_store=hdf5_store,
    )
    test_map_dataset.load_map_objects()

    add_dataset_to_HDF5(
        test_map_dataset.mapobj.data,
        None,
        None,
        "realmap",
        hdf5_store,
    )
    slice_ = test_map_dataset.__getitem__(0)[0]

    assert isinstance(slice_, torch.Tensor)
    assert len(test_map_dataset) == 4
    assert slice_.shape == (2, 32, 32, 32)


def test_transforms(test_data_single_mrc_dir):
    hdf5_store = HDF5DataStore("test.hdf5", cache_size=1)
    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}")),
        map_hdf5_store=hdf5_store,
        transforms=TRANSFORM_ALL,
        augments=[],
    )
    test_map_dataset.load_map_objects()
    test_map_dataset.transform()
    add_dataset_to_HDF5(
        test_map_dataset.mapobj.data,
        None,
        None,
        "realmap",
        hdf5_store,
    )
    slice_ = test_map_dataset.__getitem__(0)[0]

    assert len(test_map_dataset) == 64
    assert slice_.shape == (2, 32, 32, 32)


def test_dataloader_load_to_HDF5_file(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader()
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_map_dataloader.dataset.datasets[0].map_hdf5_store.save_path.exists()


def test_dataloader_load_and_decompose(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader()
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
        cshape=16,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    test_map_dataset = test_map_dataloader.dataset.datasets[0]
    slice_ = test_map_dataset.__getitem__(0)[0]

    assert slice_.shape == (2, 16, 16, 16)


def test_dataloader_load_to_HDF5_file_with_transforms(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_map_dataloader.dataset.datasets[0].map_hdf5_store.save_path.exists()


def test_add_duplicate_dataset_to_dataloader(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    duplicate_and_augment_from_hdf5(
        test_map_dataloader,
        ids=[
            next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem,
            next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem,
        ],
    )
    hdf5_store = test_map_dataloader.dataset.datasets[0].map_hdf5_store

    assert len(hdf5_store.keys()) == 3
    assert "realmap_map" in hdf5_store
    assert "1--realmap_map" in hdf5_store
    assert "2--realmap_map" in hdf5_store


def test_add_duplicate_dataset_to_dataloader_with_augments(
    test_data_single_mrc_temp_dir,
):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )
    duplicate_and_augment_from_hdf5(
        ids=[next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem],
        map_data_loader=test_map_dataloader,
        augmentations=AUGMENT_ALL,
    )
    hdf5_store = test_map_dataloader.dataset.datasets[0].map_hdf5_store
    assert len(hdf5_store.keys()) == 2
    assert "realmap_map" in hdf5_store
    assert "1--realmap_map" in hdf5_store

    assert len(test_map_dataloader.dataset.datasets[0]) == 64
    assert len(test_map_dataloader.dataset.datasets[1]) == 64

    assert len(test_map_dataloader.dataset) == 128


def test_dataloader_load_multi_process(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader()
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
        num_workers=2,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_map_dataloader.dataset.datasets[0].map_hdf5_store.save_path.exists()

    # test_map_dataloader.


def test_lru_cache(test_data_single_mrc_dir):
    cache = LRUCache(1)
    hdf5_store = HDF5DataStore("test.hdf5", cache=cache)

    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}")),
        transforms=[],
        augments=[],
        map_hdf5_store=hdf5_store,
    )
    test_map_dataset.load_map_objects()

    add_dataset_to_HDF5(
        test_map_dataset.mapobj.data,
        None,
        None,
        "realmap",
        hdf5_store,
    )

    assert "realmap_map" not in hdf5_store.cache

    _ = test_map_dataset.__getitem__(0)[0]

    assert "realmap_map" in hdf5_store.cache
