from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from caked.dataloader import DiskDataLoader, DiskDataset

ORIG_DIR = Path.cwd()

TEST_CORRUPT = Path(__file__).parent / "corrupt.mrc"
DISK_PIPELINE = "disk"
DATASET_SIZE_ALL = None
DATASET_SIZE_SOME = 3
DISK_CLASSES_FULL_MRC = ["1b23", "1dfo", "1dkg", "1e3p"]
DISK_CLASSES_SOME_MRC = ["1b23", "1dkg"]
DISK_CLASSES_MISSING_MRC = ["2b3a", "1b23"]
DISK_CLASSES_FULL_NPY = ["2", "5", "a", "d", "e", "i", "j", "l", "s", "u", "v", "x"]
DISK_CLASSES_SOME_NPY = ["2", "5"]
DISK_CLASSES_MISSING_NPY = ["2", "a", "1"]

DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"
DATATYPE_NPY = "npy"
TRANSFORM_ALL = ["normalise", "gaussianblur", "shiftmin"]
TRANSFORM_ALL_RESCALE = ["normalise", "gaussianblur", "shiftmin", "rescale=0"]
TRANSFORM_SOME = ["normalise", "gaussianblur"]
TRANSFORM_RESCALE = ["rescale=32"]


def test_class_instantiation():
    """
    Test the instantiation of the DiskDataLoader class.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_SOME_MRC,
        dataset_size=DATASET_SIZE_SOME,
        save_to_disk=False,
        training=True,
    )
    assert isinstance(test_loader, DiskDataLoader)
    assert test_loader.pipeline == DISK_PIPELINE


def test_dataset_instantiation_mrc(test_data_mrc_dir):
    """
    Test case for instantiating a DiskDataset with MRC data.
    """
    test_dataset = DiskDataset(paths=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_dataset, DiskDataset)


def test_dataset_instantiation_npy(test_data_npy_dir):
    """
    Test case for instantiating a DiskDataset with npy datatype.
    """
    test_dataset = DiskDataset(paths=test_data_npy_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_dataset, DiskDataset)


def test_load_dataset_no_classes(test_data_mrc_dir):
    """
    Test case for loading dataset without specifying classes.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_NONE, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_MRC))


def test_load_dataset_all_classes_mrc(test_data_mrc_dir):
    """
    Test case for loading a dataset with all classes using DiskDataLoader.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_MRC))


def test_load_dataset_all_classes_npy(test_data_npy_dir):
    """
    Test case for loading a dataset with all classes using npy files.

    This test creates a DiskDataLoader object with a specified pipeline, classes, and dataset size.
    It then loads the dataset from a specified datapath using npy files.
    The test asserts that the dataset is an instance of DiskDataset,
    the number of classes in the loader matches the number of classes in the specified classes list,
    and that each class in the loader matches the corresponding class in the specified classes list.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=test_data_npy_dir, datatype=DATATYPE_NPY)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_NPY)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_NPY))


def test_load_dataset_some_classes(test_data_mrc_dir):
    """
    Test case for loading a dataset with some specific classes using DiskDataLoader.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_SOME_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_SOME_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_SOME_MRC))


def test_load_dataset_missing_class(test_data_mrc_dir):
    """
    Test case for loading dataset with missing classes.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_MISSING_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    with pytest.raises(Exception, match=r".*Missing classes: .*"):
        test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)


def test_one_image(test_data_mrc_dir):
    """
    Test case for loading one image using DiskDataLoader.

    This test case verifies that the loaded image is of the correct class and type.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_NONE, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    test_dataset = test_loader.dataset
    test_item_image, test_item_name = test_dataset.__getitem__(1)
    assert test_item_name in DISK_CLASSES_FULL_MRC
    assert isinstance(test_item_image, torch.Tensor)


def test_get_loader_training_false(test_data_mrc_dir):
    """
    Test case for the `get_loader` method of the `DiskDataLoader` class when `training` is set to False.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=False,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    torch_loader = test_loader.get_loader(batch_size=64)
    assert isinstance(torch_loader, torch.utils.data.DataLoader)


def test_get_loader_training_true(test_data_mrc_dir):
    """
    Test case for the `get_loader` method of the `DiskDataLoader` class when training is set to True.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    torch_loader_train, torch_loader_val = test_loader.get_loader(
        split_size=0.8, batch_size=64
    )
    assert isinstance(torch_loader_train, torch.utils.data.DataLoader)
    assert isinstance(torch_loader_val, torch.utils.data.DataLoader)


def test_get_loader_training_fail(test_data_mrc_dir):
    """
    Test case for the `get_loader` method of the `DiskDataLoader` class when training fails.

    This test case verifies that an exception is raised when the dataset size is smaller than the split size.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    with pytest.raises(Exception, match=r".* sets must be larger than .*"):
        torch_loader_train, torch_loader_val = test_loader.get_loader(
            split_size=1, batch_size=64
        )


def test_processing_data_all_transforms(test_data_mrc_dir):
    """
    Test the processing of data with all transforms applied.

    This function creates a DiskDataLoader object with a specified pipeline, classes, dataset size,
    training flag, and all available transformations. It then loads the data from a specified datapath and datatype.
    The function asserts that the dataset has the expected normalization, shiftmin, and gaussianblur
    transforms applied. It also checks the dimensions of the loaded image and asserts that the label
    is one of the expected classes.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_ALL,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert test_loader.dataset.normalise
    assert test_loader.dataset.shiftmin
    assert test_loader.dataset.gaussianblur
    image, label = next(iter(test_loader.dataset))
    image = np.squeeze(image.cpu().numpy())
    assert len(image[0]) == len(image[1]) == len(image[2])
    assert label in DISK_CLASSES_FULL_MRC


def test_processing_data_some_transforms_npy(test_data_npy_dir):
    """
    Test case for processing data with some transformations using the DiskDataLoader class.

    This test case verifies that the DiskDataLoader correctly loads and processes data
    with some specified transformations. It checks that the loaded dataset has the expected
    properties and that the transformed images have the correct dimensions.
    """
    test_loader_transf = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_SOME,
    )
    test_loader_none = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader_none.load(datapath=test_data_npy_dir, datatype=DATATYPE_NPY)
    test_loader_transf.load(datapath=test_data_npy_dir, datatype=DATATYPE_NPY)
    assert test_loader_transf.dataset.normalise
    assert not test_loader_transf.dataset.shiftmin
    assert test_loader_transf.dataset.gaussianblur
    image_none, label_none = next(iter(test_loader_none.dataset))
    image_none = np.squeeze(image_none.cpu().numpy())
    assert len(image_none[0]) == len(image_none[1])
    assert label_none in DISK_CLASSES_FULL_NPY
    image_transf, label_transf = next(iter(test_loader_transf.dataset))
    image_transf = np.squeeze(image_transf.cpu().numpy())
    assert len(image_transf[0]) == len(image_transf[1])
    assert label_transf in DISK_CLASSES_FULL_NPY
    assert len(image_none[0]) == len(image_transf[0])
    assert len(image_none[1]) == len(image_transf[1])


def test_processing_data_rescale(test_data_mrc_dir):
    """
    Test the processing of data with rescaling.

    This function creates a DiskDataLoader object with rescaling transformations
    and verifies that the dataset is loaded correctly. It then asserts various
    properties of the dataset and performs additional checks on the loaded data.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_ALL_RESCALE,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert test_loader.dataset.normalise
    assert test_loader.dataset.shiftmin
    assert test_loader.dataset.gaussianblur
    assert test_loader.dataset.rescale == 0
    image, label = next(iter(test_loader.dataset))
    image = np.squeeze(image.cpu().numpy())
    assert len(image[0]) == len(image[1]) == len(image[2])
    assert label in DISK_CLASSES_FULL_MRC

    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_RESCALE,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert not test_loader.dataset.normalise
    assert not test_loader.dataset.shiftmin
    assert not test_loader.dataset.gaussianblur
    assert test_loader.dataset.rescale == 32
    image, label = next(iter(test_loader.dataset))
    image = np.squeeze(image.cpu().numpy())
    assert len(image[0]) == len(image[1]) == len(image[2])
    assert label in DISK_CLASSES_FULL_MRC


def test_processing_after_load(test_data_mrc_dir):
    """
    Test the processing steps after loading data using DiskDataLoader.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=False,
    )
    test_loader.debug = True
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert test_loader.transformations is None
    assert not test_loader.dataset.normalise
    assert not test_loader.dataset.shiftmin
    assert not test_loader.dataset.gaussianblur
    test_loader.transformations = TRANSFORM_ALL_RESCALE
    pre_dataset = test_loader.dataset
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    post_dataset = test_loader.dataset
    assert test_loader.dataset.normalise
    assert test_loader.dataset.shiftmin
    assert test_loader.dataset.gaussianblur
    assert len(post_dataset) == len(pre_dataset)
    pre_image, pre_label = next(iter(pre_dataset))
    post_image, post_label = next(iter(post_dataset))
    assert pre_label == post_label
    assert not torch.equal(pre_image, post_image)


def test_drop_last(test_data_mrc_dir):
    """
    Test the drop_last parameter in the get_loader method of the DiskDataLoader class.
    """
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=test_data_mrc_dir, datatype=DATATYPE_MRC)
    loader_train_true, loader_val_true = test_loader.get_loader(
        split_size=0.7, batch_size=64, no_val_drop=True
    )
    assert loader_train_true.drop_last
    assert not loader_val_true.drop_last
    loader_train_false, loader_val_false = test_loader.get_loader(
        split_size=0.7, batch_size=64, no_val_drop=False
    )
    assert loader_train_false.drop_last
    assert loader_val_false.drop_last


def test_corrupt_mrcfile(test_data_mrc_dir):
    """
    Test that corrupt mrcfiles are not loaded and throw an exception.
    """
    test_dataset = DiskDataset(paths=test_data_mrc_dir, datatype=DATATYPE_MRC)
    assert isinstance(test_dataset, DiskDataset)
    with pytest.raises(Exception, match=r".* corrupted."):
        test_dataset.read(TEST_CORRUPT)
