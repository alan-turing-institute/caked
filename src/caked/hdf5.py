from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


class HDF5DataStore:
    def __init__(
        self,
        save_path: str | Path,
        use_temp_dir: bool = True,
        cache: LRUCache | None = None,
        batch_size: int = 10,
        cache_size: int = 5,
    ):
        """
        Object to store data in HDF5 format. If use_temp_dir is True, the file is saved
        in a temporary directory and deleted when the object is deleted. This is useful
        for temporary storage of data. If use_temp_dir is False, the file is
        saved in the save_path provided. The file is not deleted when the object is deleted.

        :param save_path: (str) path to save the file
        :param cache: (LRUCache) cache object to store data
        :param use_temp_dir: (bool) whether to use a temporary directory
        :param batch_size: (int) number of items to write to the file before closing
        :param cache_size: (int) size of the cache in GB


        """
        save_path = Path(save_path)
        if use_temp_dir:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir: Path | None = Path(self.temp_dir_obj.name)
            self.save_path = self.temp_dir.joinpath(save_path.name)
        else:
            self.save_path = Path(save_path)
            self.temp_dir = None

        self.batch_size = batch_size
        self.counter = 0
        self.file = None
        if cache is None:
            self.cache = LRUCache(cache_size)
        else:
            self.cache = cache

    def open(self, mode: str = "a"):
        if self.file is None:
            self.file = h5py.File(self.save_path, mode)

    def close(self):
        if self.file is not None:
            self.file.close()  # type: ignore[unreachable]
            self.file = None

    def __del__(self):
        self.close()
        if self.temp_dir is not None:
            self.temp_dir_obj.cleanup()

    def __getitem__(self, key: str):
        with h5py.File(self.save_path, "r") as f:
            return np.array(f[key])

    def __iter__(self):
        with h5py.File(self.save_path, "r") as f:
            yield from f

    def get(self, key: str, default=None, to_torch: bool = False):
        try:
            if key in self.cache:
                return self.cache.get(key)
            with h5py.File(self.save_path, "r") as f:
                if to_torch:
                    arr = torch.from_numpy(np.array(f[key])).clone().detach()

                else:
                    arr = np.array(f[key])
                self.cache.put(key, arr)
                return arr
        except KeyError:
            return default

    def __len__(self):
        return len(self.keys())

    def add_array(
        self, array: np.ndarray, dataset_name: str, compression: str = "gzip"
    ) -> str:
        if self.check_name_in_store(dataset_name):
            dataset_name = self._add_number_to_dataset_name(dataset_name)
        with h5py.File(self.save_path, "a") as f:
            f.create_dataset(
                dataset_name, data=array, compression=compression, chunks=True
            )

        return dataset_name

    def save(self, array_list: list[np.ndarray]):
        for i, array in enumerate(array_list):
            self.add_array(array, f"array_{i}")

    def check_name_in_store(self, dataset_name: str):
        if not self.save_path.exists():
            return False
        with h5py.File(self.save_path, "r") as f:
            return dataset_name in f

    def _add_number_to_dataset_name(self, dataset_name: str, delimiter: str = "--"):
        # add a number to the end of the dataset name, take the last number and increment it
        existing_names = [name for name in self.keys() if dataset_name in name]
        last_number = (
            max(
                [
                    int(name.split(delimiter)[0])
                    for name in existing_names
                    if delimiter in name
                ]
            )
            if len(existing_names) > 1
            else 0
        )

        # dataset_name = dataset_name.split(delimiter)[0:-1]
        return f"{last_number+1}{delimiter}{dataset_name}"

    def keys(self):
        with h5py.File(self.save_path, "r") as f:
            return list(f.keys())

    def values(self, to_torch: bool = False):
        with h5py.File(self.save_path, "r") as f:
            for key in f:
                if to_torch:
                    yield torch.from_numpy(np.array(f[key]))
                else:
                    yield np.array(f[key])


class LRUCache:
    def __init__(self, max_memory_gb: int):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.cache: OrderedDict = OrderedDict()
        self.current_memory_usage = 0

    def get_memory_usage(self, obj) -> int:
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        return 0

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value):
        if key in self.cache:
            self.current_memory_usage -= self.get_memory_usage(self.cache[key])
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.current_memory_usage += self.get_memory_usage(value)
        self.evict_if_needed()

    def evict_if_needed(self):
        while self.current_memory_usage > self.max_memory_bytes:
            _, evicted_value = self.cache.popitem(last=False)
            self.current_memory_usage -= self.get_memory_usage(evicted_value)

    def __contains__(self, key):
        return key in self.cache
