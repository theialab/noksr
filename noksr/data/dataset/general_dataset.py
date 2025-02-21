import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from enum import Enum
from numpy.random import RandomState
import multiprocessing
from omegaconf import DictConfig, ListConfig
from pycg import exp


class GeneralDataset(Dataset):
    """ Only used for Carla dataset """
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, cfg, split):
        pass

class DatasetSpec(Enum):
    SCENE_NAME = 100
    SHAPE_NAME = 0
    INPUT_PC = 200
    TARGET_NORMAL = 300
    INPUT_COLOR = 400
    INPUT_SENSOR_POS = 500
    GT_DENSE_PC = 600
    GT_DENSE_NORMAL = 700
    GT_DENSE_COLOR = 800
    GT_MESH = 900
    GT_MESH_SOUP = 1000
    GT_ONET_SAMPLE = 1100
    GT_GEOMETRY = 1200
    DATASET_CFG = 1300

class RandomSafeDataset(Dataset):
    """
    A dataset class that provides a deterministic random seed.
    However, in order to have consistent validation set, we need to set is_val=True for validation/test sets.
    Usage: First, inherent this class.
           Then, at the beginning of your get_item call, get an rng;
           Last, use this rng as the random state for your program.
    """

    def __init__(self, cfg, split):
        self._seed = cfg.global_train_seed
        self._is_val = split in ['val', 'test']
        self.skip_on_error = False
        if not self._is_val:
            self._manager = multiprocessing.Manager()
            self._read_count = self._manager.dict()
            self._rc_lock = multiprocessing.Lock()

    def get_rng(self, idx):
        if self._is_val:
            return RandomState(self._seed)
        with self._rc_lock:
            if idx not in self._read_count:
                self._read_count[idx] = 0
            rng = RandomState(exp.deterministic_hash((idx, self._read_count[idx], self._seed)))
            self._read_count[idx] += 1
        return rng

    def sanitize_specs(self, old_spec, available_spec):
        old_spec = set(old_spec)
        available_spec = set(available_spec)
        for os in old_spec:
            assert isinstance(os, DatasetSpec)
        new_spec = old_spec.intersection(available_spec)
        # lack_spec = old_spec.difference(new_spec)
        # if len(lack_spec) > 0:
        #     exp.logger.warning(f"Lack spec {lack_spec}.")
        return new_spec

    def _get_item(self, data_id, rng):
        raise NotImplementedError

    def __getitem__(self, data_id):
        rng = self.get_rng(data_id)
        if self.skip_on_error:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))
            except Exception:
                # Just return a random other item.
                exp.logger.warning(f"Get item {data_id} error, but handled.")
                return self.__getitem__(rng.randint(0, len(self) - 1))
        else:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))

