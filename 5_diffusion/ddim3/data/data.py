"""Datasets for SRHD training (simulated bundles) and real-scene evaluation (PNG maps).

Both datasets return tensors already in the [-1, 1] range the UNet expects, so no
torchvision transform pipeline is needed:

    binary map in {0,1}  ->  2*x - 1
    8-bit PNG in [0,255] ->  2*(x/255) - 1

which is exactly what `ToTensor() + Normalize((0.5,), (0.5,))` did, minus the
numpy -> PIL -> numpy -> tensor round-trip that dominated the old dataloader.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_KEYS = ("image_ob_obs", "image_aug2_line_", "image_gt_obs", "image_gt_wall")


def _to_pm1(array):
    """HxW float array in [0,1] -> (1,H,W) tensor in [-1,1]."""
    return torch.from_numpy(array).unsqueeze(0).mul_(2.).sub_(1.)


class NumpyDataset(Dataset):
    """Augmented simulation bundles written by 6_simulator.

    Each `.npy` holds a dict of bit-packed 128x128 maps.  One sample is ~8 KB
    packed, so the whole training set is cached in RAM by default (`cache=True`):
    unpacking is ~40x cheaper than re-reading and re-allocating the pickle every
    step, and it removes the file-system round trip from the training loop.
    """

    def __init__(self, folder_path, H, W, cache=True, transform=None):
        self.folder_path = folder_path
        self.file_list = sorted(f for f in os.listdir(folder_path) if f.endswith('.npy'))
        if not self.file_list:
            raise FileNotFoundError(f"{folder_path}: no .npy bundles "
                                    "(run 6_simulator/run_sim.sh then run_augment.sh)")
        self.H, self.W = H, W
        self.transform = transform  # kept for API compatibility; normally None
        self.cache = cache
        self._packed = [None] * len(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def _load_packed(self, idx):
        if self._packed[idx] is not None:
            return self._packed[idx]
        raw = np.load(os.path.join(self.folder_path, self.file_list[idx]), allow_pickle=True).item()
        packed = tuple(np.ascontiguousarray(raw[k]) for k in _KEYS)
        if self.cache:
            self._packed[idx] = packed
        return packed

    def __getitem__(self, idx):
        ob_obs, ob_wall, gt_obs, gt_wall = (
            np.unpackbits(p).reshape(self.H, self.W).astype(np.float32)
            for p in self._load_packed(idx))
        observation = np.maximum(ob_obs, ob_wall)          # union of the two observed layers
        if self.transform is not None:                      # legacy path
            from PIL import Image as _Image
            out = [self.transform(_Image.fromarray(a)) for a in (observation, gt_obs, gt_wall)]
            return out[0], out[1], out[2]
        return _to_pm1(observation), _to_pm1(gt_obs), _to_pm1(gt_wall)


class CustomDataset(Dataset):
    """Real-scene evaluation inputs: `<name>_<case>_<ratio>.png` maps from 4_inversion.

    The trajectory map is looked up by *filename*, never by the index of a second
    `os.listdir()` -- the two listings are not guaranteed to line up and pairing a
    case with another case's trajectory silently corrupts the evaluation.
    """

    def __init__(self, path_obs, path_traj, path_gt_obs, path_gt_wall, transform=None):
        self.path_obs, self.path_traj = path_obs, path_traj
        self.file_obs = sorted(f for f in os.listdir(path_obs) if f.endswith('.png'))
        if not self.file_obs:
            raise FileNotFoundError(f"{path_obs}: no observation maps "
                                    "(run 4_inversion/run.py --step layout --export)")
        missing = [f for f in self.file_obs if not os.path.exists(os.path.join(path_traj, f))]
        if missing:
            raise FileNotFoundError(f"{path_traj}: missing trajectory maps for {missing[:5]}"
                                    f"{' ...' if len(missing) > 5 else ''}")
        self.gt_obs_dict = self._index_gt(path_gt_obs)
        self.gt_wall_dict = self._index_gt(path_gt_wall)
        self.transform = transform

    @staticmethod
    def _index_gt(folder):
        return {int(f.split('_')[0]): os.path.join(folder, f)
                for f in os.listdir(folder) if f.endswith('.png')}

    def __len__(self):
        return len(self.file_obs)

    @staticmethod
    def _read(path):
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.
        return _to_pm1(array)

    def __getitem__(self, idx):
        name = self.file_obs[idx]
        case_number = int(name.split(".")[0].split("_")[1])
        ratio_choice = int(name.split(".")[0].split("_")[2])
        image_obs = self._read(os.path.join(self.path_obs, name))
        image_traj = self._read(os.path.join(self.path_traj, name))
        image_gt_obs = self._read(self.gt_obs_dict[case_number])
        image_gt_wall = self._read(self.gt_wall_dict[case_number])
        return image_obs, image_traj, image_gt_obs, image_gt_wall, case_number, ratio_choice
