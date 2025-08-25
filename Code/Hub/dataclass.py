from pathlib import Path
import os
from os.path import join
import re
from torch.utils.data import Dataset
import numpy as np
from skimage import io

from Config import cfg as config
import torch


def read_image(file):
    img = io.imread(file)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def read_file(file):
    if '.npy' in file:
        data = np.load(file)
    elif '.npz' in file:
        data = np.load(file)['data']
    else:
        data = read_image(file)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 255.0
    return data


class Img2ImgDataset:

    def __init__(self, data):
        self.target_path = os.path.join(data['data_folder'], data['target']['path'])
        self.original_path = os.path.join(data['data_folder'], data['original']['path'])
        self.mask_path = os.path.join(data['data_folder'], data['mask']['path'])

        self.target_pattern = data['target']['pattern']
        self.original_pattern = data['original']['pattern']
        self.mask_pattern = data['mask']['pattern']

        self._make_dataset()

    def _make_dataset(self):
        target = re.compile(self.target_pattern)
        orig = re.compile(self.original_pattern)
        mask = re.compile(self.mask_pattern)

        number = re.compile('[0-9]+')

        self.targets = {}
        self.origs = {}
        self.masks = {}

        paths = [self.target_path, self.original_path, self.mask_path]
        patterns = [target, orig, mask]
        data_dicts = [self.targets, self.origs, self.masks]

        for path, pattern, data_dict in zip(paths, patterns, data_dicts):
            for file_name in os.listdir(path):
                if config.dataset.startswith('RITE') or config.dataset.startswith('LES-AV'):
                    n = number.findall(file_name)
                    if pattern.match(file_name) and n:
                        n = int(n[0])
                        data_dict[n] = file_name
                else:
                    if pattern.match(file_name):
                        data_dict[Path(file_name).stem] = file_name


class VesselsDataset(Dataset, Img2ImgDataset):

    def __init__(self, data, transform=None):
        Img2ImgDataset.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]

        r = read_file(join(self.original_path, retino))
        m = read_file(join(self.mask_path, mask))
        v = read_file(join(self.target_path, vessel))

        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]


class VesselsDatasetV2(Dataset):
    def __init__(self, data, transform=None):
        """
        data: dict，包含
            - data_folder: 根路径
            - original: dict {path, pattern}，原图文件夹和匹配规则
            - segmentation: dict {path, pattern}，分割图文件夹和匹配规则
            - mask: dict {path, pattern}，标签文件夹和匹配规则

        transform: 传入的变换函数，接收list输入：[img, seg, mask]，需同步处理
        """
        self.original_path = os.path.join(data['data_folder'], data['original']['path'])
        self.segmentation_path = os.path.join(data['data_folder'], data['segmentation']['path'])
        self.mask_path = os.path.join(data['data_folder'], data['mask']['path'])
        self.vessels_path = os.path.join(data['data_folder'], data['target']['path'])

        self.original_pattern = re.compile(data['original']['pattern'])
        self.segmentation_pattern = re.compile(data['segmentation']['pattern'])
        self.mask_pattern = re.compile(data['mask']['pattern'])
        self.vessels_pattern = re.compile(data['target']['pattern'])

        self._make_dataset()

        self.transform = transform

        self.indices = sorted(self.originals.keys())

    def _make_dataset(self):
        number = re.compile('[0-9]+')

        self.originals = {}
        self.segmentations = {}
        self.masks = {}
        self.vessels = {}

        # 读取文件名，存为字典 key=int 或 str，value=文件名
        for folder_path, pattern, storage_dict in [
            (self.original_path, self.original_pattern, self.originals),
            (self.segmentation_path, self.segmentation_pattern, self.segmentations),
            (self.mask_path, self.mask_pattern, self.masks),
            (self.vessels_path, self.vessels_pattern, self.vessels),
        ]:
            for file_name in os.listdir(folder_path):
                n_search = number.findall(file_name)
                if n_search:
                    key = int(n_search[0])
                    if pattern.match(file_name):
                        storage_dict[key] = file_name

        # 只保留所有四个文件夹都有对应文件的key
        keys = (set(self.originals.keys()) & set(self.segmentations.keys())
                & set(self.masks.keys()) & set(self.vessels.keys()))
        self.originals = {k: self.originals[k] for k in keys}
        self.segmentations = {k: self.segmentations[k] for k in keys}
        self.masks = {k: self.masks[k] for k in keys}
        self.vessels = {k: self.vessels[k] for k in keys}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key = self.indices[idx]
        _idx = idx

        # 读图，确保返回Tensor格式且 shape = (C,H,W)
        original_file = os.path.join(self.original_path, self.originals[key])
        seg_file = os.path.join(self.segmentation_path, self.segmentations[key])
        mask_file = os.path.join(self.mask_path, self.masks[key])
        vessel_file = os.path.join(self.vessels_path, self.vessels[key])

        original = read_file(original_file)  # 期望返回Tensor，(3,H,W)
        segmentation = read_file(seg_file)  # 期望返回Tensor，(3,H,W)
        mask = read_file(mask_file)  # 期望返回Tensor，(1,H,W)
        vessel = read_file(vessel_file)  # 期望返回Tensor，(3,H,W)
        # 拼接输入 (6, H, W)
        input_tensor = torch.cat([original, segmentation], dim=0)

        item = [input_tensor, vessel, mask]

        # 变换时传入list，保证所有同步增强
        if self.transform is not None:
            input_tensor, mask = self.transform([input_tensor, mask])
            # 如果transform不支持直接传list，可以自己写个wrapper保证同步变换

        return [_idx, item]


class Img2ImgDatasetWithSeg:
    def __init__(self, data):
        self.target_path = os.path.join(data['data_folder'], data['target']['path'])
        self.original_path = os.path.join(data['data_folder'], data['original']['path'])
        self.mask_path = os.path.join(data['data_folder'], data['mask']['path'])
        self.segmentation_path = os.path.join(data['data_folder'], data['segmentation']['path'])  # 新增

        self.target_pattern = data['target']['pattern']
        self.original_pattern = data['original']['pattern']
        self.mask_pattern = data['mask']['pattern']
        self.segmentation_pattern = data['segmentation']['pattern']  # 新增

        self._make_dataset()

    def _make_dataset(self):
        target = re.compile(self.target_pattern)
        orig = re.compile(self.original_pattern)
        mask = re.compile(self.mask_pattern)
        seg = re.compile(self.segmentation_pattern)

        number = re.compile('[0-9]+')

        self.targets = {}
        self.origs = {}
        self.masks = {}
        self.segs = {}  # 新增

        paths = [self.target_path, self.original_path, self.mask_path, self.segmentation_path]
        patterns = [target, orig, mask, seg]
        data_dicts = [self.targets, self.origs, self.masks, self.segs]

        for path, pattern, data_dict in zip(paths, patterns, data_dicts):
            for file_name in os.listdir(path):
                if config.dataset.startswith('RITE') or config.dataset.startswith('LES-AV'):
                    n = number.findall(file_name)
                    if pattern.match(file_name) and n:
                        n = int(n[0])
                        data_dict[n] = file_name
                else:
                    if pattern.match(file_name):
                        data_dict[Path(file_name).stem] = file_name


class VesselsDatasetWithSeg(Dataset, Img2ImgDatasetWithSeg):
    def __init__(self, data, transform=None):
        Img2ImgDatasetWithSeg.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        self.segmentations = self.segs
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]
        seg = self.segmentations[_index]

        r = read_file(os.path.join(self.original_path, retino))        # 原始图像
        m = read_file(os.path.join(self.mask_path, mask))              # mask
        v = read_file(os.path.join(self.target_path, vessel))          # 目标图像
        s = read_file(os.path.join(self.segmentation_path, seg))       # segmentation（3通道）

        # 确保 segmentation 是三通道
        if s.ndim == 2:
            s = np.expand_dims(s, axis=-1)
            s = np.repeat(s, 3, axis=-1)

        # 将 segmentation concat 到 retino (通道维度)
        r_concat = np.concatenate([r, s], axis=-1)  # shape: (H, W, C+3)

        item = [r_concat, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]
