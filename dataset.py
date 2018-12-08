import os
import glob
import tifffile
import numpy as np
import csv
import cv2
from collections import defaultdict
from geomet import wkt
import re
import pickle

from torchvision.transforms import Normalize

from pysemseg.datasets import SegmentationDataset
from pysemseg import transforms


class TiffFileLoader:
    def __call__(self, image_path):
        image = tifffile.imread(image_path)
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = image.transpose((1, 2, 0))


def parse_polygon_files(summary_data_dir):
    polygons = defaultdict(list)
    for csv_filepath in glob.glob(summary_data_dir + '/*.csv'):
        with open(csv_filepath, 'r') as csv_file:
            polygons_raw_data = csv.DictReader(csv_file)
            for p in polygons_raw_data:
                polygons[p['ImageId']].append(wkt.loads(p['PolygonWKT_Pix']))
    return polygons


def _maybe_load_polygon_data(root_dir):
    if os.path.exists(os.path.join(root_dir, '.cache', 'polygons')):
        with open(os.path.join(root_dir, '.cache', 'polygons'), 'rb') as f:
            return  pickle.load(f)


def _store_polygon_data(polygon_data, root_dir):
    os.makedirs(os.path.join(root_dir, '.cache'), exist_ok=True)
    with open(os.path.join(root_dir, '.cache', 'polygons'), 'wb') as f:
        pickle.dump(polygon_data, f)


def parse_image_data(root_dir, cache):
    if cache:
        polygon_data = _maybe_load_polygon_data(root_dir)
    if not polygon_data:
        polygon_data = parse_polygon_files(root_dir + '/summaryData')
    if cache:
        _store_polygon_data(polygon_data, root_dir)
    images_data = []
    for filepath in glob.glob(root_dir + '/*/Pan-Sharpen/*.tif'):
        image_id = re.match('Pan-Sharpen_(.+)[.]tif', os.path.basename(filepath)).groups()[0]
        images_data.append({
            'image_id': image_id,
            'image_filepath': filepath,
            'polygons': polygon_data[image_id]
        })
    return images_data


def create_mask(shape, polygons):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        assert polygon['type'] == 'Polygon'
        for i, poly in enumerate(polygon['coordinates']):
            if i == 0:
                color = (1, 1, 1)
            else:
                color = (0, 0, 0)
            mask = cv2.fillPoly(
                mask, [np.array(poly).astype(np.int32)], color=color)
    return mask


def create_distance_transform(mask):
    dt_mask_pos = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dt_mask_neg = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3) * -1
    dt_mask = np.where(mask == 255, dt_mask_pos, dt_mask_neg)
    return dt_mask


class SpacenetOffNadirDataset(SegmentationDataset):
    def __init__(self, data_dir, mode, val_ratio=0.05, cache=True):
        super().__init__()
        self.image_data = parse_image_data(data_dir, cache=cache)
        np.random.shuffle(self.image_data)
        val_start_index = int(val_ratio * len(self.image_data))
        if mode == 'train':
            self.image_data = self.image_data[:-val_start_index]
        else:
            self.image_data = self.image_data[-val_start_index:]


    def __getitem__(self, index):
        image_data = self.image_data[index]

        image = tifffile.imread(image_data['image_filepath'])
        image = image / np.iinfo(np.uint16).max
        if image.shape[0] == 4:
            image = image.transpose((1, 2, 0))
        mask = create_mask(image.shape[:2], image_data['polygons'])

        return image_data['image_id'], image, mask

    @property
    def number_of_classes(self):
        return 2

    @property
    def in_channels(self):
        return 4

    def __len__(self):
        return len(self.image_data)


class SpaceNetTransform:
    def __init__(self, mode):
        self.mode = mode
        self.joint_augmentations = transforms.Compose([
            transforms.RandomCropFixedSize((512, 512))
        ])
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(
                mean=[0.00931214, 0.01140157, 0.01378472, 0.02853437],
                std=[0.00012771, 0.00014903, 0.00020968, 0.00038597]
            )
        ])

    def __call__(self, image, target):
        if self.mode == 'train':
            image, target = self.joint_augmentations(image, target)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target
