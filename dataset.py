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

from pysemseg.datasets import SegmentationDataset
from pysemseg.transforms import Compose, Resize

from utils import PanSharpenLoader, MSLoader, PANLoader


NADIR_GROUPING = {
    'Nadir': [0, 25],
    'Off-nadir': [26, 40],
    'Very Off-nadir': [41, 55]
}


LOADERS = {
    'MS': MSLoader(),
    'PAN': PANLoader(),
    'Pan-Sharpen': PanSharpenLoader()
}

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


def parse_image_data(root_dir, cache, nadir, image_types):
    if cache:
        polygon_data = _maybe_load_polygon_data(root_dir)
    if not polygon_data:
        polygon_data = parse_polygon_files(root_dir + '/summaryData')
    if cache:
        _store_polygon_data(polygon_data, root_dir)
    images_data = []
    for filepath in glob.glob(root_dir + '/*/Pan-Sharpen/*.tif'):
        image_id = re.match('Pan-Sharpen_(.+)[.]tif', os.path.basename(filepath)).groups()[0]
        nadir_angle = int(
            re.match('.+[_]nadir([\d]{1,2}).+', image_id).groups()[0])
        if nadir is not None:
            low, high = NADIR_GROUPING[nadir]
            if not (low <= nadir_angle <= high):
                continue
        images_data.append({
            'image_id': image_id,
            'image_filepaths': {
                image_type: filepath.replace('Pan-Sharpen', image_type)
                for image_type in image_types
            },
            'angle': nadir_angle,
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


def _shuffle_fixed_seed(items, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(items)
    np.random.set_state(state)


class SpacenetOffNadirDataset(SegmentationDataset):
    def __init__(
            self, data_dir, mode, val_ratio=0.1, cache=True, nadir=None,
            image_types=['Pan-Sharped'], size=None):
        super().__init__()
        self.image_types = image_types
        if len(image_types) > 1:
            assert size is not None, "Specify size for the concatenated images"
        self.size = size
        self.image_data = parse_image_data(
            data_dir, cache=cache, nadir=nadir,
            image_types=self.image_types)
        _shuffle_fixed_seed(self.image_data, 1021)
        val_start_index = int(val_ratio * len(self.image_data))
        if mode == 'train':
            self.image_data = self.image_data[:-val_start_index]
        else:
            self.image_data = self.image_data[-val_start_index:]

    def __getitem__(self, index):
        image_data = self.image_data[index]

        images = []
        for image_type in self.image_types:
            image = LOADERS[image_type](
                image_data['image_filepaths'][image_type])
            image = Resize(self.size)(image)
            images.append(image)
        image = np.concatenate(images, axis=2)
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
