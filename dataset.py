import os
import glob
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
from transforms import Erode


NADIR_GROUPING = {
    'Nadir': [0, 25],
    'Off-nadir': [26, 40],
    'Very Off-nadir': [41, 55]
}


IN_CHANNELS = {
    'MS': 8,
    'PAN': 1,
    'Pan-Sharpen': 4
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


def _make_border(mask):
    eroded = Erode(kernel_size=(3,3), iterations=3)(mask)
    border = mask - eroded
    mask = np.where(border, 2, mask)
    return mask


def create_mask(shape, polygons, make_border=False):
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
    if make_border:
        mask = _make_border(mask)
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
            image_types=['Pan-Sharpen'], make_border=False, size=None):
        super().__init__()
        self.make_border = make_border
        self.image_types = image_types
        if len(image_types) > 1:
            assert size is not None, "Specify size for the concatenated images"
        self.size = tuple(size) if size else None
        self.image_data = parse_image_data(
            data_dir, cache=cache, nadir=nadir,
            image_types=self.image_types)
        _shuffle_fixed_seed(self.image_data, 1021)
        val_start_index = int(val_ratio * len(self.image_data))
        if mode == 'train':
            self.image_data = self.image_data[:-val_start_index]
        else:
            self.image_data = self.image_data[-val_start_index:]
        if nadir is not None:
            low, high = NADIR_GROUPING[nadir]
            self.image_data = [
                item for item in self.image_data
                if low <=  item['angle'] <= high
            ]

    def __getitem__(self, index):
        image_data = self.image_data[index]

        images = []
        for image_type in self.image_types:
            image = LOADERS[image_type](
                image_data['image_filepaths'][image_type])
            if self.size is not None:
                image = Resize(self.size)(image)
            images.append(image)
        image = np.concatenate(images, axis=2)
        mask = create_mask(
            image.shape[:2], image_data['polygons'], self.make_border)

        return image_data['image_id'], image, mask

    @property
    def number_of_classes(self):
        return 2 + int(self.make_border)

    @property
    def in_channels(self):
        return sum([IN_CHANNELS[it] for it in self.image_types])

    def __len__(self):
        return len(self.image_data)


class ProposalDataset(SegmentationDataset):
    def __init__(
            self, data_dir, mode, val_ratio=0.1,
            nadir=None):
        super().__init__()
        self.image_data = []
        for filepath in glob.glob(os.path.join(data_dir, "*")):
            image_id = os.path.basename(filepath)
            nadir_angle = int(
                re.match('.+[_]nadir([\d]{1,2}).+', image_id).groups()[0])
            self.image_data.append({
                'image_id': image_id,
                'filepath': filepath,
                'angle': nadir_angle
            })

        _shuffle_fixed_seed(self.image_data, 1021)
        val_start_index = int(val_ratio * len(self.image_data))
        if mode == 'train':
            self.image_data = self.image_data[:-val_start_index]
        else:
            self.image_data = self.image_data[-val_start_index:]

    def __getitem__(self, index):
        image_data = self.image_data[index]
        with open(image_data['filepath'], 'rb') as f:
            image, segm, mask = pickle.load(f)
        segm = np.expand_dims(segm, axis=-1)
        angle = np.ones_like(segm) * image_data['angle']
        image = np.concatenate(
                (image, segm, angle), axis=2)
        return image_data['image_id'], image, mask


    @property
    def number_of_classes(self):
        return 2

    @property
    def in_channels(self):
        return 15

    def __len__(self):
        return len(self.image_data)
