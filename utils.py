import rasterio
import numpy as np


PAN_RGB_THRESHOLD = 3000

def compute_mean_std(dataset):
    samples = []
    count = 0
    x = np.zeros(dataset[0][1].shape[2], dtype=np.float32)
    x2 = np.zeros(dataset[0][1].shape[2], dtype=np.float32)
    for i in range(len(dataset)):
        image = dataset[i][1]
        x2 += np.sum(image ** 2, axis=(0, 1))
        x += np.sum(image, axis=(0, 1))
        count += image.shape[0] * image.shape[1]
    mean = x / count
    std = x2 / count - mean ** 2
    return mean, std


class RasterIOLoader():
    def __call__(self, image_path):
        im_reader = rasterio.open(image_path)
        img = np.empty((
            im_reader.height,
            im_reader.width,
            im_reader.count
        ))
        for band in range(im_reader.count):
            img[:, :, band] = im_reader.read(band+1)
        return img


class PanSharpenLoader:
    def __init__(self):
        self.rasterio_loader = RasterIOLoader()
    def __call__(self, image_path):
        image = self.rasterio_loader(image_path)
        image[:, :, :3] = (
            np.clip(image[:, :, :3], 0, PAN_RGB_THRESHOLD) / PAN_RGB_THRESHOLD
        )
        image[:, :, 3] /= np.iinfo(np.uint16).max
        return image


class PANLoader:
    def __init__(self):
        self.rasterio_loader = RasterIOLoader()

    def __call__(self, image_path):
        image = self.rasterio_loader(image_path)
        image /= np.iinfo(np.uint16).max
        return image


class MSLoader:
    def __init__(self):
        self.rasterio_loader = RasterIOLoader()

    def __call__(self, image_path):
        image = self.rasterio_loader(image_path)
        image /= np.iinfo(np.uint16).max
        return image
