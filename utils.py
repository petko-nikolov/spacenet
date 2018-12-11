import tifffile
import numpy as np


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


class TiffFileLoader:
    def __call__(self, image_path):
        image = tifffile.imread(image_path)
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = image.transpose((1, 2, 0))
        return image
