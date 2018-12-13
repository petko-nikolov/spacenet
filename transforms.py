import itertools
import scipy.ndimage
import numpy as np
import cv2
from torchvision.transforms import Normalize
from pysemseg import transforms


COLLECTION_MEANS = {
    'Pan-Sharpen': [0.21074309, 0.25471676, 0.30088879, 0.02913642],
    'PAN': [0.01505574],
    'MS': [
        0.01392716, 0.010363  , 0.01250671, 0.01410346, 0.01492058,
        0.02249807, 0.02937736, 0.03253879
    ]
}


COLLECTION_STDS = {
    'Pan-Sharpen': [0.04139718, 0.04751538, 0.06173738, 0.00039741],
    'PAN': [0.00018373],
    'MS': [
        0.00015886, 0.0001348 , 0.00016091, 0.00020474, 0.00022065,
        0.00025092, 0.00034316, 0.00042284
    ]
}


class Erode:
    def __init__(self, kernel_size, iterations):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, mask):
        return cv2.erode(
            mask, kernel=self.kernel_size, iterations=self.iterations)


class RandomRotate:
    def __init__(self,  max_delta=5):
        self.max_delta = max_delta
    def __call__(self, image, mask):
        angle = np.random.uniform(-self.max_delta, self.max_delta)
        image = scipy.ndimage.interpolation.rotate(
            image, angle, mode='reflect')
        mask = scipy.ndimage.interpolation.rotate(
            mask, angle, mode='reflect')
        return image, mask


class SpaceNetTransform:
    def __init__(self, mode, image_types=['Pan-Sharpen']):
        self.mode = mode
        self.image_augmentations = transforms.Compose([
            transforms.RandomContrast(0.9, 1.1),
            transforms.RandomBrightness(-0.02, 0.02)
        ])
        self.joint_augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(0.15),
            RandomRotate(max_delta=2.),
            transforms.RandomScale(scale_range=(0.9, 1.1)),
            transforms.RandomCropFixedSize((512, 512))
        ])

        self.target_augmentations = transforms.Compose([
            Erode(kernel_size=(3,3), iterations=6)
        ])

        self.image_types = image_types

        self.mean = list(itertools.chain(
            *[COLLECTION_MEANS[image_type] for image_type in image_types]
        ))

        self.std = list(itertools.chain(
            *[COLLECTION_STDS[image_type] for image_type in image_types]
        ))

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, image, target):
        if self.mode == 'train':
            image[:, :, :3] = self.image_augmentations(image[:, :, :3])
            target = self.target_augmentations(target)
            image, target = self.joint_augmentations(image, target)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target


