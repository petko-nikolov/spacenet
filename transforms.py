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
        0.01392716, 0.010363, 0.01250671, 0.01410346, 0.01492058,
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
        self.kernel = np.ones(kernel_size, dtype=np.uint8)

    def __call__(self, mask):
        return cv2.erode(
            mask, kernel=self.kernel, iterations=self.iterations)


class RandomBrightness:
    """
    params:
        brightness level
    returns:
        image with changed birghtness
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        assert np.issubdtype(image.dtype, np.floating)
        brightness = np.random.uniform(self.low, self.high)
        image = image + brightness
        return np.clip(image, 0.0, 1.0)


class RandomRotate:
    def __init__(self,  max_delta=7):
        self.max_delta = max_delta
    def __call__(self, image, mask):
        angle = np.random.uniform(-self.max_delta, self.max_delta)
        image = self._rotate(image, angle, cv2.INTER_AREA)
        mask = self._rotate(mask, angle, cv2.INTER_NEAREST)
        return image, mask

    def _rotate(self, mask, angle, interpolation):
        mask_center = (mask.shape[1] / 2, mask.shape[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(mask_center, angle, 1.0)
        result = cv2.warpAffine(
            mask, rot_mat, (mask.shape[1], mask.shape[0]),
            flags=interpolation, borderMode=cv2.BORDER_REFLECT
        )
        return result


class SpaceNetTransform:
    def __init__(self, mode, image_types=['Pan-Sharpen'], normalize=True):
        self.mode = mode
        self.image_types = image_types
        self.normalize = normalize

        self.image_augmentations = transforms.Compose([
            transforms.RandomHueSaturation(
                hue_delta=0.05, saturation_scale_range=(0.9, 1.1)
            ),
            transforms.RandomContrast(0.8, 1.2),
            transforms.RandomBrightness(-0.1, 0.1),
            transforms.RandomGammaCorrection(min_gamma=0.9, max_gamma=1.1),
            transforms.RandomGaussianBlur()
        ])
        self.joint_augmentations = transforms.Compose([
            transforms.RandomScale(),
            transforms.RandomCropFixedSize((512, 512)),
            RandomRotate(max_delta=7.),
            transforms.Choice([
                transforms.RandomPerspective(),
                transforms.RandomShear(),
                transforms.RandomElasticTransform()
            ], p=[0.4, 0.4, 0.2]),
            transforms.RandomHorizontalFlip(0.5),
        ])

        self.mean = list(itertools.chain(
            *[COLLECTION_MEANS[image_type] for image_type in image_types]
        ))

        self.std = list(itertools.chain(
            *[COLLECTION_STDS[image_type] for image_type in image_types]
        ))


        tensor_transforms = [
            transforms.ToTensor(),
        ]
        if self.normalize:
            tensor_transforms.append(
                Normalize(mean=self.mean, std=self.std)
            )
        self.tensor_transforms = transforms.Compose(tensor_transforms)

    def __call__(self, image, target):
        if self.mode == 'train':
            image[:, :, :3] = self.image_augmentations(image[:, :, :3])
            image, target = self.joint_augmentations(image, target)
        elif self.mode == 'val':
            image = transforms.Resize((928, 928))(image)
            target = transforms.Resize((928, 928), interpolation=cv2.INTER_NEAREST)(target)

        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target


class ProposalTransform:
    def __init__(self, mode):
        self.mode = mode
        self.image_augmentations = transforms.Compose([
            transforms.RandomHueSaturation(
                hue_delta=0.05, saturation_scale_range=(0.9, 1.1)
            ),
            transforms.RandomContrast(0.9, 1.1),
            transforms.RandomBrightness(-0.05, 0.05),
            transforms.RandomGammaCorrection(min_gamma=0.9, max_gamma=1.1)
        ])
        self.joint_augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            RandomRotate(max_delta=7.),
            transforms.RandomScale(),
            # transforms.RandomCropFixedSize((512, 512))
        ])

        self.target_augmentations = transforms.Compose([
            # Erode(kernel_size=(3,3), iterations=3)
        ])

        self.mean = list(itertools.chain(
            *[COLLECTION_MEANS[image_type] for image_type in ['Pan-Sharpen', 'PAN', 'MS']]
        )) + [0.5, 32.0]

        self.std = list(itertools.chain(
           *[COLLECTION_STDS[image_type] for image_type in ['Pan-Sharpen', 'PAN', 'MS']]
        )) + [0.5, 20.0]

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


