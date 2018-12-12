import cv2
from torchvision.transforms import Normalize
from pysemseg import transforms

class SpaceNetTransform:
    def __init__(self, mode):
        self.mode = mode
        self.image_augmentations = transforms.Compose([
            transforms.RandomContrast(0.9, 1.1),
            transforms.RandomBrightness(-0.02, 0.02)
        ])
        self.joint_augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(0.15),
            transforms.RandomRotate(max_delta=3.),
            transforms.RandomScale(scale_range=(0.9, 1.1)),
            transforms.RandomCropFixedSize((512, 512))
        ])
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(
                mean=[0.21074309, 0.25471676, 0.30088879, 0.02913642],
                std=[0.04139718, 0.04751538, 0.06173738, 0.00039741]
            )
        ])

    def __call__(self, image, target):
        if self.mode == 'train':
            image[:, :, :3] = self.image_augmentations(image[:, :, :3])
            image, target = self.joint_augmentations(image, target)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target


class Erode:
    def __init__(self, kernel_size, iterations):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, mask):
        return cv2.erode(mask, kernel=kernel_size, iterations=iterations)
