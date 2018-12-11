from torchvision.transforms import Normalize
from pysemseg import transforms

class SpaceNetTransform:
    def __init__(self, mode):
        self.mode = mode
        self.image_augmentations = transforms.Compose([
            transforms.RandomContrast(0.8, 1.2),
            transforms.RandomBrightness(-0.001, 0.001)
        ])
        self.joint_augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotate(),
            transforms.RandomScale(),
            # transforms.RandomCropFixedSize((512, 512))
            transforms.RandomCropFixedSize((256, 256))
        ])
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(
                mean=[0.01019215, 0.01236063, 0.01478618, 0.0291202],
                std=[0.00015258, 0.00018149, 0.00024689, 0.00039786]
            )
        ])

    def __call__(self, image, target):
        if self.mode == 'train':
            image = self.image_augmentations(image)
            image, target = self.joint_augmentations(image, target)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target
