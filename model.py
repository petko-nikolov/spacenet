from pysemseg.models.deeplab import Deeplab, Decoder
from pysemseg.models.resnet import resnet101


def deeplab101(in_channels, n_classes, **kwargs):
    return Deeplab(
        in_channels, n_classes, resnet101,
        aspp_rates=[4, 8, 12],
        decoder=Decoder(256, 256, 48), **kwargs
    )

