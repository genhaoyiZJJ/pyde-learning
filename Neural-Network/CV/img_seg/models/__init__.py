from torch import nn
import torch


def build_model(model_name, num_cls, pretrained=True):
    if model_name == 'deeplabv3+':
        from .deeplabv3_plus import DeepLabV3Plus
        model = DeepLabV3Plus(num_classes=num_cls, pretrained=pretrained)

    elif model_name == 'unet':
        from .unet import UNet
        model = UNet(num_classes=num_cls)

    elif model_name == 'unet_resnet':
        from .unet import UNetResnet
        model = UNetResnet(num_classes=num_cls, pretrained=pretrained)

    elif model_name == 'segnet':
        from .segnet import SegNet
        model = SegNet(num_classes=num_cls, pretrained=pretrained)

    elif model_name == 'bisenetv1':
        from .bisenetv1 import BiSeNetV1
        model = BiSeNetV1(n_classes=num_cls)

    else:
        raise Exception(f'only support resnet18, vgg16_bn and se_resnet50, but got {model_name}')
        
    return model