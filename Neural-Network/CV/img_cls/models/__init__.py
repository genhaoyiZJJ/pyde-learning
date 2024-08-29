from torch import nn


def build_model(model_name, num_cls, pretrained=True):
    if model_name == 'resnet18':
        from .resnet_tv import resnet18
        
        model = resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.fc = fc
    elif model_name == 'vgg16_bn':
        from .vgg_tv import vgg16_bn
        
        model = vgg16_bn(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.classifier[-1] = fc
    elif model_name == 'se_resnet50':
        from .se_resnet import se_resnet50
        
        model = se_resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.fc = fc
    elif model_name == 'resnet20_cifar':
        from .resnet_cifar import resnet20

        return resnet20()
    else:
        raise Exception(f'only support resnet18, vgg16_bn and se_resnet50, but got {model_name}')
        
    return model