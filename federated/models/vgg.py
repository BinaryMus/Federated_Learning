from torchvision.models.vgg import VGG, make_layers, cfgs


def VGG11(num_classes):
    return VGG(make_layers(cfgs["A"]), num_classes=num_classes)
