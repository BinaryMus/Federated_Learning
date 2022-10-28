from torchvision.models.resnet import ResNet, BasicBlock


def Resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
