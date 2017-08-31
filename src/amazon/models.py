import torch
import torch.nn as nn
import torchvision.models as models
from .model_wrapper import Model
from .utils import CLASSES


def get_pretrained_model(arch, lr=1e-3, momentum=0.9, weight_decay=1e-4):
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    elif arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise Exception("Not supported architecture: %s"%arch)
    
    if arch.startswith('vgg'):
        mod = list(model.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096, len(CLASSES)))
        new_classifier = torch.nn.Sequential(*mod)
        model.classifier = new_classifier
    elif arch.startswith('resnet'):
        model.fc = nn.Linear(2048, len(CLASSES))
    elif arch.startswith('densenet'):
        model.classifier = nn.Linear(1024, len(CLASSES))
    else:
        raise Exception("Not supported architecture: %s"%arch)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    
    return Model(model, criterion, optimizer)
