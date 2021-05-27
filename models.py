import torch
from torch import nn
import torchvision
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class Resnet(nn.Module):
    def __init__(self, layers=50, pretrained=True, drop_rate=0.2):
        super(Resnet, self).__init__()
        assert layers in (18, 34, 50), 'layers must in (18, 34, 50)'
        self.pretrained = pretrained
        if layers == 18:
            torch_resnet = torchvision.models.resnet18(pretrained=self.pretrained)
            block_expansion = 1
        elif layers == 34:
            torch_resnet = torchvision.models.resnet34(pretrained=self.pretrained)
            block_expansion = 1
        elif layers == 50:
            torch_resnet = torchvision.models.resnet50(pretrained=self.pretrained)
            block_expansion = 4

        # remove the last layer which is a simple linear layer in source code:
        #   self.fc = nn.Linear(512 * block.expansion, num_classes)  index -1
        self.res_structure = list(torch_resnet.children())[:-1]
        self.resnet = nn.Sequential(*self.res_structure)
        self.extra_layer1 = nn.Sequential(
            nn.Linear(block_expansion * 512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            )

        self.age_predictor = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(drop_rate), nn.ReLU(), nn.Linear(128, 8))

        self.gender_predictor = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(drop_rate), nn.ReLU(), nn.Linear(128, 2))

    def fine_tune(self, start=0):
        """
        Whether fine_tune the resnet50.
        Set start=0 if pretrained=False
        self.resnet50Structures total 9 chidrens:
        module1: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)                  renet50 0
        module2: self.bn1 = norm_layer(self.inplanes)                                                                      renet50 1
        module3: self.relu = nn.ReLU(inplace=True)                                                                         renet50 2
        module4: self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                                           renet50 3
        module5: self.layer1 = self._make_layer(block, 64, layers[0])                                                      renet50 4
        module6: self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])   resnet50 -4
        module7: self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])   resnet50 -3
        module8: self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])   resnet50 -2
        module9: self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                                               resnet50 -1

        This function will fine tune all the module after parameter: start
        """
        for i, m in enumerate(list(self.res_structure.children())):
            for p in m.parameters():
                if i >= start:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        print(f'fine tune the resnet module after {min(max(0, start), len(list(self.res_structure.children()))-1)}')

    def forward(self, image_batch):
        output = self.resnet(image_batch)
        output = output.flatten(1)
        output = self.extra_layer1(output)

        pred_age = self.age_predictor(output)
        pred_gender = self.gender_predictor(output)

        return pred_gender, pred_age



class VGG(nn.Module):
    def __init__(self, layers=16, pretrained=True, drop_rate=0.2):
        super(VGG, self).__init__()
        assert layers in (16,19), 'layers must in (16,19)'
        self.pretrained = pretrained
        if layers == 16:
            torch_resnet = torchvision.models.vgg16(pretrained=self.pretrained)
        elif layers == 19:
            torch_resnet = torchvision.models.vgg19(pretrained=self.pretrained)

        self.vgg_structure = list(torch_resnet.children())[:-1]
        self.vgg = nn.Sequential(*self.vgg_structure)
        self.extra_layer1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            )

        self.age_predictor = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(drop_rate), nn.ReLU(), nn.Linear(128, 8))

        self.gender_predictor = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(drop_rate), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, image_batch):
        output = self.vgg(image_batch)
        output = output.flatten(1)
        output = self.extra_layer1(output)

        pred_age = self.age_predictor(output)
        pred_gender = self.gender_predictor(output)

        return pred_gender, pred_age


class Base_CNN(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(Base_CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((7, 7)),
            )

        self.linears_gender = nn.Sequential(
            nn.Linear(384*7*7, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(512, 2)
            )

        self.linears_age = nn.Sequential(
            nn.Linear(384*7*7, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(512, 8)
            )

    def forward(self, image_batch):
        output = self.cnn_layers(image_batch)
        output = output.flatten(1)
        
        pred_age = self.linears_age(output)
        pred_gender = self.linears_gender(output)

        return pred_gender, pred_age