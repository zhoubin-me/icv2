from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def gradhook(self, grad_input, grad_output):
    importance = grad_output[0] ** 2 # [N, C, H, W]
    if len(importance.shape) == 4:
        importance = torch.sum(importance, 3) # [N, C, H]
        importance = torch.sum(importance, 2) # [N, C]
    importance = torch.mean(importance, 0) # [C]
    self.importance += importance

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('输入的shape为:'+str(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print('avg_out的shape为:' + str(avg_out.shape))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #print('max_out的shape为:' + str(max_out.shape))
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Channel_Importance_Measure(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.randn(num_channels), requires_grad=False)
        nn.init.constant_(self.scale, 1.0)
        self.register_buffer('importance', torch.zeros_like(self.scale))

    def forward(self, x):
        if len(x.shape) == 4:
            x = x * self.scale.reshape([1,-1,1,1])
        else:
            x = x * self.scale.reshape([1,-1])
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #out = self.ca(out) * out
        #out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(planes * 4)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer1_importance = Channel_Importance_Measure(64 * block.expansion)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2_importance = Channel_Importance_Measure(128 * block.expansion)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3_importance = Channel_Importance_Measure(256 * block.expansion)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer4_importance = Channel_Importance_Measure(512 * block.expansion)

        self.feature = nn.AvgPool2d(4, stride=1)
        self.raw_features_importance = Channel_Importance_Measure(512 * block.expansion)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x


    def forward_with_hook(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        int_features = []
        x = self.layer1(x)
        int_features.append(x)
        x = self.layer1_importance(x)

        x = self.layer2(x)
        int_features.append(x)
        x = self.layer2_importance(x)

        x = self.layer3(x)
        int_features.append(x)
        x = self.layer3_importance(x)

        x = self.layer4(x)
        int_features.append(x)
        x = self.layer4_importance(x)

        x = self.feature(x)
        x = x.view(x.size(0), -1)
        int_features.append(x)
        x = self.raw_features_importance(x)

        importance = [self.layer1_importance.importance,
                      self.layer2_importance.importance,
                      self.layer3_importance.importance,
                      self.layer4_importance.importance,
                      self.raw_features_importance.importance]

        return x, int_features, importance



    def start_cal_importance(self):
        self._hook = [self.layer1_importance.register_backward_hook(gradhook),
                      self.layer2_importance.register_backward_hook(gradhook),
                      self.layer3_importance.register_backward_hook(gradhook),
                      self.layer4_importance.register_backward_hook(gradhook),
                      self.raw_features_importance.register_backward_hook(gradhook)]

    def get_importance(self):
        return [self.layer1_importance.importance,
                      self.layer2_importance.importance,
                      self.layer3_importance.importance,
                      self.layer4_importance.importance,
                      self.raw_features_importance.importance]


    def reset_importance(self):
        self.layer1_importance.importance.zero_()
        self.layer2_importance.importance.zero_()
        self.layer3_importance.importance.zero_()
        self.layer4_importance.importance.zero_()
        self.raw_features_importance.importance.zero_()

    def stop_cal_importance(self):
        for hook in self._hook:
            hook.remove()
        self._hook = None


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model



class Model(nn.Module):

    def __init__(self, numclass, backbone):
        super(Model, self).__init__()
        if backbone == 'resnet18':
            feature_extractor = resnet18_cbam()
        elif backbone == 'resnet50':
            feature_extractor = resnet50_cbam()
        else:
            raise ValueError("no such model")

        self.feature = feature_extractor
        self.fc = nn.Linear(
            self.feature.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def forward_with_hook(self, input):
        x, int_features, importance = self.feature.forward_with_hook(input)
        x = self.fc(x)
        return x, int_features, importance

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)


if __name__ == '__main__':
    model = Model(100, 'resnet18')

    for _ in range(10):
        x = torch.rand(16, 3, 32, 32)
        y = torch.randint(0, 50, size=(16,))

        model.feature.start_cal_importance()
        model.feature.reset_importance()
        output, features, importance = model.forward_with_hook(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        print([x.abs().sum() for x in importance])
        print([x.abs().sum() for x in features])
        print("=============")
        model.feature.stop_cal_importance()

