import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model

class cSELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(cSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        self.fc = nn.Conv2d(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.sigmoid(y)
        return x * y


class scSELayer(nn.Module):
    def __init__(self, channels, reduction=2):
        super(scSELayer, self).__init__()
        self.sSE = sSELayer(channels)
        self.cSE = cSELayer(channels, reduction=reduction)

    def forward(self, x):
        sx = self.sSE(x)
        cx = self.cSE(x)
        x = sx + cx
        return x




class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, channels),
            nn.BatchNorm2d(channels)
        )
        self.conv2 = nn.Sequential(
            conv3x3(channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        
        self.gate = scSELayer(out_channels)

    def forward(self, x, e=None):
        
        if e is not None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        
        
        x = self.gate(x)

        return x







class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)

        out = x + dilate1_out + dilate2_out + dilate3_out
        return out




class UNetResNet34(nn.Module):

    def load_pretrain(self):
        self.resnet.load_state_dict(torch.load('./models/pretrained_weights/resnet34-333f7ec4.pth'))

    def __init__(self):
        super(UNetResNet34, self).__init__()
        self.resnet = resnet34()

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        ) # 64
        
        self.encoder2 = nn.Sequential(
            self.resnet.layer1,
            scSELayer(64)
        ) # 64
        
        self.encoder3 = nn.Sequential(
            self.resnet.layer2,
            scSELayer(128)
        ) # 128
        
        self.encoder4 = nn.Sequential(
            self.resnet.layer3,
            scSELayer(256)
        ) # 256
        
        self.encoder5 = nn.Sequential(
            self.resnet.layer4,
            scSELayer(512)
        ) # 512

        self.center = Dblock(512)


        self.decoder5 = Decoder(    512, 256, 64)
        self.decoder4 = Decoder(256+ 64, 256, 64)
        self.decoder3 = Decoder(128+ 64, 128, 64)
        self.decoder2 = Decoder( 64+ 64,  64, 64)
        self.decoder1 = Decoder( 64+ 64,  64, 64)
        
        self.pixel = nn.Sequential(
            conv3x3(64*5, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv3x3(128, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.logit_pixel    = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self.image = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        
        self.logit_image = nn.Sequential(
            nn.Linear(32, 1)
        )

        self.fuse = nn.Sequential(
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.logit_fuse = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def add_depth_channels(image_tensor):
        _, _, h, w = image_tensor.size()
        for row, const in enumerate(torch.linspace(0, 1, h)):
            image_tensor[:, 1, row, :] = const
        image_tensor[:, 2] = image_tensor[:, 0] * image_tensor[:, 1]
        return image_tensor

    def forward(self, x):

        batch_size, H, W = x.shape

        # change two - add depth channels
        mean = [0.   , 0.   , 0.   ]
        std =  [1.   , 1.   , 1.   ]
        x = torch.stack([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
            ], 1)
        x = self.add_depth_channels(x)

        e1 = self.conv1(x)  # 128
        
        x = F.max_pool2d(e1, kernel_size=3, stride=2, padding=1)

        e2 = self.encoder2(x)  # ; print('e2',e2.size()) # 64
        e3 = self.encoder3(e2)  # ; print('e3',e3.size()) # 32
        e4 = self.encoder4(e3)  # ; print('e4',e4.size()) # 16
        e5 = self.encoder5(e4)  # ; print('e5',e5.size()) # 8

        f = self.center(e5)
        
        d5 = self.decoder5(f)
        d4 = self.decoder4(d5, e4) 
        d3 = self.decoder3(d4, e3) 
        d2 = self.decoder2(d3, e2) 
        d1 = self.decoder1(d2, e1)

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)


        f = F.dropout(f, p=0.50, training=self.training)
        pixel = self.pixel(f)
        logit_pixel = self.logit_pixel(pixel)  # ; print('logit',logit.size())

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        f = F.dropout(f, p=0.50, training=self.training)
        image = self.image(f)
        logit_image = self.logit_image(image).view(-1)

        f = torch.cat((
            pixel,
            F.interpolate(image.view(batch_size, -1, 1, 1), scale_factor=128, mode='nearest')
        ), 1)

        fuse = self.fuse(f)
        logit_fuse = self.logit_fuse(fuse)

        return logit_fuse, logit_pixel, logit_image

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.training = False
            self.eval()
        elif mode in ['train']:
            self.training = True
            self.train()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    net = UNetResNet34().cuda()
    import torchsummary
    print(net)
    torchsummary.summary(net, (128, 128))