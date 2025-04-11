import math

import torch
from torch import nn

from backbone.repvgg import get_RepVGG_func_by_name
import utils


class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        # return utils.compute_rotation_matrix_from_ortho6d(x)
        return utils.robust_compute_rotation_matrix_from_ortho6d(x)

class SixDRepNet5D(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet5D, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 5) #5

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        # return utils.compute_rotation_matrix_from_ortho6d(x)
        return utils.compute_rotation_matrix_from_ortho5d(x)



class ResNet(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

        self.linear_reg = nn.Linear(512 * block.expansion, 4)

        self._initialize_weights()

    def _initialize_weights(self):
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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        x = torch.flatten(x, 1)
        out = self.linear_reg(x)
        # print("Channels after linear_reg:", out.size())
        # if self.normalize_output:
            # print('redhwan________________________________________')
        out = out / out.norm(dim=1, keepdim=True)
        # print("Channels after out_norm:", x.size())

        return out


class RotMat6DDirect(torch.nn.Module):
    def __init__(self, batchnorm=False):
        super(RotMat6DDirect, self).__init__()
        self.net = WQuatNet(dim_out=6, normalize_output=False, batchnorm=batchnorm)

    def forward(self, x):
        vecs = self.net(x)
        C = sixdim_to_rotmat(vecs)
        return C


class Quat(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True, quaternion_dim=4):
        super(Quat, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel
        # Add quaternion regression layers
        self.linear_reg = nn.Linear(fea_dim, quaternion_dim) #4

    def forward(self, x):
        # print("Channels after layerx:", x.size())
        x = self.layer0(x)
        # print("Channels after layer0:", x.size())
        x = self.layer1(x)
        # print("Channels after layer1:", x.size())
        x = self.layer2(x)
        # print("Channels after layer2:", x.size())
        x = self.layer3(x)
        # print("Channels after layer3:", x.size())
        x = self.layer4(x)
        # print("Channels after layer4:", x.size())  # Print the size of x after layer4
        x = self.gap(x)
        # print("Channels after gap:", x.size())
        x = torch.flatten(x, 1)
        # print("Channels after flatten:", x.size())
        quaternion_output = self.linear_reg(x)

        return quaternion_output
        # return q


class WQuatNet(nn.Module):
    def __init__(self, backbone_name, backbone_file, deploy, pretrained=True, dim_out=4, normalize_output=True, batchnorm=False):
        super(WQuatNet, self).__init__()

        # Feature extraction backbone
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)
        self.normalize_output = normalize_output
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel
        # Add quaternion regression layers

        self.linear_reg = nn.Linear(fea_dim, dim_out)  # 4



    def forward(self, x):
        # Backbone
        # print("Channels after layerx:", x.size())
        x = self.layer0(x)
        # print("Channels after layer0:", x.size())
        x = self.layer1(x)
        # print("Channels after layer1:", x.size())
        x = self.layer2(x)
        # print("Channels after layer2:", x.size())
        x = self.layer3(x)
        # print("Channels after layer3:", x.size())
        x = self.layer4(x)
        # print("Channels after layer4:", x.size())  # Print the size of x after layer4
        x = self.gap(x)
        # print("Channels after gap:", x.size())
        # x = torch.flatten(x, 1)
        # print("Channels after flatten:", x.size())
        # x = self.linear_reg(x)  # Bx10
        # print("Channels after linear_reg:", x.size())
        # x = torch.flatten(x, 1)
        # print("Channels after flatten:", x.size())

        # Do not flatten x here, keep it as a 4D tensor

        # WQuatNet Head
        # if x.size(1) != 6:
        #     raise ValueError(
        #         "Input tensor should have 6 channels for PointFeatCNN, but got {} channels.".format(x.size(1)))

        # Split the input into two point clouds
        # x1 = x[:, :3, :, :]  # Extract the first 3 channels for the first point cloud
        # x2 = x[:, 3:, :, :]  # Extract the next 3 channels for the second point cloud
        # # print("Channels after x1, x2:", x1.size(), x2.size())
        #
        # # Combine the two point clouds
        # out = torch.cat([x1, x2], dim=1)
        # print("Channels after out:", out.size())

        x = torch.flatten(x, 1)
        # print("Channels after flatten:", x.size())

        # Use the correct attribute name for the linear regression layer
        out = self.linear_reg(x)  # Bx4 (assuming dim_out=4)
        # print("Channels after quaternion_net:", x.size())

        # Normalization
        if self.normalize_output:
            # print('redhwan________________________________________')
            out = out/ out.norm(dim=1, keepdim=True)
            # print("Channels after out_norm:", x.size())

        return out


