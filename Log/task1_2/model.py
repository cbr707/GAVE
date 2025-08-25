import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv_block = nn.Sequential(
            self.conv1,
            activf(inplace=True),
            self.conv2,
            activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class Down_8(nn.Module):
    def __init__(self, in_chan, out_chan1, out_chan2, out_chan3, kernal_size=3, stride=2, pad=1):
        super(Down_8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan1, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv2 = nn.Conv2d(in_channels=out_chan1, out_channels=out_chan2, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv3 = nn.Conv2d(in_channels=out_chan2, out_channels=out_chan3, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class Down_4(nn.Module):
    def __init__(self, in_chan, out_chan1, out_chan2, kernal_size=3, stride=2, pad=1):
        super(Down_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan1, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv2 = nn.Conv2d(in_channels=out_chan1, out_channels=out_chan2, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.relu(self.bn(x))
        return out


class Down_2(nn.Module):
    def __init__(self, in_chan, out_chan, kernal_size=3, stride=2, pad=1):
        super(Down_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        out = self.relu(self.bn(x))
        return out


class Up_n(nn.Module):
    def __init__(self, in_chan, out_chan, kernal_size=1, stride=1, pad=0, n=2):
        super(Up_n, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=n, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        out = self.upsample(x)
        return out


class dense_skip(nn.Module):
    def __init__(self, bc):
        super(dense_skip, self).__init__()

        self.down1_2 = Down_2(bc, 2 * bc, 3, 2, 1)
        self.down1_4 = Down_4(bc, 2 * bc, 4 * bc, 3, 2, 1)
        self.down1_8 = Down_8(bc, 2 * bc, 4 * bc, 8 * bc, 3, 2, 1)

        self.down2_2 = Down_2(2 * bc, 4 * bc, 3, 2, 1)
        self.down2_4 = Down_4(2 * bc, 4 * bc, 8 * bc, 3, 2, 1)

        self.down3_2 = Down_2(4 * bc, 8 * bc, 3, 2, 1)
        self.upsample3_2 = Up_n(4 * bc, 2 * bc, 1, 1, 0, 2)

        self.upsample4_2 = Up_n(8 * bc, 4 * bc, 1, 1, 0, 2)
        self.upsample4_4 = Up_n(8 * bc, 2 * bc, 1, 1, 0, 4)

        self.down21_2 = Down_2(2 * bc, 4 * bc, 3, 2, 1)
        self.down21_4 = Down_4(2 * bc, 4 * bc, 8 * bc, 3, 2, 1)

        self.down31_2 = Down_2(4 * bc, 8 * bc, 3, 2, 1)

        self.upsample41_2 = Up_n(8 * bc, 4 * bc, 1, 1, 0, 2)

        self.down32_2 = Down_2(4 * bc, 8 * bc, 3, 2, 1)

    def forward(self, x0, x1, x2, x3):
        x_0_0_down2 = self.down1_2(x0)
        x_0_0_down4 = self.down1_4(x0)
        x_0_0_down8 = self.down1_8(x0)

        x_0_1_down2 = self.down2_2(x1)
        x_0_1_down4 = self.down2_4(x1)

        x_0_2_up2 = self.upsample3_2(x2)
        x_0_2_down2 = self.down3_2(x2)

        x_0_3_up4 = self.upsample4_4(x3)
        x_0_3_up2 = self.upsample4_2(x3)

        x_1_1 = torch.stack([x_0_0_down2, x1, x_0_2_up2, x_0_3_up4]).mean(dim=0)
        x_1_2 = torch.stack([x_0_0_down4, x_0_1_down2, x2, x_0_3_up2]).mean(dim=0)
        x_1_3 = torch.stack([x_0_0_down8, x_0_1_down4, x_0_2_down2, x3]).mean(dim=0)

        x_1_1_down2 = self.down21_2(x_1_1)
        x_1_1_down4 = self.down21_4(x_1_1)

        x_1_2_down2 = self.down31_2(x_1_2)

        x_1_3_up2 = self.upsample41_2(x_1_3)

        x_2_2 = torch.stack([x_1_1_down2, x_1_2, x_1_3_up2]).mean(dim=0)
        x_2_3 = torch.stack([x_1_1_down4, x_1_2_down2, x_1_3]).mean(dim=0)

        x_2_2_down2 = self.down32_2(x_2_2)

        x_3_3 = torch.stack([x_2_2_down2, x_2_3]).mean(dim=0)

        return x0, x_1_1, x_2_2, x_3_3


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2 * base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        self.skip = dense_skip(base_ch)
        self.vit_img_size = 224
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=16 * base_ch,
            in_chans=16 * base_ch,
            global_pool='avg'
        )
        self.linear_expand = nn.Linear(16 * base_ch, 16 * base_ch * (self.vit_img_size // 32) ** 2)
        self.out_proj = nn.Conv2d(16 * base_ch, 16 * base_ch, 1)

        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)

        x1, x2, x3, x4 = self.skip(x1, x2, x3, x4)

        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)

        B, C, H, W = x.shape
        x_vit_in = F.interpolate(x, size=(self.vit_img_size, self.vit_img_size), mode='bilinear', align_corners=False)
        vit_out = self.vit(x_vit_in)
        x = self.linear_expand(vit_out).view(B, C, self.vit_img_size // 32, self.vit_img_size // 32)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.out_proj(x)

        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x


class UNet(UNetModule):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__(input_ch, output_ch, base_ch)
        print('model：baseUNet')
        print('input_ch:', input_ch)
        print('output_ch:', output_ch)
        print('base_ch:', base_ch)

    def forward(self, x):
        return [super().forward(x)]


# 基线
class GAVENet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)  # a, bv, v
        predictions.append(pred_1)
        bv_logits = pred_1[:, 1:2, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 1:2, :, :]

        pred_2 = self.second_u(pred_1)
        a_logits = pred_2[:, 0, :, :].unsqueeze(1)
        v_logits = pred_2[:, 1, :, :].unsqueeze(1)
        predictions.append(torch.cat((a_logits, bv_logits, v_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2[:, 0:1], bv, pred_2[:, 1:2]), dim=1)
            pred_2 = self.second_u(pred_2)
            a_logits = pred_2[:, 0, :, :].unsqueeze(1)
            v_logits = pred_2[:, 1, :, :].unsqueeze(1)
            predictions.append(torch.cat((a_logits, bv_logits, v_logits), dim=1))

        return predictions


# 基线，但是在后续网络中加入了原图信息辅助判断
class GAVENetV2(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__()
        self.first_u = UNet(input_ch, output_ch, base_ch)
        self.second_u = UNet((output_ch + input_ch), 2, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)[0]  # a, bv, v
        predictions.append(pred_1)
        bv_logits = pred_1[:, 1:2, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 1:2, :, :]
        pred_1 = torch.cat((x, pred_1), dim=1)

        pred_2 = self.second_u(pred_1)[0]
        a_logits = pred_2[:, 0, :, :].unsqueeze(1)
        v_logits = pred_2[:, 1, :, :].unsqueeze(1)
        predictions.append(torch.cat((a_logits, bv_logits, v_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((x, pred_2[:, 0:1], bv, pred_2[:, 1:2]), dim=1)
            pred_2 = self.second_u(pred_2)[0]
            a_logits = pred_2[:, 0, :, :].unsqueeze(1)
            v_logits = pred_2[:, 1, :, :].unsqueeze(1)
            predictions.append(torch.cat((a_logits, bv_logits, v_logits), dim=1))

        return predictions


# 基线，但是在后续网络中加入了原图信息辅助判断,四通道输出
class GAVENetV3(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__()
        self.first_u = UNet(input_ch, output_ch, base_ch)
        self.second_u = UNet((output_ch + input_ch), 3, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)[0]  # a, bv, v, a*v
        predictions.append(pred_1)
        bv_logits = pred_1[:, 1:2, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 1:2, :, :]
        pred_1 = torch.cat((x, pred_1), dim=1)

        pred_2 = self.second_u(pred_1)[0]
        a_logits = pred_2[:, 0, :, :].unsqueeze(1)
        v_logits = pred_2[:, 1, :, :].unsqueeze(1)
        av_logits = pred_2[:, 2, :, :].unsqueeze(1)
        predictions.append(torch.cat((a_logits, bv_logits, v_logits, av_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((x, pred_2[:, 0:1], bv, pred_2[:, 1:2], pred_2[:, 2:3]), dim=1)
            pred_2 = self.second_u(pred_2)[0]
            a_logits = pred_2[:, 0, :, :].unsqueeze(1)
            v_logits = pred_2[:, 1, :, :].unsqueeze(1)
            av_logits = pred_2[:, 2, :, :].unsqueeze(1)
            predictions.append(torch.cat((a_logits, bv_logits, v_logits, av_logits), dim=1))

        return predictions


# 可更换编码器的UNet
class SMPUNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch=16, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        """
        通用的 Smp Unet 封装类。

        参数:
        - input_ch: 输入通道数
        - output_ch: 输出通道数
        - encoder_name: 编码器名称 (如 'resnet18', 'resnet34', 'mobilenet_v2')
        - encoder_weights: 预训练权重 ('imagenet' 或 None)
        - kwargs: 额外传给 smp.Unet 的参数
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=input_ch,
            classes=output_ch,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            **kwargs
        )
        print('SMPUNet')
        print('input_ch:', input_ch)
        if output_ch == 2:
            print('请一定注意损失函数是否匹配！！！！！！！')
        print('output_ch:', output_ch)
        print('encoder:', encoder_name)
        print('weights:', encoder_weights)
        print('decoder channels:', decoder_channels)

    def forward(self, x):
        return self.model(x)


class SMPUNetV2(SMPUNet):
    def __init__(self, input_ch, output_ch, base_ch=16, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        """
        通用的 Smp Unet 封装类。

        参数:
        - input_ch: 输入通道数
        - output_ch: 输出通道数
        - encoder_name: 编码器名称 (如 'resnet18', 'resnet34', 'mobilenet_v2')
        - encoder_weights: 预训练权重 ('imagenet' 或 None)
        - kwargs: 额外传给 smp.Unet 的参数
        """
        super().__init__(input_ch, output_ch, base_ch=base_ch, num_iterations=num_iterations, encoder_name=encoder_name,
                         encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                         decoder_channels=decoder_channels)

    def forward(self, x):
        x = x[:, 3:6]
        return self.model(x)


class SMPUNetV3(SMPUNet):
    def __init__(self, input_ch, output_ch, base_ch=16, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        """
        通用的 Smp Unet 封装类。

        参数:
        - input_ch: 输入通道数
        - output_ch: 输出通道数
        - encoder_name: 编码器名称 (如 'resnet18', 'resnet34', 'mobilenet_v2')
        - encoder_weights: 预训练权重 ('imagenet' 或 None)
        - kwargs: 额外传给 smp.Unet 的参数
        """
        super().__init__(input_ch, output_ch, base_ch=base_ch, num_iterations=num_iterations, encoder_name=encoder_name,
                         encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                         decoder_channels=decoder_channels)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []
        x = x[:, 3:6]
        pred = self.model(x)
        predictions.append(pred)
        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)
            pred = self.model(pred)
            predictions.append(pred)

        return predictions


# 基于smpunet的GAVENet，跑的比较慢，主体血管提取的比较一般，但是血管细分的比较好
class SMPGAVENet(GAVENet):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5,
                 encoder_name='resnet18', encoder_weights='imagenet',
                 encoder_depth=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        # self.first_u = SMPUNet(input_ch, output_ch, encoder_name, encoder_weights)
        # self.second_u = SMPUNet(output_ch, 2, encoder_name, encoder_weights)
        decoder_channels = []
        counter = encoder_depth
        while counter:
            decoder_channels = [x * 2 for x in decoder_channels]
            decoder_channels.append(base_ch)
            counter -= 1
        decoder_channels = tuple(decoder_channels)
        self.first_u = SMPUNet(
            input_ch,
            output_ch,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )
        self.second_u = SMPUNet(
            output_ch,
            2,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


# unet+smpunet的GAVE架构,但是没跑起来过
class SMPGAVENetV2(GAVENet):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5,
                 encoder_name='resnet18', encoder_weights='imagenet',
                 encoder_depth=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        # self.first_u = SMPUNet(input_ch, output_ch, encoder_name, encoder_weights)
        # self.second_u = SMPUNet(output_ch, 2, encoder_name, encoder_weights)
        decoder_channels = []
        counter = encoder_depth
        while counter:
            decoder_channels = [x * 2 for x in decoder_channels]
            decoder_channels.append(base_ch)
            counter -= 1
        decoder_channels = tuple(decoder_channels)

        self.first_u = UNetModule(input_ch, output_ch, 32)

        self.second_u = SMPUNet(
            output_ch,
            2,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


class SMPGAVENetV3(SMPGAVENet):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__(input_ch, output_ch, base_ch, num_iterations, encoder_name=encoder_name,
                         encoder_weights=encoder_weights, encoder_depth=encoder_depth)
        self.second_u = SMPUNet(
            input_ch + output_ch,
            output_ch,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)  # a, bv, v
        predictions.append(pred_1)
        pred_1 = torch.sigmoid(pred_1)
        pred_1 = torch.cat((x, pred_1), dim=1)

        pred_2 = self.second_u(pred_1)
        predictions.append(pred_2)

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((x, pred_2), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(pred_2)

        return predictions


# 仅修复模式，二阶段不接入原图
class SMPGAVENetV4(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, encoder_name='resnet50',
                 encoder_weights=None, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        super().__init__()
        self.first_u = SMPUNet(
            input_ch,
            output_ch,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )
        self.second_u = SMPUNet(
            input_ch,
            output_ch,
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)  # a, bv, v
        predictions.append(pred_1)
        pred_1 = torch.sigmoid(pred_1)

        pred_2 = self.second_u(pred_1)
        predictions.append(pred_2)

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = self.second_u(pred_2)
            predictions.append(pred_2)

        return predictions


# 基于smpunet的循环UNet,只跑单血管，跑起来也很慢，没成功过
class RRSMPUNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch=16, num_iterations=5,
                 encoder_name='resnet18', encoder_weights='imagenet',
                 encoder_depth=5):
        super().__init__()
        # 固定输入通道为 6，输出通道为 1
        self.num_iterations = num_iterations
        self.input_ch = input_ch
        self.output_ch = output_ch
        print('正在使用RRSMPUNet，请务必注意通道数和损失函数是否正确')
        print('当前输入通道为：', input_ch)
        print('当前输出通道为：', output_ch)
        decoder_channels = []
        counter = encoder_depth
        while counter:
            decoder_channels = [x * 2 for x in decoder_channels]
            decoder_channels.append(base_ch)
            counter -= 1
        decoder_channels = tuple(decoder_channels)

        # 因为原来的第二个 U-Net 是输出 2 通道（a 和 v），
        # 现在我们只需要单通道输出，所以要重写它
        # self.first_u = UNetModule(4, 2, base_ch)
        self.second_u = SMPUNet(
            input_ch,
            output_ch,
            encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    def forward(self, x):  # shape: (B, 6, H, W)

        predictions = []

        # 第一次 U-Net 处理
        pred_1 = self.second_u(x)  # shape: (B, C, H, W)
        predictions.append(pred_1)
        x = x[:, 0:(self.input_ch - self.output_ch), :, :]  # (B, I-C, H, W)
        pred = pred_1  # shape: (B, C, H, W)
        # 如果不需要 bv / a / v 的拆分，可以直接迭代 refine
        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)  # 映射到 0-1
            pred = torch.cat((x, pred), dim=1)  # (B, I, H, W)
            pred = self.second_u(pred)
            predictions.append(pred)

        return predictions


# 基于smpunet的循环UNet,固定输入6输出2，没成功过
class RRSMPUNet(RRSMPUNet):
    def __init__(self, input_ch, output_ch, base_ch=16, num_iterations=5,
                 encoder_name='resnet18', encoder_weights='imagenet',
                 encoder_depth=5):
        super().__init__(input_ch, output_ch, base_ch=base_ch, num_iterations=num_iterations, encoder_name=encoder_name,
                         encoder_weights=encoder_weights, encoder_depth=encoder_depth)

    def forward(self, x):  # shape: (B, 6, H, W)

        predictions = []

        # 第一次 U-Net 处理
        pred_1 = self.second_u(x)  # shape: (B, 2, H, W)
        bv = x[:, 4:5, :, :]
        x = x[:, 0:3, :, :]  # (B, 3, H, W)
        a_logits = pred_1[:, 0, :, :].unsqueeze(1)
        v_logits = pred_1[:, 1, :, :].unsqueeze(1)
        predictions.append(torch.cat((a_logits, v_logits), dim=1))

        pred = pred_1  # shape: (B, 2, H, W)
        # 如果不需要 bv / a / v 的拆分，可以直接迭代 refine
        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)  # 映射到 0-1
            pred = torch.cat((x, pred[:, 0:1], bv, pred[:, 1:2]), dim=1)  # (B, I, H, W)
            pred = self.second_u(pred)  # shape: (B, 2, H, W)
            predictions.append(pred)

        return predictions
