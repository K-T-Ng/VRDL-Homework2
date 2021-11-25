import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ======================== Mish activation ====================
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ========================= custom layer =======================
class Convolutional(nn.Module):
    def __init__(self, batch_norm, filters_in, filters_out,
                 size, stride, pad, activation):
        '''
        convert [convolutional] block from yolov3.cfg to nn.Module
        [convolutional]
        batch_normalize: (bool), use batch_norm or not
        filters_in     : (int), number of input channel
        filters_out    : (int), number of output channel
        size           : (int), kernel_size
        stride         : (int), stride
        pad            : (bool), if True, padding size = kernel size / 2
                                 if False, padding size = 0
        activation     : (str), the name of activation function
        '''
        super(Convolutional, self).__init__()

        self.use_bn = batch_norm
        self.activation = activation

        padding = (size - 1) // 2 if pad else 0
        self.conv = nn.Conv2d(in_channels=filters_in,
                              out_channels=filters_out,
                              stride=stride,
                              kernel_size=size,
                              padding=padding,
                              bias=not self.use_bn)  # bn has bias already

        if self.use_bn:
            self.batchnorm = nn.BatchNorm2d(filters_out)

        if activation:
            if activation == 'leaky':
                self.activate = nn.LeakyReLU()
            if activation == 'mish':
                self.activate = Mish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.activation:
            x = self.activate(x)
        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x2, x1), dim=1)
        return out


# ======================= custom residual block ======================
class Residual_Block(nn.Module):
    def __init__(self, filters_in, filters_embed, filters_out):
        super(Residual_Block, self).__init__()
        self.conv1 = Convolutional(True, filters_in, filters_embed,
                                   size=1, stride=1, pad=1, activation="leaky")
        self.conv2 = Convolutional(True, filters_embed, filters_out,
                                   size=3, stride=1, pad=1, activation="leaky")

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = short_cut + x

        return out


# ======================= Darknet 53 backbone =======================
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        '''
        Network architecture is followed from the yolov3.cfg
        '''
        # convolution
        self.conv_0_0 = Convolutional(filters_in=3, filters_out=32,
                                      size=3, stride=1, pad=True,
                                      batch_norm=True, activation='leaky')

        # down sampling convoluation
        self.conv_0_1 = Convolutional(filters_in=32, filters_out=64,
                                      size=3, stride=2, pad=True,
                                      batch_norm=True, activation='leaky')

        # residual block * 1
        f_in, f_out, f_embed = 64, 64, 32
        self.resblock_1_0 = Residual_Block(f_in, f_embed, f_out)

        # down sampling convoluation
        self.conv_1_1 = Convolutional(filters_in=64, filters_out=128,
                                      size=3, stride=2, pad=True,
                                      batch_norm=True, activation='leaky')

        # residual block * 2
        f_in, f_out, f_embed = 128, 128, 64
        self.resblock_2_0 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_2_1 = Residual_Block(f_in, f_embed, f_out)

        # down sampling convoluation
        self.conv_2_2 = Convolutional(filters_in=128, filters_out=256,
                                      size=3, stride=2, pad=True,
                                      batch_norm=True, activation='leaky')

        # residual block * 8
        f_in, f_out, f_embed = 256, 256, 128
        self.resblock_3_0 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_1 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_2 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_3 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_4 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_5 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_6 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_3_7 = Residual_Block(f_in, f_embed, f_out)

        # down sampling convoluation
        self.conv_3_8 = Convolutional(filters_in=256, filters_out=512,
                                      size=3, stride=2, pad=True,
                                      batch_norm=True, activation='leaky')

        # residual block * 8
        f_in, f_out, f_embed = 512, 512, 256
        self.resblock_4_0 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_1 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_2 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_3 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_4 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_5 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_6 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_4_7 = Residual_Block(f_in, f_embed, f_out)

        # down sampling convoluation
        self.conv_4_8 = Convolutional(filters_in=512, filters_out=1024,
                                      size=3, stride=2, pad=True,
                                      batch_norm=True, activation='leaky')

        # residual block * 4
        f_in, f_out, f_embed = 1024, 1024, 512
        self.resblock_5_0 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_5_1 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_5_2 = Residual_Block(f_in, f_embed, f_out)
        self.resblock_5_3 = Residual_Block(f_in, f_embed, f_out)

    def forward(self, x):
        '''
        There are 3 feature maps that need to output
            - after resblock_3_7
            - after resblock_4_7
            - after resblock_5_3
        '''
        # suppose x's shape is (3,608,608)
        x = self.conv_0_0(x)      # (32,608,608)

        x = self.conv_0_1(x)      # (64,304,304)
        x = self.resblock_1_0(x)  # (64,304,304)

        x = self.conv_1_1(x)      # (128,152,152)
        x = self.resblock_2_0(x)  # (128,152,152)
        x = self.resblock_2_1(x)  # (128,152,152)

        x = self.conv_2_2(x)      # (256,76,76)
        x = self.resblock_3_0(x)  # (256,76,76)
        x = self.resblock_3_1(x)  # (256,76,76)
        x = self.resblock_3_2(x)  # (256,76,76)
        x = self.resblock_3_3(x)  # (256,76,76)
        x = self.resblock_3_4(x)  # (256,76,76)
        x = self.resblock_3_5(x)  # (256,76,76)
        x = self.resblock_3_6(x)  # (256,76,76)
        small = self.resblock_3_7(x)  # (256,76,76)

        x = self.conv_3_8(small)  # (512,38,38)
        x = self.resblock_4_0(x)  # (512,38,38)
        x = self.resblock_4_1(x)  # (512,38,38)
        x = self.resblock_4_2(x)  # (512,38,38)
        x = self.resblock_4_3(x)  # (512,38,38)
        x = self.resblock_4_4(x)  # (512,38,38)
        x = self.resblock_4_5(x)  # (512,38,38)
        x = self.resblock_4_6(x)  # (512,38,38)
        medium = self.resblock_4_7(x)  # (512,38,38)

        x = self.conv_4_8(medium)  # (1024,19,19)
        x = self.resblock_5_0(x)   # (1024,19,19)
        x = self.resblock_5_1(x)   # (1024,19,19)
        x = self.resblock_5_2(x)   # (1024,19,19)
        large = self.resblock_5_3(x)   # (1024,19,19)

        return small, medium, large


# ==================== yolo fpn =================================
class FPN_YOLOV3(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = filters_in
        fo_0, fo_1, fo_2 = filters_out

        fi_1, fi_2 = fi_1 + 256, fi_2 + 128

        # large
        self.large_convset = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=512, filters_out=1024, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=1024, filters_out=512, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=512, filters_out=1024, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=1024, filters_out=512, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
        )
        self.large_conv_1 = \
            Convolutional(filters_in=512, filters_out=1024, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish')
        self.large_conv_2 = \
            Convolutional(filters_in=1024, filters_out=fo_0, size=1, stride=1,
                          pad=True, batch_norm=True, activation=None)

        # medium
        self.medium_conv_0 = \
            Convolutional(filters_in=512, filters_out=256, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish')
        self.medium_upsample = Upsample(scale_factor=2)
        self.medium_route = Route()

        self.medium_convset = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=256, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=256, filters_out=512, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=512, filters_out=256, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=256, filters_out=512, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=512, filters_out=256, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
        )
        self.medium_conv_1 = \
            Convolutional(filters_in=256, filters_out=512, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish')
        self.medium_conv_2 = \
            Convolutional(filters_in=512, filters_out=fo_1, size=1, stride=1,
                          pad=True, batch_norm=True, activation=None)

        # small
        self.small_conv_0 = \
            Convolutional(filters_in=256, filters_out=128, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish')
        self.small_upsample = Upsample(scale_factor=2)
        self.small_route = Route()

        self.small_convset = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=128, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=128, filters_out=256, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=256, filters_out=128, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=128, filters_out=256, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
            Convolutional(filters_in=256, filters_out=128, size=1, stride=1,
                          pad=True, batch_norm=True, activation='mish'),
        )
        self.small_conv_1 = \
            Convolutional(filters_in=128, filters_out=256, size=3, stride=1,
                          pad=True, batch_norm=True, activation='mish')
        self.small_conv_2 = \
            Convolutional(filters_in=256, filters_out=fo_2, size=1, stride=1,
                          pad=True, batch_norm=True, activation=None)

    def forward(self, x0, x1, x2):
        '''
        x0(19,19,1024) large
        x1(38,38,512)  medium
        x2(76,76,256)  small
        '''
        # large
        temp_L = self.large_convset(x0)
        out_L = self.large_conv_1(temp_L)
        out_L = self.large_conv_2(out_L)  # (19,19,255)

        # medium
        temp_M = self.medium_conv_0(temp_L)
        temp_M = self.medium_upsample(temp_M)
        temp_M = self.medium_route(x1, temp_M)
        temp_M = self.medium_convset(temp_M)
        out_M = self.medium_conv_1(temp_M)
        out_M = self.medium_conv_2(out_M)  # (38,38,255)

        # small
        temp_S = self.small_conv_0(temp_M)
        temp_S = self.small_upsample(temp_S)
        temp_S = self.small_route(x2, temp_S)
        temp_S = self.small_convset(temp_S)
        out_S = self.small_conv_1(temp_S)
        out_S = self.small_conv_2(out_S)  # (76,76,255)

        return out_S, out_M, out_L  # small, medium, large


# ======================= yolo head ====================
class Yolo_head(nn.Module):
    def __init__(self, num_class, anchors, stride):
        super(Yolo_head, self).__init__()

        self.anchors = anchors  # (n_anchors, 2) tensor, indicating W and H
        self.stride = stride    # int
        self.num_cls = num_class
        self.num_anc = len(anchors)

    def forward(self, p):
        '''
        p: (torch.tensor), a feature map that feed in this yolo head
            shape=(batch size, channels, num grid, num grid)
            for example
                In yolov3, suppose that
                    - input image shape=(4,3,608,608)
                    - num_cls=80
                    - num_anc=3
                There are 3 possible input for yolo head
                    - (4, 255, 19, 19)
                    - (4, 255, 38, 38)
                    - (4, 255, 76, 76)
                where 255 = num_anc * ( 4 + 1 + num_cls )
                    - 4 positional info
                    - 1 objectness prediction
                    - num_cls class predictions
        '''
        bs, num_grid = p.size(0), p.size(3)

        # p (4, 255, 19, 19) -> (4, 3, 85, 19, 19) -> (4, 19, 19, 3, 85)
        p = p.view(bs, self.num_anc, 5+self.num_cls, num_grid, num_grid)
        p = p.permute(0, 3, 4, 1, 2)

        p_de = self.decode(p.clone())
        # ((4,19,19,3,85), (4,19,19,3,85)): (predict, predict_decode)
        return (p, p_de)

    def decode(self, p):
        # p(4, 19, 19, 3, 85)
        bs, num_grid = p.shape[:2]

        device = p.device
        stride = self.stride
        anchors = (1.0 * self.anchors).to(device)

        txty = p[..., 0:2]  # (4,19,19,3,2)
        twth = p[..., 2:4]  # (4,19,19,3,2)
        conf = p[..., 4:5]  # (4,19,19,3,1)
        prob = p[..., 5:]   # (4,19,19,3,80)

        # y: (19) -> (19,1) -> (19,19),  y[i][j] = i
        # x: (19) -> (1,19) -> (19,19),  x[i][j] = j
        y = torch.arange(0, num_grid).unsqueeze(1).repeat(1, num_grid)
        x = torch.arange(0, num_grid).unsqueeze(0).repeat(num_grid, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(bs, 1, 1, 3, 1)
        grid_xy = grid_xy.float().to(device)

        pred_xy = (torch.sigmoid(txty) + grid_xy) * stride
        pred_wh = (torch.exp(twth).clamp(max=cfg.TRAIN["CLAMP"]) * anchors) *\
            stride
        pred_conf = torch.sigmoid(conf)
        pred_prob = torch.sigmoid(prob)
        pred_bbox = torch.cat([pred_xy, pred_wh, pred_conf, pred_prob], dim=-1)
        # (4,19,19,3,85)

        if not self.training:
            return pred_bbox.view(-1, 5+self.num_cls)
        else:
            return pred_bbox


# ========================== yolov3 =================================
class Yolov3(nn.Module):
    def __init__(self, init_weights=True):
        super(Yolov3, self).__init__()

        self.anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.n_class = cfg.DATA["NUM"]
        self.out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.n_class + 5)

        self.backbone = Darknet53()
        self.fpn = FPN_YOLOV3(filters_in=[1024, 512, 256],
                              filters_out=[self.out_channel,
                                           self.out_channel,
                                           self.out_channel])

        # small
        self.small_head = Yolo_head(num_class=self.n_class,
                                    anchors=self.anchors[0],
                                    stride=self.strides[0])

        # medium
        self.medium_head = Yolo_head(num_class=self.n_class,
                                     anchors=self.anchors[1],
                                     stride=self.strides[1])

        # large
        self.large_head = Yolo_head(num_class=self.n_class,
                                    anchors=self.anchors[2],
                                    stride=self.strides[2])

        if init_weights:
            self.init_weights()

    def forward(self, x):
        out = []
        # If x(bs,3,608,608), 3 anchors per scale, 80 classes

        # x_s(bs,256,76,76), x_m(bs,512,38,38), x_l(bs,1024,19,19)
        x_s, x_m, x_l = self.backbone(x)

        # x_s(bs,255,76,76), x_m(bs,255,38,38), x_l(bs,255,19,19)
        x_s, x_m, x_l = self.fpn(x_l, x_m, x_s)

        out.append(self.small_head(x_s))
        out.append(self.medium_head(x_m))
        out.append(self.large_head(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def load_darknet_weights(self, weight_file, cutoff=52):
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # header
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            # the only weights in yolov3 is convolutional layer
            if isinstance(m, Convolutional):
                if count == cutoff:
                    break
                count += 1

                conv_layer = m.conv
                if m.use_bn:
                    # Load BN bias, weights, running mean and variance
                    bn_layer = m.batchnorm
                    num_b = bn_layer.bias.numel()

                    bn_b = torch.from_numpy(weights[ptr:ptr+num_b])
                    bn_b = bn_b.view_as(bn_layer.bias.data)
                    ptr += num_b

                    bn_w = torch.from_numpy(weights[ptr:ptr+num_b])
                    bn_w = bn_w.view_as(bn_layer.weight.data)
                    ptr += num_b

                    bn_rm = torch.from_numpy(weights[ptr:ptr+num_b])
                    bn_rm = bn_rm.view_as(bn_layer.running_mean)
                    ptr += num_b

                    bn_rv = torch.from_numpy(weights[ptr:ptr+num_b])
                    bn_rv = bn_rv.view_as(bn_layer.running_var)
                    ptr += num_b

                    # print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias (use bias iff not use bn)
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr+num_b])
                    conv_b = conv_b.view_as(conv_layer.bias.data)
                    ptr += num_b

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr+num_w])
                conv_w = conv_w.view_as(conv_layer.weight.data)
                ptr += num_w

                # print("loading weight {}".format(conv_layer))

if __name__ == '__main__':
    net = Yolov3()

    in_img = torch.randn(2, 3, 608, 608)
    p, p_d = net(in_img)

    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
