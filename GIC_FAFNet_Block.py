import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv,autopad

__all__ = (
    "MFAAM",
    "DEM",
    "MFIAM"
)
#*************FAPN*************
class MFAAM(nn.Module):


    def __init__(self, channels, out_channels,r=4):
        super(MFAAM, self).__init__()
        inter_channels = int(channels[0] // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels[0], inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 3 * channels[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3 * channels[0]),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[0], inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 3 * channels[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3 * channels[0]),
        )

        self.sigmoid = nn.Sigmoid()
        self.schannels = int(channels[0])
        self.tail_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x_list):
        x_low, x, x_high = x_list

        xa = x + x_low + x_high

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg

        split_c1, split_c2, split_c3 = torch.split(xlg, ([self.schannels,self.schannels,self.schannels]), dim=1)

        sigmoid1 = self.sigmoid(split_c1)
        sigmoid2 = self.sigmoid(split_c2)
        sigmoid3 = self.sigmoid(split_c3)

        out = x_low * sigmoid1 + x * sigmoid2 + x_high * sigmoid3
        out = self.tail_conv(out)

        return out

class DCVZL(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_ * 4, c2, 1)  # optional act=FReLU(c2)
        self.cv2 = Conv(c1, c1, 3, 1)
        self.dcn = DCNv2(c_,c_,kernel_size=3)
        self.cbam =CBAM(c2)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        x00 = self.cv1(x)
        x01 = self.cv2(x)

        x10 = self.cv1(x01)
        x11 = self.cv2(x01)

        x20 = self.cv1(x11)
        x21 = self.cv1(x11)
        x21 = self.dcn(x21)
        x21 = self.cbam(x21)

        x3 = torch.cat([x00,x10,x20,x21],dim=1)

        return self.cv3(x3)


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class DEM(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(4 * c_, c2, 1)  # optional act=FReLU(c2)
        self.cv2 = Conv(c1, c1, 3, 1)
        self.dcn = DCNv2(c1,c1,kernel_size=3)
        self.cbam =CBAM(c1=c1)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        x00 = self.cv1(x)
        x01 = self.cv2(x)

        x10 = self.cv1(x01)
        x11 = self.dcn(x01)
        x11 = self.cbam(x11)

        x3 = torch.cat([x00,x10,x11],dim=1)

        return self.cv3(x3)

#*************MFIAB*************
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x

class Mamba_inception(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, g=1, e=1.0):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv3 = Conv(c1, c_, 3, 1)
        self.cv4 = Conv(c_, c2, 3, 1, d=2)
        self.cv5 = Conv(c1, c_, 3, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(c_, c2, 3, 1, d=3)
        self.attention1 = MLCA(c2)
        self.attention2 = MLCA(c1)

        self.ln_1 = nn.LayerNorm(c1)
        self.self_attention = SS2D(d_model=c1, d_state=16, dropout=0.05)
        self.drop_path = DropPath(0.05)

    def forward(self, x):
        x1 = x.permute((0, 2, 3, 1))
        x2 = self.ln_1(x1)
        x2 = x2.permute((0, 3, 2, 1))
        x3 = self.self_attention(x2)
        x3 = x3.permute((0, 1, 2, 3))

        x4 = self.attention2(self.drop_path(x3))
        x1 = x1.permute(0,3,2,1)
        x4 = x1 + x4

        y1 = self.attention1(self.cv1(x))

        y2 = self.attention1(self.cv7(self.cv6(self.cv5(x))))

        y3 = self.attention1(self.cv4(self.cv3(x)))

        x4 = x4.permute(0,1,3,2)

        z = torch.cat((y1, y2, y3, x4), dim=1)

        return z

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.mamba_yolo import SS2D
class Basic(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((5 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MFIAM(Basic):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Mamba_inception(self.c, self.c, shortcut, g) for _ in range(n))