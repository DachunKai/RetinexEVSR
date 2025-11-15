import math
import numbers

import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


def to_3d(x):
    """Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    """Restore spatial dimensions: (B, H*W, C) → (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class StoDepth_BasicBlock(nn.Module):
    """
        Stochastic depth residual block. Stochastic Depth allows skipping
        layers randomly during training.
    """

    expansion = 1

    def __init__(self, prob=0.5, m=None, multFlag=False, inplanes=64, planes=64, stride=1, downsample=None):
        """
        :param prob: Probability for stochastic depth (default 0.5)
        :param m: Bernoulli sampling distribution
        :param multFlag: If True, multiply output by self.prob
        :param inplanes: Number of input channels
        :param planes: Number of output channels
        :param stride: Convolution stride
        :param downsample: Downsample layer for residual connection
        """
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = m
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class Guide(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=1, input_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = Basic2d(input_planes*2, input_planes, norm_layer)

    def forward(self, feat, weight):
        weight = torch.cat((feat, weight), dim=1)
        weight = self.conv(weight)
        return weight


class BiasFree_LayerNorm(nn.Module):
    """LayerNorm without bias."""

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """LayerNorm with learnable bias."""

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Apply LayerNorm (bias-free or with bias) over channel dimension."""

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class Mlp(nn.Module):
    """Feed-forward block with depth-wise convolution."""
    
    def __init__(self, dim=64, out_dim=64, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim,
                                              kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, input_size):
        # bs x hw x c
        B, L, C = x.size()
        H, W = input_size
        assert H * W == L, "output H x W is not the same with L!"

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)  # bs, hidden_dim, 32x32
        x = self.dwconv(x)

        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        return x


def Forward(x, model):
    """Apply ``model`` to each tensor in a feature sequence."""

    feat = []
    t = len(x)
    for i in range(0, t):
        feat_i = x[i]
        feat_i = model(feat_i)
        feat.append(feat_i)
    return feat


class CrossAttn(nn.Module):
    """Cross-attention module for image and event features."""
    
    def __init__(self, dim=32, num_heads=4, bias=False, proj_d=4):
        super().__init__()
        self.num_heads = num_heads
        self.proj_d = proj_d  # k, v projection depth
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

        self.conv_k = nn.Conv2d(dim, dim * proj_d, kernel_size=3, padding=1, stride=1, groups=dim, bias=False)
        self.conv_v = nn.Conv2d(dim, dim * proj_d, kernel_size=3, padding=1, stride=1, groups=dim, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        k = self.conv_k(k)
        v = self.conv_v(v)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # q shape: [b, head, c, h*w]
        k = rearrange(k, 'b (head d c) h w -> b head d c (h w)', head=self.num_heads, d=self.proj_d)  # k shape: [b, head, d, c, h*w]
        v = rearrange(v, 'b (head d c) h w -> b head d c (h w)', head=self.num_heads, d=self.proj_d)

        # Normalize query and key
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute attention for each d
        attn_list = []
        scaling_factor = math.sqrt(c)  # Scaling factor for scaled dot-product attention
        for i in range(self.proj_d):
            # Compute scaled attention for each d (stacked into the head dimension)
            attn = (q @ k[:, :, i, :, :].transpose(-2, -1)) / scaling_factor  # scaled attn shape: [b, head, c, c]
            attn = attn.softmax(dim=-1)
            attn_list.append(attn)

        # Stack the attention maps along the d dimension
        attn = torch.stack(attn_list, dim=2)  # attn shape: [b, head, d, c, c]

        # Aggregate value features
        out = 0  # initialize to zero
        for i in range(self.proj_d):
            # Perform attention-weighted sum for each d, then sum along the d dimension
            out += attn[:, :, i] @ v[:, :, i]  # out shape: [b, head, c, h*w]

        # Reshape back to original dimensions
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Apply projection and projection dropout
        out = self.project_out(out)
        return out


class PROP(nn.Module):
    """A single PROP module for video super-resolution."""
    def __init__(
            self,
            inp_dim,
            embed_dim,
            num_resblocks=5,
            has_event=False,
            event_dim=None,
            num_heads=4,
            bias=False,
            proj_d=4,
            ffn_expansion_factor=2,
            LayerNorm_type='WithBias'
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.embed_dim = embed_dim
        self.has_event = has_event
        if has_event:
            assert event_dim is not None, "event_dim must be specified when has_event is True."
            self.conv_before_fusion = nn.Conv2d(event_dim, embed_dim, 3, 1, 1)
            self.norm1_image = LayerNorm(embed_dim, LayerNorm_type)
            self.norm1_event = LayerNorm(embed_dim, LayerNorm_type)
            self.attn_fusion = CrossAttn(dim=embed_dim, num_heads=num_heads, bias=bias, proj_d=proj_d)
            self.norm2 = nn.LayerNorm(embed_dim)
            mlp_hidden_dim = int(embed_dim * ffn_expansion_factor)
            self.ffn = Mlp(dim=embed_dim, out_dim=embed_dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.vsr_backbone = ConvResidualBlocks(self.inp_dim, embed_dim, num_resblocks)

    def forward(self, x, flows, events=None):
        """
        :param x: List[ Tensor[n,c,h,w] ], length is t
        :param flows:  List[ Tensor[n,c,h,w] ], length is t-1
        :param events:  List[ Tensor[n,c,h,w] ], length is t
        :return outs: List[ Tensor[n,c,h,w] ], length is t
        """
        assert len(flows) == len(x) - 1
        t = len(x)
        n,c,h,w = x[0].shape

        outs = []
        feat_prop = flows[0].new_zeros(n, c, h, w)
        for i in range(0, t):
            feat_current = x[i]
            if self.has_event:
                feat_event = events[i]
                feat_event = self.conv_before_fusion(feat_event)
            if i > 0:
                flow_n1 = flows[i-1]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = outs[-2]
                    flow_n2 = flows[i-2]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
            else:
                cond = torch.cat([feat_current, feat_current, feat_current], dim=1)

            feat_prop = self.vsr_backbone(cond)

            if self.has_event:
                feat_prop = feat_prop + self.attn_fusion(feat_prop, feat_event)

                # mlp
                feat_prop = to_3d(feat_prop) # b, h*w, c
                feat_prop = feat_prop + self.ffn(self.norm2(feat_prop), (h, w))
                feat_prop = to_4d(feat_prop, h, w)

            outs.append(feat_prop)
        return outs


class PROPs(nn.Module):
    """A sequence of PROP modules for video super-resolution."""
    def __init__(
            self,
            inp_dim,
            embed_dim,
            num_resblocks=5,
            num_PROP=1,
            reverse=(True, False),
            has_event=False,  # include has_event parameter
            event_dim=None,  # default event_dim
            proj_d=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PROP(
                inp_dim=inp_dim,
                embed_dim=embed_dim,
                num_resblocks=num_resblocks,
                has_event=has_event,  # pass has_event
                event_dim=event_dim if has_event else None,  # pass event_dim only if has_event is True
                proj_d=proj_d
            )
            for _ in range(num_PROP)
        ])
        self.reverse = reverse
        self.has_event = has_event  # store has_event state

    def forward(self, features, flows, events=None):
        """
        :param features: List[Tensor[n,c,h,w]], length is t
        :param flows: List[Tensor[n,c,h,w]], length is t-1
        :param events: List[Tensor[n,c,h,w]] or None, length is t (optional)
        :return x: List[Tensor[n,c,h,w]], length is t
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            reverse = self.reverse[i]

            if not reverse:
                if self.has_event and events is not None:
                    features = layer(features, flows, events=events)  # pass events
                else:
                    features = layer(features, flows)
            else:
                if self.has_event and events is not None:
                    features = layer(features[::-1], flows[::-1], events=events[::-1])  # pass reversed events
                else:
                    features = layer(features[::-1], flows[::-1])
                features = features[::-1]

        return features



class GuideFeatExt(nn.Module):
    def __init__(self, bc=2, prob=0.5, block=StoDepth_BasicBlock, multFlag=True, layers=(2, 2, 2, 2), norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=1):
        super().__init__()
        self._norm_layer = norm_layer

        prob_0_L = (1, prob)
        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        self.conv_event = Basic2d(5, bc * 2, norm_layer=None, kernel_size=5, padding=2)
        in_channels = bc * 2

        self.inplanes = in_channels
        self.layer1_img, self.layer1_event = self._make_layer(block, in_channels * 2, layers[0], stride=1)
        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)

        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img, self.layer2_event = self._make_layer(block, in_channels * 4, layers[1], stride=2)
        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)

        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img, self.layer3_event = self._make_layer(block, in_channels * 8, layers[2], stride=2)
        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img, self.layer4_event = self._make_layer(block, in_channels * 8, layers[3], stride=2)

        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer1d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)

        self.conv = Basic2d(in_channels * 2, in_channels, norm_layer)

    def forward(self, img, event):
        """
        :param img: [B, 3, H, W]
        :param eve: [B, 5, H, W]
        """
        c0_img = self.conv_img(img) # [B, 8, H, W]
        c0_event = self.conv_event(event) # [B, 8, H, W]

        c1_img = self.layer1_img(c0_img) # [B, 16, H, W]
        c1_event = self.layer1_event(c0_event) # [B, 16, H, W]
        c1_event_dyn = self.guide1(c1_event, c1_img) # [B, 16, H, W]

        c2_img = self.layer2_img(c1_img) # [B, 32, H/2, W/2]
        c2_event = self.layer2_event(c1_event_dyn) # [B, 32, H/2, W/2]
        c2_event_dyn = self.guide2(c2_event, c2_img) # [B, 32, H/2, W/2]

        c3_img = self.layer3_img(c2_img) # [B, 64, W/4, W/4]
        c3_event = self.layer3_event(c2_event_dyn) # [B, 64, H/4, W/4]
        c3_event_dyn = self.guide3(c3_event, c3_img) # [B, 64, H/4, W/4]

        c4_img = self.layer4_img(c3_img) # [B, 64, H/8, W/8]
        c4_event = self.layer4_event(c3_event_dyn) # [B, 64, H/8, W/8]

        c4 = c4_img + c4_event # [B, 64, H/8, W/8]
        dc3 = self.layer3d(c4) # [B, 64, H/4, W/4]
        c3 = dc3 + c3_event_dyn # [B, 64, H/4, W/4]

        dc2 = self.layer2d(c3) # [B, 32, H/2, W/2]
        c2 = dc2 + c2_event_dyn # [B, 32, H/2, W/2]

        dc1 = self.layer1d(c2) # [B, 16, H, W]
        c1 = dc1 + c1_event_dyn # [B, 16, H, W]
        c1 = self.conv(c1) # [B, 8, H, W]
        c0 = c1 + c0_event # [B, 8, H, W]

        return c0_event, c3, c2, c0


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        img_downsample, depth_downsample = None, None
        if stride != 1 or self.inplanes != planes * block.expansion:
            img_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            depth_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        img_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, img_downsample)]
        depth_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, depth_downsample)]
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            img_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            depth_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*img_layers), nn.Sequential(*depth_layers)


class ChannelAttn(nn.Module):
    """Channel attention block (CBAM-style)."""

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super(ChannelAttn, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        x_avg = x.mean((2, 3), keepdim=True)
        x_max = F.adaptive_max_pool2d(x, 1)
        x_avg = self.fc2(self.act(self.fc1(x_avg)))
        x_max = self.fc2(self.act(self.fc1(x_max)))
        x_attn = x_avg + x_max
        return x * x_attn.sigmoid()
