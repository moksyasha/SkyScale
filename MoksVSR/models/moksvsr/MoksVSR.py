import torch
import torch.nn as nn
import torch.nn.functional as F
from models.flow.FastFlowNet_v2 import FastFlowNet
import time
from torchvision.ops import deform_conv2d
import math


class ResBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    
    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class BottleneckResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()

        # First convolutional block with 1x1 kernel
        self.conv1 = nn.Conv2d(num_filters, num_filters//2, 1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Second convolutional block with 3x3 kernel
        self.conv2 = nn.Conv2d(num_filters//2, num_filters//2, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Third convolutional block with 1x1 kernel and no activation function
        self.conv3 = nn.Conv2d(num_filters//2, num_filters, 1, stride=1, padding=0)
    
    def forward(self, x):
        # input through the conv layers
        x_out = self.relu1(self.conv1(x))
        x_out = self.relu2(self.conv2(x_out))
        # skip connection
        return x + self.conv3(x_out)


class ResBlocksWithInputConv(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, num_blocks=30):
        super().__init__()
        # input conv
        self.layers = []
        self.layers.append(nn.Conv2d(input_channels, num_filters, 3, stride=1, padding=1))
        self.layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        self.layers.append(nn.Sequential(*[BottleneckResBlock(num_filters) for _ in range(num_blocks)]))
        self.result = nn.Sequential(*self.layers)

    def forward(self, feat):
        return self.result(feat)


class MoksVSR(nn.Module):
    def __init__(self, nb=30, nc=64):
        super().__init__()
        self.num_blocks = nb
        self.mid_channel = nc
        self.fastflow = FastFlowNet().cuda().eval()
        self.fastflow.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_mix.pth'))
        self.lrelu = nn.LeakyReLU(0.1)
        self.forward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
        self.backward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
     ##
        self.fusion = nn.Conv2d(2*self.mid_channel, self.mid_channel, 3, stride=1, padding=1)
        self.up_conv1 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
        self.hr_conv = nn.Conv2d(self.mid_channel, 3, 3, stride=1, padding=1)

    def flow_warp(self, x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
        '''Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        '''
        _N, _C, H, W = x.shape
        device = flow.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=x.dtype),
            torch.arange(0, W, device=device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid_flow = flow.permute((0, 2, 3, 1)) + grid # flow shape: NCHW to NHWC
        
        # normalize grid flows to [-1, 1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def compute_flows_fast(self, lrs):
        # compute optical flow by FastFlowNet
        N, T, C, H, W = lrs.shape
        # 19 6 180 320
        x_start = lrs[:, :-1, :, :, :].reshape(-1, C, H, W) # reshape from (N, T, C, H, W) into (N*T*C, H, W)
        x_end = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)
        forward_flows = self.fastflow(torch.cat((x_start, x_end), 1))#.view(N, T-1, 2, H, W)
        backward_flows = self.fastflow(torch.cat((x_end, x_start), 1))#.view(N, T-1, 2, H, W)
        return forward_flows.unsqueeze(0), backward_flows.unsqueeze(0)

    def forward(self, lr):
        """Forward function.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w)
            batch of clips, imgs per clip, channels, h, w
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w)
        """
        #1 20 3 180 320
        N, T, C, H, W = lr.shape
        # print(N, T, C, H, W)
        # flows shape
        # 19 2 180 320

        forward_flows, backward_flows = self.compute_flows_fast(lr)

        # backward propagation
        # 1 64 h w
        features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        outputs = []
        for t in range(T-1, -1, -1):
            if t != T-1: # warp features if not looking at last frame of clip 
                features = self.flow_warp(features, backward_flows[:, t, :, :, :]) # feature warping (W step)
            features = torch.cat((features, lr[:, t, :, :, :]), dim=1) # concatenate NCHW tensors along C axis
            features = self.backward_res(features) # residual processing (R step)
            outputs.append(features)
        outputs = outputs[::-1]
        
        # forward propagation
        features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        for t in range(T):
            lr_curr = lr[:, t, :, :, :]
            if t != 0: # warp features if not looking at first frame of clip 
                features = self.flow_warp(features, forward_flows[:, t-1, :, :, :]) # feature warping (W step)
            features = torch.cat((features, lr_curr), dim=1)
            features = self.forward_res(features) # residual processing (R step)
        
            # fuse forward and backward features, then upsample
            bidirectional_features = torch.cat((features, outputs[t]), dim=1)
            fused = self.lrelu(self.fusion(bidirectional_features))
            out = F.pixel_shuffle(self.lrelu(self.up_conv1(fused)), 2)
            out = F.pixel_shuffle(self.lrelu(self.up_conv2(out)), 2)
            out = self.hr_conv(out)
            out += F.interpolate(lr_curr, scale_factor=4.0, mode='bilinear', align_corners=False) # add residuals to upsampled input
            outputs[t] = out
            
        return torch.stack(outputs, dim=1) # turn list of NCHW tensors with length T into a NTCHW tensor


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class SecondOrder(nn.Module):
    def __init__(self, input_channels=128, out_channels=64, kernel=3,
    deform_groups=16, max_residue_magnitude=10):
        super().__init__()

        self.out_channels = out_channels
        self.deform_groups = deform_groups
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.max_residue_magnitude = max_residue_magnitude

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1), # 3 * kernel3 * kernel3 * groups
        )

        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, input_channels, 3, 3))
        #self.groups = 1                 
        #self.deform_groups = 16
        self.bias = nn.Parameter(torch.Tensor(out_channels))                
        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """Forward function."""
        # (feat_prop, cond(cond_n1, feat_current, cond_n2), flow_n1, flow_n2)
        # 1 196 h w
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)

        # 1 432 h w
        out = self.conv_offset(extra_feat)

        # 1 144 h w
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)
        # x    1 128 h w
        # off  1 288 180 320
        # mask 1 144 180 320
        return deform_conv2d(x, offset, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, mask)
        # return deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, self.groups,
        #                                self.deform_groups)


class DeformableAlignment(nn.Module):
    def __init__(self, input_channels=128, out_channels=64, kernel=3,
        deform_groups=8, max_residue_magnitude=10):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.deform_groups = deform_groups
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.max_residue_magnitude = max_residue_magnitude
        self.kernel = kernel
        # convolution for current feat and reference (feat_propagated, prev_feauter, flow1, flow2)
        # to get offsets and masks
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels * 3 + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, kernel * kernel * 3 * self.deform_groups, 3, 1, 1), # 3 * kernel3 * kernel3 * groups
        )
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.input_channels // 2, 3, 3))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))                
        self.init_weights()

    def init_weights(self):
        n = self.input_channels
        from torch.nn.modules.utils import _pair, _single
        for k in _pair(self.kernel):
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        """Init constant offset."""
        # constant_init(self.conv_offset[-1], val=0, bias=0)
        if hasattr(self.conv_offset[-1], 'weight') and self.conv_offset[-1].weight is not None:
            nn.init.constant_(self.conv_offset[-1].weight, 0)
        if hasattr(self.conv_offset[-1], 'bias') and self.conv_offset[-1].bias is not None:
            nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, current_feature_warped, feat_prop, feat_prev, flow1, flow2):
        out = self.conv_offset(torch.cat([current_feature_warped, feat_prop, feat_prev, flow1, flow2], dim=1))
        # here we can plus optical flow
        offset1, offset2, mask = torch.chunk(out, 3, dim=1)

        offset1 = offset1 + flow1.flip(1).repeat(1,
                                                    offset1.size(1) // 2, 1,
                                                    1)
        offset2 = offset2 + flow2.flip(1).repeat(1,
                                                    offset2.size(1) // 2, 1,
                                                    1)
        # 1 144
        offset = torch.cat((offset1, offset2), dim=1)
        # 1 72
        mask = torch.sigmoid(mask)
        # curr feat 1 64 h w
        return deform_conv2d(current_feature_warped, offset, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, mask)


class MoksPlus(nn.Module):
    def __init__(self, nb=10, nf=64):
        super().__init__()
        self.num_blocks = nb
        self.num_features = nf
        self.fastflow = FastFlowNet().cuda().eval()
        self.fastflow.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_mix.pth'))
        
        self.feat_extract = ResBlocksWithInputConv(input_channels=3, num_filters=64, num_blocks=30)

        self.max_residue_magnitude=10
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward1', 'forward1']
        for i, module in enumerate(modules):
            self.deform_align[module] = DeformableAlignment(
                2 * self.num_features,
                self.num_features,
                kernel=3,
                deform_groups=16,
                max_residue_magnitude=self.max_residue_magnitude)
            self.backbone[module] = ResBlocksWithInputConv(
                2 * self.num_features, self.num_features, self.num_blocks)

        ##
        # upsampling module
        self.reconstruction = ResBlocksWithInputConv(
            3 * self.num_features, self.num_features, 5)

        self.psh_upsample = nn.Conv2d(
            self.num_features,
            self.num_features * 2 * 2,
            3,
            padding=1)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    
    def upsample(self, lr, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []

        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lr.size(1)):
            # get [0] features from forward and backward branch
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']

            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            #hr = [feature, back, forward] / 192
            # pass to rhe residual blocks
            hr = self.reconstruction(hr)
            hr = self.lrelu(F.pixel_shuffle(self.psh_upsample(hr), 2))
            hr = self.lrelu(F.pixel_shuffle(self.psh_upsample(hr), 2))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            hr += self.img_upsample(lr[:, i, :, :, :])
            
            outputs.append(hr)
        res = torch.stack(outputs, dim=1)
        #print(res)
        return res

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.
        """
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)

        if 'backward1' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_deform = flows.new_zeros(n, self.num_features, h, w)
        feat_current = flows.new_zeros(n, self.num_features, h, w)
        
        # init features for deform alignment
        feat_prev = flows.new_zeros(n, self.num_features, h, w)
        flow_prev = flows.new_zeros(n, self.num_features, h, w)

        frame = 2
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][idx]
            frame -= 1
            # first frame has no flow
            if i > 0:
                #get correspond flow
                flow_current = flows[:, flow_idx[i], :, :, :]

                # feature prealignment by flow (main feature)
                feat_current = self.flow_warp(feat_current, flow_current)
                
                if frame == 0:  # second-order features
                    flow_prev = flows[:, flow_idx[i-2], :, :, :]
                    flow_prev = flow_prev + self.flow_warp(flow_prev, flow_current)
                    feat_prev = feats["spatial"][frame_idx[i-2]]
                    feat_prev = self.flow_warp(feat_prev, flow_prev)
                    frame = 2

                # flow-guided deformable convolution
                # align current feature with reference feat_prop, feat_prev
                feat_deform = self.deform_align[module_name](feat_current, feat_deform,
                                    feat_prev, flow_prev, flow_current)
            
            # 1 128 h w
            # concat aligned by flow and by deform conv
            feat = torch.cat([feat_current] + [feat_deform], dim=1)

            # 1 64 h w
            # residuals propagation, alignment?
            back = self.backbone[module_name](feat)

            # save current feature
            # 1 64 h w
            feat_deform = feat_deform + back
            feats[module_name].append(feat_deform)

        if 'backward1' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, lr):
        """Forward function.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w)
            batch of clips, imgs per clip, channels, h, w
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w)
        """
        #1 20 3 180 320
        N, T, C, H, W = lr.shape
        # print(N, T, C, H, W)
        # flows shape
        # 3 2 180 320

        forward_flows, backward_flows = self.compute_flows_fast(lr)

        # backward propagation
        # 1 64 h w
        #features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        # 4 64 h w
        feats = {}
        feats_all = self.feat_extract(lr[0])
        feats_all = feats_all.view(N, T, -1, H, W)
        feats["spatial"] = [feats_all[:, i, :, :, :] for i in range(0, T)]

        # feature propagation
        #for iter_ in [1, 2]:
        for direction in ['backward', 'forward']:
            module = f'{direction}1'

            feats[module] = []
            if direction == 'backward':
                flows = backward_flows
            elif forward_flows is not None:
                flows = forward_flows
            else:
                flows = backward_flows.flip(1)

            feats = self.propagate(feats, flows, module)

        return self.upsample(lr, feats)    

    def flow_warp(self, x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
        '''Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        '''
        _N, _C, H, W = x.shape
        device = flow.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=x.dtype),
            torch.arange(0, W, device=device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid_flow = flow.permute((0, 2, 3, 1)) + grid # flow shape: NCHW to NHWC
        
        # normalize grid flows to [-1, 1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def compute_flows_fast(self, lrs):
        # compute optical flow by FastFlowNet
        N, T, C, H, W = lrs.shape
        # 19 6 180 320
        x_start = lrs[:, :-1, :, :, :].reshape(-1, C, H, W) # reshape from (N, T, C, H, W) into (N*T*C, H, W)
        x_end = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)
        forward_flows = self.fastflow(torch.cat((x_start, x_end), 1))#.view(N, T-1, 2, H, W)
        backward_flows = self.fastflow(torch.cat((x_end, x_start), 1))#.view(N, T-1, 2, H, W)
        return forward_flows.unsqueeze(0), backward_flows.unsqueeze(0)


class TestMoksVSRDouble(nn.Module):
    def __init__(self, nb=30, nc=64):
        super().__init__()
        self.num_blocks = nb
        self.mid_channel = nc
        self.fastflow = FastFlowNet().cuda().eval()
        self.fastflow.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_mix.pth'))
        # self.lrelu = nn.LeakyReLU(0.1)
    #     self.forward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
    #     self.backward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
    #  ##
    #  ##
    #     self.fusion = nn.Conv2d(2*self.mid_channel, self.mid_channel, 3, stride=1, padding=1)
    #     self.up_conv1 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
    #     self.up_conv2 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
    #     self.hr_conv = nn.Conv2d(self.mid_channel, 3, 3, stride=1, padding=1)
    ##
        self.feat_extract = ResBlocksWithInputConv(input_channels=3, num_filters=64, num_blocks=30)

        # propagation branches
        self.max_residue_magnitude=10
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrder(
                2 * self.mid_channel,
                self.mid_channel,
                kernel=3,
                deform_groups=16,
                max_residue_magnitude=self.max_residue_magnitude)
            self.backbone[module] = ResBlocksWithInputConv(
                (2 + i) * self.mid_channel, self.mid_channel, num_blocks=30)

        # upsampling module
        self.reconstruction = ResBlocksWithInputConv(
            5 * self.mid_channel, self.mid_channel, 5)
        # self.upsample1 = PixelShufflePack(
        #     mid_channels, mid_channels, 2, upsample_kernel=3)
        # self.upsample2 = PixelShufflePack(
        #     mid_channels, 64, 2, upsample_kernel=3)

        self.psh_upsample = nn.Conv2d(
            self.mid_channel,
            self.mid_channel * 2 * 2,
            3,
            padding=1)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def upsample(self, lr, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lr.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(F.pixel_shuffle(self.psh_upsample(hr), 2))
            hr = self.lrelu(F.pixel_shuffle(self.psh_upsample(hr), 2))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            hr += self.img_upsample(lr[:, i, :, :, :])
            
            outputs.append(hr)
        res = torch.stack(outputs, dim=1)
        #print(res)
        return res

    def flow_warp(self, x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
        '''Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        '''
        _N, _C, H, W = x.shape
        device = flow.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=x.dtype),
            torch.arange(0, W, device=device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid_flow = flow.permute((0, 2, 3, 1)) + grid # flow shape: NCHW to NHWC
        
        # normalize grid flows to [-1, 1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def compute_flows_fast(self, lrs):
        # compute optical flow by FastFlowNet
        N, T, C, H, W = lrs.shape
        # 19 6 180 320
        x_start = lrs[:, :-1, :, :, :].reshape(-1, C, H, W) # reshape from (N, T, C, H, W) into (N*T*C, H, W)
        x_end = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)
        forward_flows = self.fastflow(torch.cat((x_start, x_end), 1))#.view(N, T-1, 2, H, W)
        backward_flows = self.fastflow(torch.cat((x_end, x_start), 1))#.view(N, T-1, 2, H, W)
        return forward_flows.unsqueeze(0), backward_flows.unsqueeze(0)


    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        # PyTorch 2.0 could not compile data type of 'range'
        #frame_idx = range(0, t + 1)
        #flow_idx = range(-1, t)
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        # 1 64 hw
        feat_prop = flows.new_zeros(n, self.mid_channel, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]

            # second-order deformable alignment
            if i > 0:
                # 1 2 h w
                flow_n1 = flows[:, flow_idx[i], :, :, :]

                # 1 64 h w
                cond_n1 = self.flow_warp(feat_prop, flow_n1)

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + self.flow_warp(flow_n2,
                                                  flow_n1)
                    cond_n2 = self.flow_warp(feat_n2, flow_n2)

                # flow-guided deformable convolution
                # 1 192 h w
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                # 1 128 h w
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]

            # 1 128 h w
            feat = torch.cat(feat, dim=1)

            # 1 64 h w
            back = self.backbone[module_name](feat)
            # 1 64 h w
            feat_prop = feat_prop + back
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats


    def forward(self, lr):
        """Forward function.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w)
            batch of clips, imgs per clip, channels, h, w
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w)
        """
        #1 20 3 180 320
        N, T, C, H, W = lr.shape
        # print(N, T, C, H, W)
        # flows shape
        # 3 2 180 320

        forward_flows, backward_flows = self.compute_flows_fast(lr)

        # backward propagation
        # 1 64 h w
        #features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        # 4 64 h w
        feats = {}
        feats_all = self.feat_extract(lr[0])
        feats_all = feats_all.view(N, T, -1, H, W)
        feats["spatial"] = [feats_all[:, i, :, :, :] for i in range(0, T)]
        

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []
                if direction == 'backward':
                    flows = backward_flows
                elif forward_flows is not None:
                    flows = forward_flows
                else:
                    flows = backward_flows.flip(1)

                feats = self.propagate(feats, flows, module)

        return self.upsample(lr, feats)
        # warped_back = torch.empty((N, self.mid_channel, H, W))
        # for t in range(T-1, -1, -1):
        #     if t != T-1: # warp features if not looking at last frame of clip 
        #         features = self.flow_warp(features, backward_flows[:, t, :, :, :]) # feature warping (W step)

        # outputs = []
        # for t in range(T-1, -1, -1):
        #     if t != T-1: # warp features if not looking at last frame of clip 
        #         features = self.flow_warp(features, backward_flows[:, t, :, :, :]) # feature warping (W step)
        #     features = torch.cat((features, lr[:, t, :, :, :]), dim=1) # concatenate NCHW tensors along C axis
        #     features = self.backward_res(features) # residual processing (R step)
        #     outputs.append(features)
        # # 4 (1 64 h w)

        # outputs = outputs[::-1]
        
        # # forward propagation
        # features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        # for t in range(T):
        #     lr_curr = lr[:, t, :, :, :]
        #     if t != 0: # warp features if not looking at first frame of clip 
        #         features = self.flow_warp(features, forward_flows[:, t-1, :, :, :]) # feature warping (W step)
        #     features = torch.cat((features, lr_curr), dim=1)
        #     features = self.forward_res(features) # residual processing (R step)
        
        #     # fuse forward and backward features, then upsample
        #     bidirectional_features = torch.cat((features, outputs[t]), dim=1)
        #     fused = self.lrelu(self.fusion(bidirectional_features))
        #     out = F.pixel_shuffle(self.lrelu(self.up_conv1(fused)), 2)
        #     out = F.pixel_shuffle(self.lrelu(self.up_conv2(out)), 2)
        #     out = self.hr_conv(out)
        #     out += F.interpolate(lr_curr, scale_factor=4.0, mode='bilinear', align_corners=False) # add residuals to upsampled input
        #     outputs[t] = out
            
        # return torch.stack(outputs, dim=1) # turn list of NCHW tensors with length T into a NTCHW tensor


class TestMoksVSR(nn.Module):
    def __init__(self, nb=10, nc=16):
        super().__init__()
        self.num_blocks = nb
        self.mid_channel = nc
        self.fastflow = FastFlowNet().cuda().eval()
        self.fastflow.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_mix.pth'))
        self.lrelu = nn.LeakyReLU(0.1)
        self.forward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
        self.backward_res = nn.Sequential(nn.Conv2d(self.mid_channel+3, self.mid_channel, 3, stride=1, padding=1), self.lrelu, *[ResBlock(self.mid_channel) for _ in range(self.num_blocks)])
     ##
        self.fusion = nn.Conv2d(2*self.mid_channel, self.mid_channel, 3, stride=1, padding=1)
        self.up_conv1 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(self.mid_channel, 4*self.mid_channel, 3, stride=1, padding=1)
        self.hr_conv = nn.Conv2d(self.mid_channel, 3, 3, stride=1, padding=1)

    def flow_warp(self, x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
        '''Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        '''
        _N, _C, H, W = x.shape
        device = flow.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=x.dtype),
            torch.arange(0, W, device=device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid_flow = flow.permute((0, 2, 3, 1)) + grid # flow shape: NCHW to NHWC
        
        # normalize grid flows to [-1, 1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def compute_flows_fast(self, lrs):
        # compute optical flow by FastFlowNet
        N, T, C, H, W = lrs.shape
        # 19 6 180 320
        x_start = lrs[:, :-1, :, :, :].reshape(-1, C, H, W) # reshape from (N, T, C, H, W) into (N*T*C, H, W)
        x_end = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)
        forward_flows = self.fastflow(torch.cat((x_start, x_end), 1))#.view(N, T-1, 2, H, W)
        backward_flows = self.fastflow(torch.cat((x_end, x_start), 1))#.view(N, T-1, 2, H, W)
        return forward_flows.unsqueeze(0), backward_flows.unsqueeze(0)

    def forward(self, lr):
        """Forward function.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w)
            batch of clips, imgs per clip, channels, h, w
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w)
        """
        #1 20 3 180 320
        N, T, C, H, W = lr.shape
        # print(N, T, C, H, W)
        # flows shape
        # 19 2 180 320

        forward_flows, backward_flows = self.compute_flows_fast(lr)

        # backward propagation
        # 1 64 h w
        features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        outputs = []
        for t in range(T-1, -1, -1):
            if t != T-1: # warp features if not looking at last frame of clip 
                features = self.flow_warp(features, backward_flows[:, t, :, :, :]) # feature warping (W step)
            features = torch.cat((features, lr[:, t, :, :, :]), dim=1) # concatenate NCHW tensors along C axis
            features = self.backward_res(features) # residual processing (R step)
            outputs.append(features)
        outputs = outputs[::-1]
        
        # forward propagation
        features = lr.new_zeros((N, self.mid_channel, H, W)) # initialize features as zeros with same dtype and device as x
        for t in range(T):
            lr_curr = lr[:, t, :, :, :]
            if t != 0: # warp features if not looking at first frame of clip 
                features = self.flow_warp(features, forward_flows[:, t-1, :, :, :]) # feature warping (W step)
            features = torch.cat((features, lr_curr), dim=1)
            features = self.forward_res(features) # residual processing (R step)
        
            # fuse forward and backward features, then upsample
            bidirectional_features = torch.cat((features, outputs[t]), dim=1)
            fused = self.lrelu(self.fusion(bidirectional_features))
            out = F.pixel_shuffle(self.lrelu(self.up_conv1(fused)), 2)
            out = F.pixel_shuffle(self.lrelu(self.up_conv2(out)), 2)
            out = self.hr_conv(out)
            out += F.interpolate(lr_curr, scale_factor=4.0, mode='bilinear', align_corners=False) # add residuals to upsampled input
            outputs[t] = out
            
        return torch.stack(outputs, dim=1) # turn list of NCHW tensors with length T into a NTCHW tensor

