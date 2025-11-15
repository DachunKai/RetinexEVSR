from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.sci_arch import SCI
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import *


@ARCH_REGISTRY.register()
class RetinexEVSR(nn.Module):
    """Seeing the Unseen: Zooming in the Dark with Event Cameras (AAAI 2026)
    
    This model performs 4x low-light video super-resolution using event cameras.
    
    Args:
        dim (int): Channel dimension for intermediate features. Default: 32.
        num_blocks (int): Number of residual blocks per propagation branch. Default: 5.
        spynet_path (str): Path to pretrained SPyNet weights. Default: None.
        sci_path (str): Path to pretrained SCI weights. Default: None.
    """

    def __init__(
        self,
        dim: int = 32,
        num_blocks: int = 5,
        spynet_path: str = None,
        sci_path: str = None
    ) -> None:
        super().__init__()
        self.dim = dim

        self.spynet = SpyNet(spynet_path)
        self.sci = SCI(sci_path)
        self.iee = GuideFeatExt(bc=4)
        self.r_embed = ConvResidualBlocks(3, dim, 5)

        # ================== Encoder ==================
        self.encoder_1 = PROPs(
            inp_dim=dim * 3,
            embed_dim=dim,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[True],  # backward
            has_event=False
        )
        self.patchmerge_1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        self.encoder_2 = PROPs(
            inp_dim=dim * 6,
            embed_dim=dim * 2,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[False],  # forward
            has_event=False
        )
        self.patchmerge_2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)

        # ================== BottleNeck ==================
        self.bottle_1 = PROPs(
            inp_dim=dim * 12,
            embed_dim=dim * 4,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[True],  # backward
            has_event=False
        )
        self.bottle_2 = PROPs(
            inp_dim=dim * 12,
            embed_dim=dim * 4,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[False],  # forward
            has_event=True,
            event_dim=64
        )

        # ================== Decoder ==================
        self.patchexpand_1 = nn.ConvTranspose2d(
            in_channels=dim * 4, out_channels=dim * 2, 
            kernel_size=2, stride=2, padding=0, output_padding=0
        )
        self.fusion_1 = nn.Conv2d(dim * 4, dim * 2, kernel_size=3, padding=1)
        self.decoder_1 = PROPs(
            inp_dim=dim * 6,
            embed_dim=dim * 2,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[True],  # backward
            has_event=True,
            event_dim=32
        )
        self.patchexpand_2 = nn.ConvTranspose2d(
            in_channels=dim * 2, out_channels=dim, 
            kernel_size=2, stride=2, padding=0, output_padding=0
        )
        self.fusion_2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.decoder_2 = PROPs(
            inp_dim=dim * 3,
            embed_dim=dim,
            num_resblocks=num_blocks,
            num_PROP=1,
            reverse=[False],  # forward
            has_event=True,
            event_dim=8
        )
        
        self.conv = Basic2d(in_channels=3 + 8, out_channels=dim, norm_layer=nn.BatchNorm2d)
        self.channel_attn = ChannelAttn(channels=dim)
        
        # ================== Reconstruction Head ==================
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 3, 3, 1, 1)
        )
        
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def compute_flow(self, lqs: torch.Tensor, flows: Dict) -> Dict:
        """Compute optical flow between consecutive frames.
        
        Args:
            lqs: Input sequence (n, t, c, h, w)
            flows: Dictionary to store flow results
            
        Returns:
            Updated flows dictionary with all flow scales
        """
        n, t, _, h, w = lqs.shape
        flows = {
            'forward': [],
            'backward': [],
            'forward_ds2': [],
            'backward_ds2': [],
            'forward_ds4': [],
            'backward_ds4': []
        }
        
        for i in range(t - 1):
            lq1, lq2 = lqs[:, i], lqs[:, i + 1]
            flow_backward = self.spynet(lq1, lq2)
            flow_forward = self.spynet(lq2, lq1)
            
            # Downsample flows for multi-scale processing
            flow_backward_ds2 = F.avg_pool2d(flow_backward, 2, 2) / 2.0
            flow_forward_ds2 = F.avg_pool2d(flow_forward, 2, 2) / 2.0
            flow_backward_ds4 = F.avg_pool2d(flow_backward_ds2, 2, 2) / 2.0
            flow_forward_ds4 = F.avg_pool2d(flow_forward_ds2, 2, 2) / 2.0
            
            flows['forward'].append(flow_forward)
            flows['backward'].append(flow_backward)
            flows['forward_ds2'].append(flow_forward_ds2)
            flows['backward_ds2'].append(flow_backward_ds2)
            flows['forward_ds4'].append(flow_forward_ds4)
            flows['backward_ds4'].append(flow_backward_ds4)
        
        return flows

    def compute_enhanced_imgs(self, lqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhance low-quality images using Retinex-based SCI.
        
        Args:
            lqs: Input sequence (n, t, c, h, w)
            
        Returns:
            (illumination, reflectance) sequences (both (n, t, c, h, w))
        """
        n, t, c, h, w = lqs.shape
        lqs_flat = lqs.view(n * t, c, h, w)
        
        illuminations = []
        reflectances = []
        for img in lqs_flat:
            illumination, reflectance = self.sci(img.unsqueeze(0))
            illuminations.append(illumination)
            reflectances.append(reflectance)
        
        return (
            torch.stack(illuminations).view(n, t, c, h, w),
            torch.stack(reflectances).view(n, t, c, h, w)
        )

    def compute_event_features(self, imgs: torch.Tensor, voxels: torch.Tensor, events: Dict) -> Dict:
        """Compute event features from event voxels and images.
        
        Args:
            imgs: Enhanced images (reflectance) (n, t, c, h, w)
            voxels: Event voxels (n, t, bins, h, w)
            events: Dictionary to store event features
            
        Returns:
            Updated events dictionary with features at different resolutions
        """
        n, t, _, h, w = voxels.shape
        events = {
            'ds1': [],  # Full resolution: (n, 8, h, w)
            'ds2': [],  # 1/2 resolution: (n, 32, h//2, w//2)
            'ds4': []   # 1/4 resolution: (n, 64, h//4, w//4)
        }
        
        for i in range(t):
            voxel = voxels[:, i]  # (n, bins, h, w)
            img = imgs[:, i]      # (n, 3, h, w)
            c0_event, c4, c2, c1 = self.iee(img, voxel)
            events['ds1'].append(c1)
            events['ds2'].append(c2)
            events['ds4'].append(c4)
        
        return events

    def _upsample(self, 
                 feats_decoder2: List[torch.Tensor], 
                 events_ds1: List[torch.Tensor], 
                 i_lqs: torch.Tensor, 
                 lqs: torch.Tensor) -> torch.Tensor:
        """Apply illumination-aware refinement and upsampling.
        
        Args:
            feats_decoder2: Features from decoder stage 2 (t elements, each (n, c, h, w))
            events_ds1: Event features at full resolution (t elements, each (n, 8, h, w))
            i_lqs: Enhanced illumination images (n, t, c, h, w)
            lqs: Original low-quality inputs (n, t, c, h, w)
            
        Returns:
            Output high-resolution sequence (n, t, c, 4h, 4w)
        """
        n, t, c, h, w = lqs.shape
        
        # Apply illumination-guided refinement
        for i in range(t):
            # Concatenate event features and illumination
            event_illum = torch.cat([events_ds1[i], i_lqs[:, i]], dim=1)  # (n, 8+3, h, w)
            light_ill_feat = self.conv(event_illum)
            light_ill_feat = self.channel_attn(light_ill_feat)
            light_ill_feat = torch.sigmoid(light_ill_feat)
            feats_decoder2[i] = feats_decoder2[i] * light_ill_feat + feats_decoder2[i]
        
        # Apply final reconstruction head
        feats_decoder2 = Forward(feats_decoder2, self.tail)  # (n, c, 4h, 4w) per frame
        
        # Add residual connection
        outputs = []
        for i in range(t):
            residual = F.interpolate(lqs[:, i], scale_factor=4, mode='bilinear', align_corners=False)
            outputs.append(feats_decoder2[i] + residual)
        
        return torch.stack(outputs, dim=1)  # (n, t, c, 4h, 4w)

    def forward(self, lqs: torch.Tensor, voxels: torch.Tensor) -> torch.Tensor:
        """Forward pass for RetinexEVSR.
        
        Args:
            lqs: Input low-quality sequence (n, t, c, h, w)
            voxels: Input event voxels (n, t, bins, h, w)
            
        Returns:
            Output high-resolution sequence (n, t, c, 4h, 4w)
        """
        n, t, c, h, w = lqs.shape
        feats = {}
        
        # ================== Step 1: Retinex Decomposition ==================
        i_lqs, r_lqs = self.compute_enhanced_imgs(lqs)
        
        # ================== Step 2: Optical Flow Computation ==================
        flows = self.compute_flow(r_lqs, {})
        
        # ================== Step 3: Event Feature Extraction ==================
        events = self.compute_event_features(i_lqs, voxels, {})
        
        # ================== Step 4: Information Propagation ==================
        feats_ = self.r_embed(r_lqs.view(n * t, c, h, w)).view(n, t, self.dim, h, w)
        feats['encoder1'] = [feats_[:, i] for i in range(t)]  # (t, n, dim, h, w)
        
        # Encoder propagation
        feats['encoder1'] = self.encoder_1(feats['encoder1'], flows['backward'])
        feats['encoder2'] = Forward(feats['encoder1'], self.patchmerge_1)
        feats['encoder2'] = self.encoder_2(feats['encoder2'], flows['forward_ds2'])
        
        # BottleNeck propagation
        feats['bottle'] = Forward(feats['encoder2'], self.patchmerge_2)
        feats['bottle'] = self.bottle_1(feats['bottle'], flows['backward_ds4'])
        feats['bottle'] = self.bottle_2(feats['bottle'], flows['forward_ds4'], events['ds4'])
        
        # Decoder processing
        feats['decoder1'] = Forward(feats['bottle'], self.patchexpand_1)
        
        # Feature fusion
        for i in range(t):
            feat_encoder_2 = feats['encoder2'][i]
            feat_decoder_1 = feats['decoder1'][i]
            feats['decoder1'][i] = self.lrelu(self.fusion_1(
                torch.cat([feat_encoder_2, feat_decoder_1], dim=1)
            ))
        
        feats['decoder1'] = self.decoder_1(feats['decoder1'], flows['backward_ds2'], events['ds2'])
        feats['decoder2'] = Forward(feats['decoder1'], self.patchexpand_2)
        
        for i in range(t):
            feat_encoder_1 = feats['encoder1'][i]
            feat_decoder_2 = feats['decoder2'][i]
            feats['decoder2'][i] = self.lrelu(self.fusion_2(
                torch.cat([feat_encoder_1, feat_decoder_2], dim=1)
            ))
        
        feats['decoder2'] = self.decoder_2(feats['decoder2'], flows['forward'], events['ds1'])
        
        # ================== Step 5: Final Upsampling ==================
        return self._upsample(
            feats_decoder2=feats['decoder2'],
            events_ds1=events['ds1'],
            i_lqs=i_lqs,
            lqs=lqs
        )