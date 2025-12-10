# model.py - Main Network Architecture (c) itrustal.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DeformableAttention3d(nn.Module):
    """Simple attention deployment with stability improvements"""
    def __init__(self, channels, num_heads=2, tumor_label=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.tumor_label = tumor_label

        # Temperature parameter for softmax
        self.temperature = 0.5

        # Predict attention weights directly
        self.attn_conv = nn.Conv3d(channels * 2, num_heads, 1, bias=False)
        self.value_conv = nn.Conv3d(channels, channels, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

        self.register_buffer('_validate_tumor_label_at_runtime', torch.tensor(True))
        self.register_buffer('tumor_bias', torch.zeros(1))

    def forward(self, enc, dec):
        B, C, D, H, W = dec.shape

        # Predict spatial attention weights
        attn_input = torch.cat([enc, dec], dim=1)
        attn_logits = self.attn_conv(attn_input)  # [B, num_heads, D, H, W]

        # Temperature scaling (soften attention)
        attn_logits = attn_logits / self.temperature

        # Clamp logits before softmax (CRITICAL - prevents exp overflow)
        attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)

        # Flatten spatial dims for softmax
        attn_weights = torch.softmax(attn_logits.view(B, self.num_heads, -1), dim=-1)
        attn_weights = attn_weights.view(B, self.num_heads, 1, D, H, W)

        # Attention dropout for regularization
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1, training=True)

        # Get values
        v = self.value_conv(enc).view(B, self.num_heads, self.head_dim, D, H, W)

        # Weighted sum
        out = (v * attn_weights).reshape(B, C, D, H, W)
        out = self.proj(out)
        out = self.norm(out)

        # Clamp output before residual addition
        if self.training:
            out = torch.clamp(out, min=-5.0, max=5.0)

        if self.tumor_label < C:
            out[:, self.tumor_label:self.tumor_label+1] += self.tumor_bias

        return dec + out


class Morphology3d(nn.Module):
    """Depthwise separable morphology"""
    def __init__(self, channels):
        super().__init__()
        self.erosion = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dilation = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.edge_detect = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

        with torch.no_grad():
            self.erosion.weight.fill_(-0.1)
            self.erosion.weight[:, :, 1, 1, 1] = 0.2
            self.dilation.weight.fill_(0.1)
            self.dilation.weight[:, :, 1, 1, 1] = 0.2

        nn.init.kaiming_normal_(self.edge_detect.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        eroded = torch.tanh(self.erosion(x))
        dilated = torch.tanh(self.dilation(x))
        edges = torch.sigmoid(self.edge_detect(x))
        return dilated - eroded + edges


class BoundaryRefinement(nn.Module):
    """GBR: Gated boundary refinement module"""
    def __init__(self, channels):
        super().__init__()

        reduced = max(16, channels // 4)

        self.b1 = nn.Sequential(
            nn.Conv3d(channels, reduced, 3, padding=1, dilation=1, bias=False),
            nn.InstanceNorm3d(reduced, affine=True),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv3d(channels, reduced, 3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm3d(reduced, affine=True),
            nn.ReLU(inplace=True)
        )

        self.morphology = Morphology3d(channels)

        # Fusion
        self.fusion = nn.Conv3d(channels + reduced * 2 + channels, channels // 2, 1, bias=False)
        self.attention = nn.Sequential(
            nn.Conv3d(channels // 2, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True)
        )

        self.register_buffer('gate', torch.tensor(0.1))

    def forward(self, x, boundary_strength=1.0):
        # Replaced 'inp' with 'x' because we removed the inner function
        b1_out = self.b1(x)
        b2_out = self.b2(x)
        morph_out = self.morphology(x)

        fused = torch.cat([x, b1_out, b2_out, morph_out], dim=1)
        fused = self.fusion(fused)

        att = self.attention(fused)
        refined = self.refine(x * att)

        # Clamp refined features to prevent extreme values
        if self.training:
            refined = torch.clamp(refined, min=-3.0, max=3.0)

        # Scale down gate contribution for stability
        gate_val = torch.sigmoid(self.gate) * boundary_strength * 0.5  # Added 0.5 scaling
        return x + refined * gate_val


class Encoder(nn.Module):
    def __init__(self, in_ch, base):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        # Encoding Setup (base*6 max instead of base*8)
        self.e1 = ResidualBlock(in_ch, base, dropout=0.1)
        self.e2 = ResidualBlock(base, base*2, dropout=0.15)
        self.e3 = ResidualBlock(base*2, base*4, dropout=0.2)
        self.e4 = ResidualBlock(base*4, base*6, dropout=0.25)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    """Channel dimensions for concatenation"""
    def __init__(self, base, num_classes, deep_supervision=True, tumor_label=2):
        super().__init__()

        # Attention modules
        self.att3 = DeformableAttention3d(base*4, num_heads=2, tumor_label=tumor_label)
        self.att2 = DeformableAttention3d(base*2, num_heads=2, tumor_label=tumor_label)
        self.att1 = DeformableAttention3d(base, num_heads=2, tumor_label=tumor_label)

        # Upsampling: base*6 → base*4, etc.
        self.up3 = nn.ConvTranspose3d(base*6, base*4, 2, stride=2, bias=False)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2, bias=False)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2, bias=False)

        # Residual blocks take concatenated channels (skip + attention)
        self.rb3 = ResidualBlock(base*8, base*4, dropout=0.2)   # base*8 input
        self.rb2 = ResidualBlock(base*4, base*2, dropout=0.15)  # base*4 input
        self.rb1 = ResidualBlock(base*2, base, dropout=0.1)     # base*2 input

        # Boundary refinement
        self.ref3 = BoundaryRefinement(base*4)
        self.ref2 = BoundaryRefinement(base*2)
        self.ref1 = BoundaryRefinement(base)

        # Output head with reduced dropout for stability
        self.head_dropout = nn.Dropout3d(p=0.2)  # Reduced from 0.3 for stability
        self.head = nn.Conv3d(base, num_classes, 1)

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux2 = nn.Conv3d(base*2, num_classes, 1)
            self.aux3 = nn.Conv3d(base*4, num_classes, 1)

    def forward(self, x4, x3, x2, x1, use_refinement=False, boundary_strength=1.0):
        # Stage 3: x4 (base*6) → up3 → y3 (base*4)
        y3 = self.up3(x4)
        y3 = F.interpolate(y3, size=x3.shape[2:], mode='trilinear', align_corners=False)
        att3 = self.att3(x3, y3)  # att3: base*4
        y3 = torch.cat([y3, att3], dim=1)  # y3: base*8
        y3 = self.rb3(y3)  # rb3: base*8 → base*4
        if use_refinement:
            y3 = self.ref3(y3, boundary_strength)

        # Stage 2: y3 (base*4) → up2 → y2 (base*2)
        y2 = self.up2(y3)
        y2 = F.interpolate(y2, size=x2.shape[2:], mode='trilinear', align_corners=False)
        att2 = self.att2(x2, y2)  # att2: base*2
        y2 = torch.cat([y2, att2], dim=1)  # y2: base*4
        y2 = self.rb2(y2)  # rb2: base*4 → base*2
        if use_refinement:
            y2 = self.ref2(y2, boundary_strength)

        # Stage 1: y2 (base*2) → up1 → y1 (base)
        y1 = self.up1(y2)
        y1 = F.interpolate(y1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        att1 = self.att1(x1, y1)  # att1: base
        y1 = torch.cat([y1, att1], dim=1)  # y1: base*2
        y1 = self.rb1(y1)  # rb1: base*2 → base
        
        if use_refinement:
            y1 = self.ref1(y1, boundary_strength)
        
        y1 = self.head_dropout(y1) if self.training else y1

        out = self.head(y1)

        if self.deep_supervision and self.training:
            aux2 = self.aux2(y2)
            aux3 = self.aux3(y3)
            return out, [aux2, aux3]
        return out


class BAUN3D(nn.Module):
    """Core Model Implementation"""
    def __init__(self, config):
        super().__init__()
        base = config.base_channels

        self.encoder = Encoder(config.in_channels, base)
        # Bottleneck to maintain base*6 channels
        self.bottleneck = ResidualBlock(base*6, base*6, dropout=0.25)
        self.decoder = Decoder(base, config.num_classes,
                             deep_supervision=config.deep_supervision,
                             tumor_label=config.tumor_label)

        self.use_refinement = getattr(config, 'use_boundary_refinement', False)
        self.boundary_strength = 1.0

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x4 = self.bottleneck(x4)
        return self.decoder(x4, x3, x2, x1, self.use_refinement, self.boundary_strength)


def build_model(config):
    model = BAUN3D(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built with {total_params:,} parameters")
    return model
