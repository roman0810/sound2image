import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768, self_att=True):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        if self_att:
            self.layernorm_1 = nn.LayerNorm(channels)
            self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
            self.self_att = True
        else:
            self.self_att = False

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        if self.self_att:
            residue_short = x
            x = self.layernorm_1(x)
            x = self.attention_1(x)
            x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPUs = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]

        self.encoder1 = nn.ModuleList([
            SwitchSequential(nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(32, 32), AttentionBlock(8, 4, self_att=False)),
            SwitchSequential(ResidualBlock(32, 32), AttentionBlock(8, 4, self_att=False)),
            SwitchSequential(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
            ]).to(self.GPUs[1])

        self.encoder2 = nn.ModuleList([
            SwitchSequential(ResidualBlock(32, 64), AttentionBlock(8, 8, self_att=True)),
            SwitchSequential(ResidualBlock(64, 64), AttentionBlock(8, 8, self_att=True)),
            SwitchSequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
            ]).to(self.GPUs[2])

        self.encoder3 = nn.ModuleList([
            SwitchSequential(ResidualBlock(64, 128), AttentionBlock(8, 16, self_att=True)),
            SwitchSequential(ResidualBlock(128, 128), AttentionBlock(8, 16, self_att=True)),
            SwitchSequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(128, 128)),
            SwitchSequential(ResidualBlock(128, 128)),
        ]).to(self.GPUs[3])

        self.bottleneck = SwitchSequential(
            ResidualBlock(128, 128),
            AttentionBlock(8, 16),
            ResidualBlock(128, 128),
        ).to(self.GPUs[4])

        self.decoder1 = nn.ModuleList([
            SwitchSequential(ResidualBlock(256, 128)),
            SwitchSequential(ResidualBlock(256, 128)),
            SwitchSequential(ResidualBlock(256, 128), Upsample(128)),
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16, self_att=True)),
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16, self_att=True))
            ]).to(self.GPUs[5])

        self.decoder2 = nn.ModuleList([
            SwitchSequential(ResidualBlock(192, 128), AttentionBlock(8, 16, self_att=True), Upsample(128)),
            SwitchSequential(ResidualBlock(192, 64), AttentionBlock(8, 8, self_att=True)),
            SwitchSequential(ResidualBlock(128, 64), AttentionBlock(8, 8, self_att=True))
            ]).to(self.GPUs[6])

        self.decoder3 = nn.ModuleList([
            SwitchSequential(ResidualBlock(96, 64), AttentionBlock(8, 8, self_att=True), Upsample(64)),
            SwitchSequential(ResidualBlock(96, 32), AttentionBlock(8, 4, self_att=False)),
            SwitchSequential(ResidualBlock(64, 32)),
            SwitchSequential(ResidualBlock(64, 32)),
        ]).to(self.GPUs[7])

    def forward(self, x, context, time):
        skip_connections = []
        # проводим операции каждого блока на своей карте и кладем результаты на карту где их будут принимать
        for layers in self.encoder1:
            x = layers(x.to(self.GPUs[1]), context.to(self.GPUs[1]), time.to(self.GPUs[1]))
            skip_connections.append(x.to(self.GPUs[7]))

        for layers in self.encoder2:
            x = layers(x.to(self.GPUs[2]), context.to(self.GPUs[2]), time.to(self.GPUs[2]))
            skip_connections.append(x.to(self.GPUs[6]))

        for layers in self.encoder3:
            x = layers(x.to(self.GPUs[3]), context.to(self.GPUs[3]), time.to(self.GPUs[3]))
            skip_connections.append(x.to(self.GPUs[5]))


        x = self.bottleneck(x.to(self.GPUs[4]), context.to(self.GPUs[4]), time.to(self.GPUs[4]))


        for layers in self.decoder1:
            x = torch.cat((x.to(self.GPUs[5]), skip_connections.pop()), dim=1)
            x = layers(x, context.to(self.GPUs[5]), time.to(self.GPUs[5]))

        for layers in self.decoder2:
            x = torch.cat((x.to(self.GPUs[6]), skip_connections.pop()), dim=1)
            x = layers(x, context.to(self.GPUs[6]), time.to(self.GPUs[6]))

        for layers in self.decoder3:
            x = torch.cat((x.to(self.GPUs[7]), skip_connections.pop()), dim=1)
            x = layers(x, context.to(self.GPUs[7]), time.to(self.GPUs[7]))

        return x


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class UNetWithCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_embedding = TimeEmbedding(320).to("cuda:0")
        self.unet = UNet()
        self.final = FinalLayer(32, 3).to("cuda:0")
        self.audio_ctx_dim = config.audio_ctx_dim

    def forward(self, x, time, context=None):
        if context is None:
            context = torch.zeros(x.shape[0], 1, self.audio_ctx_dim).to(x.device)

        time = self.time_embedding(time.to("cuda:0"))
        output = self.unet(x, context, time)
        output = self.final(output.to("cuda:0"))
        return output
