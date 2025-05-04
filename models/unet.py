import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.audio_ctx_dim = config.audio_ctx_dim  # d_audio из энкодера
        
        # Downsample блоки
        self.down_blocks = nn.ModuleList([
            DownBlock(3, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512)
        ])
        
        # Middle блок с Cross-Attention
        self.mid_block = MidBlock(512, self.audio_ctx_dim)
        
        # Upsample блоки
        self.up_blocks = nn.ModuleList([
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 3)
        ])
        
        # Нормализация и активация
        self.norm = nn.GroupNorm(1, 3)
        self.act = nn.SiLU()
        
    def forward(self, x, t, audio_embed):
        """
        Args:
            x: Тензор изображения [batch, 3, h, w]
            t: Тензор временных шагов [batch]
            audio_embed: Аудио-эмбеддинги [batch, seq_len, d_audio]
        Returns:
            Тензор шума [batch, 3, h, w]
        """
        # Downsample path
        skips = []
        for block in self.down_blocks:
            x = block(x, t)
            skips.append(x)
        
        # Middle block
        x = self.mid_block(x, t, audio_embed)
        
        # Upsample path
        for block in self.up_blocks:
            x = block(x, skips.pop(), t)

        # Финальная нормализация
        return self.norm(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        
    def forward(self, x, t):
        h = F.silu(self.conv1(x))
        h = h + self.time_embed(t[:, None])[:, :, None, None]
        h = F.silu(self.conv2(h))
        return self.downsample(h)


class MidBlock(nn.Module):
    """Средний блок с Cross-Attention"""
    def __init__(self, dim, audio_ctx_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        
        # Cross-Attention
        self.attn = CrossAttentionBlock(dim, audio_ctx_dim, dim)
        
        self.norm2 = nn.GroupNorm(32, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        
    def forward(self, x, t, audio_embed):
        B, C, H, W = x.shape
        h = self.norm1(x)
        h = F.silu(self.conv1(h))
        
        # Применяем Cross-Attention
        h_flat = h.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        h_attn = self.attn(h_flat, audio_embed)      # [B, H*W, C]
        h = h_attn.permute(0, 2, 1).view(B, C, H, W)
        
        h = self.norm2(h)
        h = F.silu(self.conv2(h))
        return x + h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_ch*2, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        
    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        x = self.upsample(x)
        h = F.silu(self.conv1(x))
        h = h + self.time_embed(t[:, None])[:, :, None, None]
        h = F.silu(self.conv2(h))
        return h


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_query, d_audio, d_out):
        """
        d_query - число каналов изображения
        d_audio - латентня размерность фрагмента аудио
        d_out   - число каналов в выходном изображении
        """
        super().__init__()
        self.W_Q = nn.Linear(d_query, d_out)
        self.W_K = nn.Linear(d_audio, d_out)
        self.W_V = nn.Linear(d_audio, d_out)
        
    def forward(self, x, audio_embed):
        """
        x - фичи изображения размера [batch, h,w, ch]
        audio_embed - латентные представления всех токенов аудио размера [batch, seq_len, d_audio]
        
        Returns:
            [batch, h*w, d_out] - обогащенные признаки
        """
        Q = self.W_Q(x)          # [B, h*w, d_out]
        K = self.W_K(audio_embed) # [B, seq_len, d_out]
        V = self.W_V(audio_embed) # [B, seq_len, d_out]
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1]**0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)