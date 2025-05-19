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
            DownBlock(3, 32, 0.3),
            DownBlock(32, 64, 0.3),
            DownBlock(64, 128, 0.3),
            DownBlock(128, 256, 0.3)
        ])

        # Middle блок с Cross-Attention
        # self.mid_block_top = MidBlock(64, self.audio_ctx_dim, 0.2)
        self.mid_block_half = MidBlock(128, self.audio_ctx_dim, 0.2)
        self.mid_block_bot = MidBlock(256, self.audio_ctx_dim, 0.2)

        # Upsample блоки
        self.up_blocks = nn.ModuleList([
            UpBlock(256, 128, 0.3),
            UpBlock(128, 64, 0.3),
            UpBlock(64, 32, 0.3),
            UpBlock(32, 3, 0.3)
        ])
        # # Downsample блоки
        # self.down_blocks = nn.ModuleList([
        #     DownBlock(3, 16),
        #     DownBlock(16, 32),
        #     DownBlock(32, 64)
        # ])
        #
        # # Middle блок с Cross-Attention
        # self.mid_block = MidBlock(64, self.audio_ctx_dim)
        #
        # # Upsample блоки
        # self.up_blocks = nn.ModuleList([
        #     UpBlock(64, 32),
        #     UpBlock(32, 16),
        #     UpBlock(16, 3)
        # ])
        
        # Нормализация и активация
        # self.norm = nn.GroupNorm(1, 3)
        self.act = nn.SiLU()
        
    def forward(self, x, t, audio_embed=None):
        """
        Args:
            x: Тензор изображения [batch, 3, h, w]
            t: Тензор временных шагов [batch]
            audio_embed: Аудио-эмбеддинги [batch, seq_len, d_audio]
        Returns:
            Тензор шума [batch, 3, h, w]
        """
        if audio_embed is None:
            # Используйте нулевые эмбеддинги или пропустите Cross-Attention
            audio_embed = torch.zeros(x.shape[0], 1, self.audio_ctx_dim).to(x.device)

        # Downsample path
        skips = []
        for block in self.down_blocks:
            x = block(x, t)
            skips.append(x)
        
        # Middle block
        x = self.mid_block_bot(x, t, audio_embed)
        # Готовим дополнительную CrossAttention обработку для последнего моста
        skips[-2] = self.mid_block_half(skips[-2], t, audio_embed)

        
        # Upsample path
        for block in self.up_blocks:
            x = block(x, skips.pop(), t)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        self.drop1 = nn.Dropout2d(p_drop)
        
    def forward(self, x, t):
        h = F.silu(self.conv1(x))
        h = h + self.time_embed(t[:, None])[:, :, None, None]
        h = self.conv2(h)
        h = self.BN1(h)
        h = F.silu(h)
        h = self.downsample(h)
        h = self.drop1(h)
        return h


class MidBlock(nn.Module):
    """Средний блок с Cross-Attention"""
    def __init__(self, dim, audio_ctx_dim, p_drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(dim)

        # Cross-Attention
        self.attn = CrossAttentionBlock(dim, audio_ctx_dim, dim)
        
        self.BN2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.drop1 = nn.Dropout2d(p_drop)
        
    def forward(self, x, t, audio_embed):
        B, C, H, W = x.shape
        h = self.conv1(x)
        h = self.BN1(h)
        h = F.silu(h)
        
        # Применяем Cross-Attention
        h_flat = h.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        h_attn = self.attn(h_flat, audio_embed)      # [B, H*W, C]
        h = h_attn.permute(0, 2, 1).view(B, C, H, W)
        
        h = self.BN2(h)
        h = F.silu(self.conv2(h))
        h = self.drop1(h)
        #Зачем тут изначально стояло x + h????
        return h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch*2, in_ch*2, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_ch*2, out_ch, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        self.drop1 = nn.Dropout2d(p_drop)
        
    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        x = self.upsample(x, output_size=(-1, x.shape[-3], x.shape[-2]*2, x.shape[-1]*2))
        x = F.silu(x)
        h = self.conv1(x)
        h = self.BN1(h)
        h = F.silu(h)
        h = h + self.time_embed(t[:, None])[:, :, None, None]
        h = F.silu(self.conv2(h))
        h = self.drop1(h)
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
