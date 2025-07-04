import torch
import torch.nn as nn
import math
from typing import Union, List
import torch.nn.functional as F

class Diffusion(nn.Module):
    def __init__(self,
                 timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 image_size: int = 256,
                 device: Union[str, torch.device] = "cuda"):
        super().__init__()
        self.timesteps = timesteps
        self.image_size = image_size
        self.device = device

        # Определение расписания beta
        if beta_schedule == "linear":
            self.betas = self.linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Расчет производных параметров
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Расчет параметров для q_posterior
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.intermediate_outputs = [None, None]
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, device=self.device)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Косинусное расписание как в Improved DDPM
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Прямой процесс диффузии (добавление шума)
        
        Args:
            x0: Исходные изображения [batch, 3, H, W]
            t: Временные шаги для каждого примера [batch]
        Returns:
            Зашумленные изображения и добавленный шум
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noisy_images = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
        return noisy_images, noise

    def reverse_process(
        self,
        model: nn.Module,
        audio_embeds: torch.Tensor,
        batch_size: int = 1,
        img_size: int = None,
        timesteps: int = None,
        use_ddim: bool = False,
        eta: float = 0.0,
        guidance_scale: float = 7.5,  # Коэффициент guidance (обычно 5-10)
        unconditional_prob: float = 0.1,  # Вероятность безусловного режима при обучении
    ) -> torch.Tensor:
        """
        Обратный процесс диффузии с Classifier-Free Guidance.

        Args:
            model: U-Net модель с поддержкой unconditional_embed=None
            audio_embeds: Аудио эмбеддинги [batch, seq_len, d_audio]
            guidance_scale: Сила влияния условия (>=1). 1 = нет guidance
            unconditional_prob: Вероятность подачи None как условия при обучении
        """
        img_size = img_size or self.image_size
        timesteps = timesteps or self.timesteps

        # Начальный шум
        x = torch.randn((batch_size, 3, img_size, img_size), device=self.device)

        # Определение шагов для sampling
        sequence = list(reversed(range(0, self.timesteps, self.timesteps // timesteps)))

        for i in sequence:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Два предсказания: условное и безусловное
            with torch.no_grad():
                # Условное предсказание (с аудио)
                t_embed = self.get_time_embedding(t).to(self.device)
                pred_noise_cond = model(x, t_embed, audio_embeds)
                # pred_noise = pred_noise_cond

                # Безусловное предсказание (None вместо audio_embeds)
                pred_noise_uncond = model(x, t_embed, None)

                # Комбинирование через CFG
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            # Обновление x (DDIM или DDPM)
            if use_ddim:
                x = self.ddim_step(x, pred_noise, t, i, eta)
            else:
                x = self.ddpm_step(x, pred_noise, t, i)

        return x.clamp(0, 1)

    def ddpm_step(self, x: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, i: int) -> torch.Tensor:
        """Один шаг денойзинга по DDPM."""
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        # Уравнение 11 из DDPM статьи
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        ) + torch.sqrt(beta_t) * noise
        
        return x

    def ddim_step(self, x: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, i: int, eta: float) -> torch.Tensor:
        """Один шаг денойзинга по DDIM."""
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Направление к x_t
        direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * pred_noise
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction + sigma_t * noise
        return x.clamp(-1.0, 1.0)

    def loss_fn(self, model: nn.Module, x0: torch.Tensor, audio_embeds: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Расчет loss для обучения (MSE между предсказанным и реальным шумом)
        
        Args:
            model: U-Net модель
            x0: Исходные изображения [batch, 3, H, W]
            audio_embeds: Аудио эмбеддинги [batch, seq_len, d_audio]
            t: Временные шаги (если None - выбираются случайно)
        Returns:
            Значение loss
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device).long()
        
        # Добавление шума
        noisy_images, noise = self.forward_process(x0, t)
        t_embed = self.get_time_embedding(t).to(self.device)
        
        # Предсказание шума моделью
        pred_noise = model(noisy_images, t_embed, audio_embeds)
        
        # MSE между реальным и предсказанным шумом
        return F.mse_loss(pred_noise, noise)

    def self_perceptual_loss(self, model: nn.Module, x0: torch.Tensor, audio_embeds: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Расчет комбинированных потерь из MSE между настоящим и сгенерированным шумом и MSE между эмбедингами на шаге t и t+1

        Args:
            model: U-Net модель
            x0: Исходные изображения [batch, 3, H, W]
            audio_embeds: Аудио эмбеддинги [batch, seq_len, d_audio]
            t: Временные шаги (если None - выбираются случайно)
        Returns:
            noise_loss
            feature_loss
        """
        if t is None:
            t = torch.randint(0, self.timesteps-1, (x0.shape[0],), device=self.device).long()

        noisy_images, noise = self.forward_process(x0, t)
        noisy_next, _ = self.forward_process(x0, t + 1)

        t_embed = self.get_time_embedding(t).to(self.device)
        t_embed_next = self.get_time_embedding(t + 1).to(self.device)

        model.eval()
        if type(model) == torch.nn.parallel.distributed.DistributedDataParallel:
            model.module.unet.bottleneck[0].register_forward_hook(self.get_activation(0))
            model.module.unet.bottleneck[2].register_forward_hook(self.get_activation(1))
        else:
            model.unet.bottleneck[0].register_forward_hook(self.get_activation(0))
            model.unet.bottleneck[2].register_forward_hook(self.get_activation(1))

        with torch.no_grad():
            _ = model(noisy_next, t_embed_next, audio_embeds)

            target_features = self.intermediate_outputs.copy()

        model.train()
        pred_noise = model(noisy_images, t_embed, audio_embeds)

        loss = None
        for output, target in zip(self.intermediate_outputs, target_features):
            if loss is None:
                loss = self.cos_sim(output, target).mean()
            else:
                loss += self.cos_sim(output, target).mean()

        return F.l1_loss(pred_noise, noise), loss



    def get_activation(self, index):
        def hook(model, input, output):
            self.intermediate_outputs[index] = output
        return hook



    def get_time_embedding(self, timestep, dtype=torch.float):
        assert len(timestep.shape) == 1, "timestep должен быть одномерным тензором размером [batch_size]"

        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160).to(self.device)
        x = timestep[:, None].to(dtype).to(self.device) * freqs[None, :]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
