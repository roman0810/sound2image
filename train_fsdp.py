import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed import init_process_group, destroy_process_group
# import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import os
import argparse

from utils.config import ModelConfig
from models.unet import UNetWithCrossAttention, ResidualBlock, AttentionBlock
from models.diffusion import Diffusion
from utils.EmbedsDataset import EmbedsDataset
import time


class FSDP_Trainer:
    def __init__(self,
        model: torch.nn.Module,
        train_data: EmbedsDataset,
        val_data: EmbedsDataset,
        config: ModelConfig):

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # не поддерживается GPU на суперкомпе
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        init_process_group(backend="nccl")

        torch.cuda.set_device(self.local_rank)

        # --- data ---
        self.train_data = DataLoader(
            train_data,
            batch_size=config.BS,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
            drop_last = True
        )
        self.val_data = DataLoader(
            val_data,
            batch_size=config.BS,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(val_data, shuffle=False, drop_last=True),
            drop_last = True
        )

        # --- train tools ---
        self.unconditional_prob = config.unconditional_prob
        self.save_every = config.save_every

        self.epochs_run = 0
        self.train_noise_losses = []
        self.train_feature_losses = []
        self.val_losses = []

        self.perceptual_gamma = 0.985
        self.perceptual_scale = 0.2
        self.difference = 2.5

        # --- model ---
        self.model = model.to(self.local_rank)

        if os.path.exists(config.snapshot_path):
            LR = self._load_snapshot(config.snapshot_path)
        else:
            print(f'GPU[{self.local_rank}]: Snapshot path {config.snapshot_path} does not exist')
            LR = config.lr

        def fsdp_auto_wrap_policy(module, recurse, nonwrapped_numel):
            return isinstance(module, (ResidualBlock, AttentionBlock))

        self.model = FSDP(
            self.model,
            auto_wrap_policy=fsdp_auto_wrap_policy,
            mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            device_id=self.local_rank,
            use_orig_params=True,
        )
        self.diffusion = Diffusion(
            timesteps=config.timesteps,
            image_size=config.image_size,
            device=torch.device(f'cuda:{self.local_rank}')
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LR)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.gamma)
        self.scaler = GradScaler()
        if config.compile:
            self.model = torch.compile(self.model)

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_noise_losses = snapshot["TRAIN_NOISE_LOSSES"]
        self.train_feature_losses = snapshot["TRAIN_FEATURE_LOSSES"]
        self.val_losses = snapshot["VAL_LOSSES"]
        self.perceptual_scale = snapshot["SP_SCALE"]

        LR = snapshot["LR"]
        print(f'GPU[{self.local_rank}]: Resuming training at epoch {self.epochs_run} | LR = {LR}')
        return LR


    def _run_epoch(self, epoch):
        start_time = time.time()
        data_size = len(self.train_data)

        # потери обязаны быть синхронизированы между всеми GPU, храним их в тензорах

        train_epo_noise_losses = torch.tensor(0.0).to(self.local_rank)
        train_epo_feature_losses = torch.tensor(0.0).to(self.local_rank)
        val_epo_losses = torch.tensor(0.0).to(self.local_rank)

        train_samples_done = torch.tensor(0.0).to(self.local_rank)
        val_samples_done = torch.tensor(0.0).to(self.local_rank)

        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)

            noise_loss, feature_loss = self._train_batch(source, targets)
            train_epo_noise_losses += noise_loss
            train_epo_feature_losses += feature_loss

            train_samples_done += torch.tensor(1.0).to(self.local_rank)

        # синхронизируем тренировочные потери на всех GPU
        torch.distributed.all_reduce(train_epo_noise_losses, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_epo_feature_losses, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_samples_done, op=torch.distributed.ReduceOp.SUM)

        self.train_noise_losses.append((train_epo_noise_losses/train_samples_done).item())
        self.train_feature_losses.append((train_epo_feature_losses/train_samples_done).item())

        self.scheduler.step()

        self.model.eval()
        for source, targets in self.val_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)

            val_epo_losses += self._validate_batch(source, targets)
            val_samples_done += torch.tensor(1.0).to(self.local_rank)


        torch.distributed.all_reduce(val_epo_losses, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_samples_done, op=torch.distributed.ReduceOp.SUM)

        self.val_losses.append((val_epo_losses/val_samples_done).item())

        print(f'GPU[{self.local_rank}]: Epoch {epoch} | Time {int(time.time()-start_time)}')


    def _train_batch(self, source, targets):
        # unconditional_prob - вероятность безусловной генерации
        if torch.rand(1) < self.unconditional_prob:
            source = None
                
        self.optimizer.zero_grad()
        with autocast('cuda', dtype=torch.bfloat16):
            noise_loss, feature_loss = self.diffusion.self_perceptual_loss(self.model, targets, source)

        # балансировка noise и self_perceptual потерь
        if (self.perceptual_scale*self.difference*feature_loss > noise_loss).item():
            self.perceptual_scale = self.perceptual_scale * self.perceptual_gamma

        loss = self.perceptual_scale*feature_loss + noise_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return noise_loss, feature_loss*self.perceptual_scale

    # ВНИМАНИЕ! Валидация считается ТОЛЬКО по мини-батчу каждой конкретной GPU
    @torch.no_grad()
    def _validate_batch(self, source, targets):
        with autocast('cuda', dtype=torch.bfloat16):
            loss = self.diffusion.loss_fn(self.model, targets, source)

        return loss

    def _save_snapshot(self, epoch, name):
        # только процесс GPU:0 сохраняет модель
        if self.local_rank != 0:
            print(f"GPU[{self.local_rank}] Warning! The save procces was skipped")
            return

        # ждем завершения всех процессов
        torch.distributed.barrier()

        # загружаем на GPU:0 все веса
        # ВНИМАНИЕ! Модель может не влезть на 1 карту
        model_state_dict = FSDP.full_state_dict(self.model, rank0_only=True)

        snapshot = {}
        # ВНИМАНИЕ! Возможны конфликты префиксов из-за обертки FSDP
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["TRAIN_NOISE_LOSSES"] = self.train_noise_losses
        snapshot["TRAIN_FEATURE_LOSSES"] = self.train_feature_losses
        snapshot["VAL_LOSSES"] = self.val_losses
        snapshot["LR"] = self.scheduler.get_last_lr()[0]
        snapshot["SP_SCALE"] = self.perceptual_scale

        torch.save(snapshot, f"{name}.pt")
        print(f'Epoch {epoch} | Training snapshot saved at {name}.pt')

    def train(self, max_epochs: int):
        b_sz = len(next(iter(self.train_data))[0])
        data_size = len(self.train_data)

        print(f"Training: Epoches {self.epochs_run}:{max_epochs} | BS {b_sz} | Batches {data_size}")
        for epoch in range(self.epochs_run, max_epochs):
            self.train_data.sampler.set_epoch(epoch)
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, "snapshot")

        self._save_snapshot(epoch, "result")


    def __del__(self):
        destroy_process_group()


# ВНИМАНИЕ! snapshot_path - это адрес загружаемой модели. Обученная модель будет сохранена как result.pt
# на текущей версии доля безусловной генерации фиксированна
def main(save_every: int, total_epochs: int, snapshot_path: str = "snapshot.pt"):
    # задаем параметры инициализации
    config = ModelConfig({"image_size": 128,
                          "sample_rate": 48000,
                          "audio_ctx_dim": 768,
                          "image_path": "data/images",
                          "embed_path": "data/embeds/sound_embeds.h5",
                          "lr": 0.0005,
                          "gamma": 0.98,
                          "BS": 3,
                          "unconditional_prob": 0.1,
                          "timesteps": 1000,
                          "save_every": save_every,
                          "snapshot_path": snapshot_path,
                          "compile": True})

    # инициализируем датасет и модель
    dataset = EmbedsDataset(config.image_path, config.embed_path)
    train_datset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-8000, 8000])
    model = UNetWithCrossAttention(config)

    trainer = FSDP_Trainer(
        model,
        train_datset,
        val_dataset,
        config
        )

    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-epochs", type=int, help="Общее количество эпох (int)")
    parser.add_argument("--save-every", type=int, help="Интервал сохранения модели (int)")

    args = parser.parse_args()

    main(args.save_every, args.total_epochs)
