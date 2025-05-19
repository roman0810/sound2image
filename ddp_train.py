from utils.config import ModelConfig
from models.unet import UNetWithCrossAttention
from models.diffusion import Diffusion
from utils.SoundDataset import SoundDataset

import os
import numpy as np
import torch
from archisound import ArchiSound
import time

from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    init_process_group(backend='nccl',
                       init_method='env://')

# Внедрить сюда обучение через прослойку класса Diffusion
# work in progres...
class Trainer:
    def __init__(self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: ModelConfig
    ) -> None:

        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ["RANK"])
        print(f"-->> glob rank: {self.global_rank}; local rank: {self.local_rank}")

        self.diffusion = Diffusion(timesteps=config.timesteps, 
                                    image_size=config.image_size,
                                    device=torch.device(f'cuda:{self.local_rank}'))

        self.autoencoder = ArchiSound.from_pretrained("dmae1d-ATC64-v2").to(torch.device(f'cuda:{self.local_rank}'))

        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.unconditional_prob = config.unconditional_prob
        self.save_every = config.save_every

        self.num_val_batches = config.num_val_batches
        self.epochs_run = 0
        self.train_losses = []
        self.val_losses = []

        if os.path.exists(config.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(config.snapshot_path)
        else:
            print(f'Snapshot path {config.snapshot_path} does not exist')

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_losses = snapshot["TRAIN_LOSSES"]
        self.val_losses = snapshot["VAL_LOSSES"]
        print(f'Resuming training from snapshot at epoch {self.epochs_run}')

    def _train_batch(self, source, targets):
        self.model.train()

        if torch.rand(1) < self.unconditional_prob:
            audio_embeds = None
        else:
            with torch.no_grad():
                audio_embeds = self.autoencoder.encode(source)
                
            audio_embeds = audio_embeds.permute(0, 2, 1)  # [B, seq_len, d_audio]

        self.optimizer.zero_grad()
        loss = self.diffusion.loss_fn(self.model, targets, audio_embeds)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate_batch(self, source, targets):
        self.model.eval()
        with torch.no_grad():
            audio_embeds = self.autoencoder.encode(source)
            audio_embeds = audio_embeds.permute(0, 2, 1)

            loss = self.diffusion.loss_fn(self.model, targets, audio_embeds)
        return loss.item()

    def _run_epoch(self, epoch):
        start_time = time.time()
        data_size = len(self.train_data)

        train_epo_losses = []
        val_epo_losses = []

        # DataLoder не перемешивает данные, поэтому используем его 
        # единственный объект для обучения и валидации
        for i , (source, targets) in enumerate(self.train_data):
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)

            if i < data_size - self.num_val_batches:
                train_epo_losses.append(self._train_batch(source, targets))
            else:
                val_epo_losses.append(self._validate_batch(source, targets))

        self.train_losses.append(sum(train_epo_losses)/len(train_epo_losses))
        self.val_losses.append(sum(val_epo_losses)/len(val_epo_losses))

        print(f'GPU:{self.global_rank} | Epoch {epoch} | Time {int(time.time()-start_time)}')

    def _save_snapshot(self, epoch, name):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["TRAIN_LOSSES"] = self.train_losses
        snapshot["VAL_LOSSES"] = self.val_losses

        torch.save(snapshot, f"{name}.pt")
        print(f'Epoch {epoch} | Training snapshot saved at {name}.pt')

    def train(self, max_epochs: int):
        b_sz = len(next(iter(self.train_data))[0])
        data_size = len(self.train_data)

        print(f"Training: Epoches {max_epochs-self.epochs_run} | BS {b_sz} | Batches {data_size}")
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, "snapshot")

        self._save_snapshot(epoch, "result")

def load_train_objs(config):
    # инициализация датасета
    train_set = SoundDataset(config.image_path, config.sound_path)

    # инициализация модели и ее оптимизатора
    model = UNetWithCrossAttention(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int=8):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last = True
    )


# испытанно, что максимум одного узла:
# BS = 128
# В теории после рездеоения нагрузки на N улов эти числа должны поделенны на N,
# однако локальный DataLoader со слишком большим BS может забить ram узла

# можно хорошо сэкономить работая лишь с одним DataLoader для обучения и валидации

def main(save_every: int, total_epochs: int, snapshot_path: str = "snapshot.pt"):
    # запускаем движок распределенного обучения
    ddp_setup()

    # задаем параметры инициализации
    config = ModelConfig({"image_size": 128,
                          "audio_ctx_dim": 32,
                          "image_path": "data/images",
                          "sound_path": "data/sounds",
                          "lr": 0.0005,
                          "BS": 128,
                          "unconditional_prob": 0.08,
                          "timesteps": 1000,
                          "num_val_batches": 10000//128,
                          "save_every": save_every,
                          "snapshot_path": snapshot_path})

    # инициализируем датасет, модель и оптимизатор
    dataset, model, optimizer = load_train_objs(config)

    train_loader = prepare_dataloader(dataset, batch_size=config.BS)

    trainer = Trainer(model, train_loader, optimizer, config)

    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-epochs", type=int, help="Общее количество эпох (int)")
    parser.add_argument("--save-every", type=int, help="Интервал сохранения модели (int)")

    args = parser.parse_args()

    main(args.save_every, args.total_epochs)
