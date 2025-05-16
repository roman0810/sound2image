from utils.config import ModelConfig
from models.unet import UNetWithCrossAttention
from models.diffusion import Diffusion
from utils.SoundDataset import SoundDataset

import os
import numpy as np
import torch

from torch.utils.data import DataLoader
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
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = config.save_every
        self.epochs_run = 0
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
        print(f'Resuming training from snapshot at epoch {self.epochs_run}')

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source.float())
        loss = torch.nn.BCELoss()(output,targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f'GPU:{self.global_rank} | Epoch {epoch} | BS {b_sz} | Steps {len(self.train_data)}')
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f'Epoch {epoch} | Training snapshot saved at snapshot.pt')

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs(config):
    # инициализация датасета
    train_set = SoundDataset(config.image_path, config.sound_path)

    # инициализация модели и ее оптимизатора
    model = UNetWithCrossAttention(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last = True
    )

def main(save_every: int, total_epochs: int, snapshot_path: str = "snapshot.pt"):
    # запускаем движок распределенного обучения
    ddp_setup()

    # задаем параметры инициализации
    config = ModelConfig({"image_size": 128,
                          "audio_ctx_dim": 32,
                          "image_path": "data/images",
                          "sound_path": "data/sounds",
                          "lr": 0.0005,
                          "BS": 256,
                          "timesteps": 1000,
                          "save_every": save_every,
                          "snapshot_path": snapshot_path})

    # инициализируем датасет, модель и оптимизатор
    dataset, model, optimizer = load_train_objs(config)

    train_data = prepare_dataloader(dataset, batch_size=config.BS)
    trainer = Trainer(model, train_data, optimizer, config)

    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-epochs", type=int, help="Общее количество эпох (int)")
    parser.add_argument("--save-every", type=int, help="Интервал сохранения модели (int)")

    args = parser.parse_args()

    main(args.save_every, args.total_epochs)
