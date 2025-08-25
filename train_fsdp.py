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
from models.unet import UNetWithCrossAttention
from models.diffusion import Diffusion
from utils.EmbedsDataset import EmbedsDataset


class FSDP_Trainer:
	def __init__(self,
		model: torch.nn.Module,
		train_data: EmbedsDataset,
		val_data: EmbedsDataset,
		config: ModelConfig):

	    local_rank = int(os.environ["LOCAL_RANK"])
	    self.rank = int(os.environ["RANK"])
	    world_size = int(os.environ["WORLD_SIZE"])

	    init_process_group(backend="nccl")

	    torch.cuda.set_device(self.rank)

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
	    self.model = model.to(self.rank)

        if os.path.exists(config.snapshot_path):
            LR = self._load_snapshot(config.snapshot_path)
        else:
            print(f'GPU[{self.rank}]: Snapshot path {config.snapshot_path} does not exist')
            LR = config.lr

	    self.model = FSDP(
	        self.model,
	        auto_wrap_policy=fsdp_auto_wrap_policy,
	        mixed_precision=torch.distributed.fsdp.MixedPrecision(
	            param_dtype=torch.bfloat16,
	            reduce_dtype=torch.bfloat16,
	            buffer_dtype=torch.bfloat16,
	        ),
	        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
	        device_id=rank,
	        use_orig_params=True,
	    )
        self.diffusion = Diffusion(
        	timesteps=config.timesteps, 
            image_size=config.image_size,
            device=torch.device(f'cuda:{self.rank}')
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
        print(f'GPU[{self.rank}]: Resuming training at epoch {self.epochs_run} | LR = {LR}')
        return LR


    def _run_epoch(self, epoch):
        start_time = time.time()
        data_size = len(self.train_data)

        train_epo_noise_losses = []
        train_epo_feature_losses = []
        val_epo_losses = []

        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.rank)
            targets = targets.to(self.rank)

            noise_loss, feature_loss = self._train_batch(source, targets)
            train_epo_noise_losses.append(noise_loss)
            train_epo_feature_losses.append(feature_loss)

        self.scheduler.step()

        self.model.eval()
        for source, targets in self.val_data:
            source = source.to(self.rank)
            targets = targets.to(self.rank)

            val_epo_losses.append(self._validate_batch(source, targets))

        self.train_noise_losses.append(sum(train_epo_noise_losses)/len(train_epo_noise_losses))
        self.train_feature_losses.append(sum(train_epo_feature_losses)/len(train_epo_feature_losses))
        self.val_losses.append(sum(val_epo_losses)/len(val_epo_losses))

        print(f'GPU[{self.rank}]: Epoch {epoch} | Time {int(time.time()-start_time)}')


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
        self.scaler.step(optimizer)
        self.scaler.update()

        return noise_loss.item(), feature_loss.item()*self.perceptual_scale

    def __del__(self):
    	destroy_process_group()