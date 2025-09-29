from utils.config import ModelConfig

warmup_config = ModelConfig({
	"on_epo_scheduler": True,
	"start_factor": 0.01,
	"end_factor": 1.0,
	# "epoches": 1,
	"loss_type": "default"
	})

pretrain_config = ModelConfig({
	"on_epo_scheduler": False,
	"start_factor": None,
	"end_factor": None,
	# "epoches": 2,
	"loss_type": "default"
	})

tune_config = ModelConfig({
	"on_epo_scheduler": True,
	"start_factor": 1.0,
	"end_factor": 0.01,
	# "epoches": 1,
	"loss_type": "perceptual"
	})