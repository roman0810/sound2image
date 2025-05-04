class ModelConfig:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        return self.config[name]