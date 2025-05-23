import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config

class AudioEncoder:
    def __init__(self,
                model_name,
                device,
                sample_rate = 48000):
        self.sample_rate = sample_rate
        self.device = device

        config = Wav2Vec2Config.from_pretrained(model_name)
        config.apply_spec_augment = False

        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config).to(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def stereo_to_mono(self, audio_tensor):
        return torch.mean(audio_tensor, dim=1, keepdim=True)


    def preprocess_audio(self, audio_mono):
        if self.sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=16000
            )
            audio_mono = resampler(audio_mono.cpu())
        return audio_mono.to(self.device)


    def get_wav2vec2_features(self, audio_stereo_tensor):
        inputs = self.feature_extractor(
            audio_stereo_tensor.squeeze(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs['input_values'] = inputs['input_values'].squeeze().to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        return last_hidden_state

    def __call__(self, audio):
        audio = self.stereo_to_mono(audio)
        audio = self.preprocess_audio(audio)
        features = self.get_wav2vec2_features(audio)

        return features
