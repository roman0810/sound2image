from torch.utils.data import Dataset
import glob
from torchvision.io import read_image
import torchaudio
import torch

class SoundDataset(Dataset):
    def __init__(self , image_path, sound_path):
        names = []
        for path in glob.glob(f'{image_path}/*.jpg'):
            name = path.split('/')[-1][:-4]
            names.append(name)

        self.names = names
        self.im_path = image_path
        self.au_path = sound_path
        self.stanart_len = 441000

    def __len__(self):
        return len(self.names)

    def __getitem__(self , index):
        image = read_image(f"{self.im_path}/{self.names[index]}.jpg")/255.0

        audio_path = f"{self.au_path}/{self.names[index]}.wav"
        waveform, _ = torchaudio.load(audio_path)

        # необходим стерео звук, если он одноканальный то приводим к стерео
        if waveform.shape[0] == 1:
            stereo = torch.zeros((2, waveform.shape[1]), dtype=torch.float)
            stereo[0] = waveform[0]
            stereo[1] = waveform[0]
            waveform = stereo

        elif waveform.shape[0] != 2:
            raise ValueError(f"audio {self.names[index]} must be stereo or mono, but {waveform.shape[0]} channels were given")

        # все тензоры должны быть стандартного размера (только для обучения)
        if waveform.shape[1] < self.stanart_len:
            ext_waveform = torch.zeros((2, self.stanart_len), dtype=torch.float)
            ext_waveform[:, :waveform.shape[1]] = waveform
            waveform = ext_waveform
        elif waveform.shape[1] > self.stanart_len:
            waveform = waveform[:, :self.stanart_len]

        return waveform.float(), image.float()
