from torch.utils.data import Dataset
import glob
from torchvision.io import read_image
import torch
import h5py


class EmbedsDataset(Dataset):
    def __init__(self, image_path, embed_path):
        self.file = h5py.File(embed_path, 'r')
        self.image_path = image_path

    def __len__(self):
        return len(self.file["data"])

    def __getitem__(self , index):
        raw = self.file["data"][index]

        name = raw["metadata"].decode('UTF-8')
        embeds = raw["tensor"]

        image = read_image(f"{self.image_path}/{name}.jpg")/255.0

        return torch.from_numpy(embeds), image

    def __del__(self):
        self.file.close()
