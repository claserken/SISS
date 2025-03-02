import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os 
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import json

class ShapesDataset(Dataset):
    def __init__(
        self, 
        num_samples_per_config=5000, 
        configs: list[str] = ["000", "100", "010", "001"],
        transform = None
    ):
        self.num_samples_per_config = num_samples_per_config
        self.transform = transform
        self.configs = configs
        self.available_configs = [
            shape + color + size 
            for shape in ["0", "1", "2"] 
            for color in ["0", "1", "2"] 
            for size in ["0", "1"]
        ]
        for config in configs:
            assert config in self.available_configs, f"{config} is not in available_configs: {self.available_configs}"

        self.image_path_dict = dict()
        self.image_paths = []
        for config in configs:
            path_pattern = f"input/single-body_2d_3classes/train/CLEVR_{config}_*.png"
            paths = glob.glob(path_pattern)
            self.image_path_dict[config] = paths
            self.image_paths.extend(paths)
        self.randomized_index = list(range(len(self.image_paths)))
        random.shuffle(self.randomized_index)

    def __getitem__(self, index):
        img_path = self.image_paths[self.randomized_index[index]]
            
        img = Image.open(img_path) #.convert('RGB')
        if self.transform is not None:
           img = self.transform(img)

        return img
        
    #     name_labels = img_path.split("_")[-2]
        
    #     with open(img_path.replace(".png", ".json"), 'r') as f:
    #        my_dict = json.loads(f.read())
    #        _size = my_dict[0]
    #        _color = my_dict[1][:3]
        
    #     size, color = _size, _color
    #    # # Define colors mapping
    #    # colors_map = {
    #    #     '0': [0.9, 0.1, 0.1],
    #    #     '1': [0.1, 0.1, 0.9],
    #    #     '2': [0.1, 0.9, 0.1]
    #    # }
    #    # # Assign size and color based on label values
    #    # size = 2.6 if int(name_labels[2]) == 0 else self.test_size
    #    # color = colors_map[name_labels[1]]
        
    #     # Convert size and color to numpy arrays
    #     size = np.array(size, dtype=np.float32)
    #     color = np.array(color, dtype=np.float32)
        
    #     # Create the label dictionary
    #     label = {0: int(name_labels[0]), 1: color, 2: size}
        
    #     return {"image": img, "label": label}

    def set_transform(self, transform):
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)