import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision.io import read_image
from torchvision import transforms
import numpy as np

class SDData(Dataset):
    # filter can be 'all', 'deletion', or 'nondeletion'
    def __init__(self, img_dir: str, labels_fpath: str, filter: str, transform=None):
        with open(labels_fpath, 'r') as f:
            labels = json.load(f)
        all_names = list(labels.keys())
        all_labels = torch.tensor(list(labels.values())) 
        
        match filter:
            case "all":
                idx_filter = torch.arange(all_labels.shape[0])
            case "deletion":
                idx_filter = torch.where(all_labels == 1)[0]
            case "nondeletion":
                idx_filter = torch.where(all_labels == 0)[0]
            case _:
                raise ValueError('Invalid filter.') 
            
        self.img_dir = img_dir
        self.img_names = [all_names[i] for i in idx_filter.tolist()]
        self.img_labels = all_labels[idx_filter]
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        img_label = self.img_labels[idx]
        img = read_image(self.img_dir + img_path).to(torch.float)

        if self.transform:
            img = self.transform(img)
        
        return img, img_label

# dataset = SDData(IMG_DIR, LABELS_PATH, False)
# infinite_sampler = InfiniteSampler(dataset)
# train_dataloader = DataLoader(dataset, batch_size=4, sampler=infinite_sampler, drop_last=True)

# it = iter(train_dataloader)
# for i in range(100):
#     print(i)
#     batch = next(it)
#     print(batch[1])
	