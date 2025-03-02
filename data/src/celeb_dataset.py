from torch.utils.data import Dataset
from PIL import Image
import os

class CelebAHQ(Dataset):
    def __init__(self, filter: str, data_path, remove_img_names=None, transform=None):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.transform = transform

        match filter:
            case "all":
                pass
            case "deletion":
                if remove_img_names is None:
                    raise ValueError('Deletion filter requires removal class to be specified.')
                self.image_files = remove_img_names
            case "nondeletion":
                if remove_img_names is None:
                    raise ValueError('Nondeletion filter requires removal class to be specified.')
                self.image_files = [f for f in os.listdir(data_path) if f not in remove_img_names]
            case _:
                raise ValueError('Invalid filter.') 
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_path, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image