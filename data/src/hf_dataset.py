from datasets import load_dataset
from torch.utils.data import Dataset

class HFDataset(Dataset):
    # filter can be 'all', 'deletion', or 'nondeletion'
    def __init__(self, filter: str, name: str, split: str, image_key: str, class_to_remove=None, transform=None):
        self.transform = transform
        self.image_key = image_key
        self.data = load_dataset(name, split=split)

        match filter:
            case "all":
                pass
            case "deletion":
                if class_to_remove is None:
                    raise ValueError('Deletion filter requires removal class to be specified.')
                self.data = self.data.filter(lambda x: x["label"] == class_to_remove)
            case "nondeletion":
                if class_to_remove is None:
                    raise ValueError('Nondeletion filter requires removal class to be specified.')
                self.data = self.data.filter(lambda x: x["label"] != class_to_remove)
            case _:
                raise ValueError('Invalid filter.') 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[int(idx)][self.image_key]
        if self.transform:
            image = self.transform(image)
        return image