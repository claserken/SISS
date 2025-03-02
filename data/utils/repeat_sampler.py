import torch
from torch.utils.data import Sampler

class RepeatedSampler(Sampler):
    def __init__(self, data_source, num_repeats):
        """
        Arguments:
        - data_source (Dataset): The dataset to sample from.
        - num_repeats (int): The number of times each image should be repeated.
        """
        self.data_source = data_source
        self.num_repeats = num_repeats

    def __len__(self):
        # The length of the sampler is the number of samples multiplied by the number of repeats
        return len(self.data_source) * self.num_repeats

    def __iter__(self):
        # Repeat each index 'num_repeats' times in place without shuffling
        indices = torch.arange(len(self.data_source)).repeat_interleave(self.num_repeats)
        return iter(indices.tolist())


