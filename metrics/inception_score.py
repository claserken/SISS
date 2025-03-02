# Adapted from https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/image/inception.py#L34-L218
import torch
from torch import Tensor
from typing import Any, List, Optional, Sequence, Tuple, Union
from metrics.classifier import Classifier

def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)

class InceptionScore:
    def __init__(self, classifier: Classifier, splits: int = 10, remove_class=None):
        self.splits = splits if remove_class is None else splits - 1
        self.remove_class = remove_class
        self.classifier = classifier
        self.logits = []
        
    def update(self, imgs: Tensor):
        """Update the state with extracted features."""
        logits = self.classifier.compute_logits(imgs)

        if self.remove_class is not None:
            mask = logits.argmax(-1) != self.remove_class
            logits = logits[mask]

            mask = torch.arange(logits.size(-1)) != self.remove_class
            logits = logits[:, mask]

        self.logits.append(logits)
    
    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric on logits of shape [batch_size, num_classes]."""

        # combine all updates
        logits = dim_zero_cat(self.logits)

        # random permute the features
        idx = torch.randperm(logits.shape[0])
        logits = logits[idx]

        # calculate probs and logits
        prob = logits.softmax(dim=1)
        log_prob = logits.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [
            p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)
        ]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()
