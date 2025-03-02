import torch
from torch import Tensor
from typing import Any, List, Optional, Sequence, Tuple, Union
import hydra 
# from metrics.mnist_resnet import resnet18
# from metrics.cifar_resnet import resnet56

class Classifier:
    def __init__(self, classifier, classifier_ckpt: str, classifier_args: dict, transform, device):
        self.classifier = classifier(**classifier_args).to(device)
        if classifier_ckpt is not None:
            params = torch.load(classifier_ckpt)
            self.classifier.load_state_dict(params)
        self.classifier.eval()
        self.transform = transform
        
    def compute_logits(self, imgs: Tensor, batch_size: int = 2048) -> Tensor:
        """
        Input: imgs is shape [N, C, H, W] with range [0, 1] 
        Output: logits of shape [N, num_classes=10]
        """
        num_imgs = imgs.size(0)
        num_batches = (num_imgs + batch_size - 1) // batch_size
        logits_batched = []

        if self.transform is not None:
            imgs = self.transform(imgs)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_imgs)
                imgs_batch = imgs[start_idx:end_idx]
                logits_batch = self.classifier(imgs_batch)
                logits_batched.append(logits_batch)

        logits = torch.cat(logits_batched, dim=0)
        return logits

    def compute_class_frequency(self, imgs: Tensor, img_class: int) -> float:
        """
        Input: imgs is shape [N, C, H, W] with range [0, 1] and digit is a number from 0-9
        Output: Fraction of imgs which are of specified digit
        """
        if self.transform is not None:
            imgs = self.transform(imgs)
    
        with torch.no_grad():
            logits = self.classifier(imgs)

        preds = logits.argmax(-1)

        count = (preds == img_class).sum().item()
        fraction = count / imgs.size(0)
        return fraction
