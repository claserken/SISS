import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import time
from metrics.classifier import Classifier
from data.src.celeb_dataset import CelebAHQ

class FIDEvaluator:
    def __init__(self, inception_batch_size: int, device, classifier: Classifier = None, remove_class: int = None, filter_fake=True):
        self.batch_size = inception_batch_size
        self.device = device
        self.remove_class = remove_class
        self.classifier = classifier
        self.filter_fake = filter_fake
        # Note: FID with normalize requires [0, 1] ranges
        self.fid_computer = FrechetInceptionDistance(normalize=True, reset_real_features=False).to(device)

    def load_cifar(self, limit=None):
        print("Loading CIFAR dataset as FID real examples...")
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        iter_count = 0
        for images, labels in tqdm(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            if self.remove_class is not None:
                images = images[labels != self.remove_class]
            self.fid_computer.update(images, real=True)
            iter_count += 1
            if iter_count == limit:
                break
        torch.cuda.empty_cache() # Not sure why but helps clear a lot of memory?
    
    def load_celeb(self):
        print('Loading CelebAHQ as FID real examples...')
        celeb_dataset = CelebAHQ(filter='all', data_path='data/examples/celeba_hq_256', transform=transforms.ToTensor())
        dataloader = DataLoader(celeb_dataset, batch_size=self.batch_size, shuffle=False)
        for images in tqdm(dataloader):
            images = images.to(self.device)
            self.fid_computer.update(images, real=True)
        torch.cuda.empty_cache() # Not sure why but helps clear a lot of memory?
    
    def add_fake_images(self, fake_imgs):
        if self.remove_class is not None and self.filter_fake:
            logits = self.classifier.compute_logits(fake_imgs)
            preds = logits.argmax(-1)
            mask = (preds != self.remove_class)
            fake_imgs = fake_imgs[mask]

        for i in range(0, len(fake_imgs), self.batch_size):
            batch_fake_imgs = fake_imgs[i:i+self.batch_size]
            self.fid_computer.update(batch_fake_imgs, real=False) # fid normalized requires [0, 1] ranges

    def compute(self, reset=True, verbose=False):
        start_time = time.time()
        fid_score = self.fid_computer.compute()
        end_time = time.time()
        total_time = end_time - start_time
        
        if verbose:
            print(f"FID score: {fid_score}")
            print(f"Time taken for computing FID score: {total_time}")

        if reset: 
            self.fid_computer.reset()
        return fid_score
