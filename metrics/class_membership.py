# Motivated by Carlini et al. LiRA and the "Goldilock's zone" in https://arxiv.org/pdf/2301.13188.pdf
import random
import torch
from tqdm.auto import tqdm
from typing import List

class MembershipLoss:
    def __init__(
        self,
        dataset_all,
        dataset_deletion,
        noise_scheduler,
        unet,
        num_image_samples,
        num_noise_samples,
        eval_batch_size,
        device
    ):
        self.dataset_all = dataset_all
        self.dataset_deletion = dataset_deletion
        # self.golden_timestep = golden_timestep
        # self.timesteps = timesteps
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.num_image_samples = num_image_samples
        self.num_noise_samples = num_noise_samples
        self.eval_batch_size = eval_batch_size
        self.device = device

    def sample_images(self):
        # Get the length of the dataset
        dataset_all_len = len(self.dataset_all)
        dataset_deletion_len = len(self.dataset_deletion)

        # Generate a list of random indices
        all_random_indices = random.sample(
            range(dataset_all_len), self.num_image_samples
        )

        if dataset_deletion_len == 1:
            deletion_random_indices = [0] * self.num_image_samples
        else:
            deletion_random_indices = random.sample(
                range(dataset_deletion_len), self.num_image_samples
            )

        # Initialize an empty list to store the sampled images
        all_sampled_images = []
        deletion_sampled_images = []

        # Iterate over the random indices and append the corresponding images
        for idx in all_random_indices:
            image = self.dataset_all[idx]  # Assuming dataset returns (image, label) pairs
            all_sampled_images.append(image)

        for idx in deletion_random_indices:
            image = self.dataset_deletion[idx]  # Assuming dataset returns (image, label) pairs
            deletion_sampled_images.append(image)

        self.all_sampled_images = torch.stack(all_sampled_images, dim=0).to(self.device)
        self.deletion_sampled_images = torch.stack(deletion_sampled_images, dim=0).to(self.device)

    def sample_noises(self):
        # Make sure to sample images first!
        noise = torch.randn((self.num_noise_samples, *self.all_sampled_images.shape[1:]), device=self.device)
        self.noise = noise

    def compute_membership_losses(self, timesteps: List[int]):
        """
        Computes membership loss on sampled all and deletion images at golden timestep and averages across sampled noises
        :return: Tuple of all membership loss and deletion membership loss
        """
        assert self.all_sampled_images.shape == self.deletion_sampled_images.shape
        membership_losses = []
        for timestep in timesteps:
            # Expand the sampled images to match the number of noise samples
            all_sampled_images_expanded = self.all_sampled_images.unsqueeze(1).expand(-1, self.num_noise_samples, -1, -1, -1)
            deletion_sampled_images_expanded = self.deletion_sampled_images.unsqueeze(1).expand(-1, self.num_noise_samples, -1, -1, -1)

            # Expand the noise to match the number of sampled images
            noise_expanded = self.noise.unsqueeze(0).expand(self.num_image_samples, -1, -1, -1, -1)

            # Flatten the expanded images and noise
            all_sampled_images_flat = all_sampled_images_expanded.reshape(-1, *all_sampled_images_expanded.shape[2:])
            deletion_sampled_images_flat = deletion_sampled_images_expanded.reshape(-1, *deletion_sampled_images_expanded.shape[2:])
            noise_flat = noise_expanded.reshape(-1, *noise_expanded.shape[2:])

            # Expand the golden timestep to match the flattened shape
            timestep_expanded = torch.full((all_sampled_images_flat.shape[0],), timestep, device=self.device)

            # Add noise to the flattened images
            all_noisy_images_flat = self.noise_scheduler.add_noise(all_sampled_images_flat, noise_flat, timestep_expanded)
            deletion_noisy_images_flat = self.noise_scheduler.add_noise(deletion_sampled_images_flat, noise_flat, timestep_expanded)

            # Perform batched forward pass through the UNet
            all_membership_losses = []
            deletion_membership_losses = []
            
            with torch.no_grad():
                for i in tqdm(range(0, all_noisy_images_flat.shape[0], self.eval_batch_size)):
                    batch_all_noisy_images = all_noisy_images_flat[i:i+self.eval_batch_size]
                    batch_deletion_noisy_images = deletion_noisy_images_flat[i:i+self.eval_batch_size]
                    batch_noise = noise_flat[i:i+self.eval_batch_size]
                    batch_timesteps = timestep_expanded[:batch_noise.shape[0]]

                    batch_all_model_output = self.unet(batch_all_noisy_images, batch_timesteps, return_dict=False)[0]
                    batch_deletion_model_output = self.unet(batch_deletion_noisy_images, batch_timesteps, return_dict=False)[0]

                    batch_all_loss = torch.sum((batch_all_model_output - batch_noise) ** 2, dim=[1, 2, 3])
                    batch_deletion_loss = torch.sum((batch_deletion_model_output - batch_noise) ** 2, dim=[1, 2, 3])
                
                    all_membership_losses.append(batch_all_loss)
                    deletion_membership_losses.append(batch_deletion_loss)

            all_membership_loss = torch.mean(torch.cat(all_membership_losses))
            deletion_membership_loss = torch.mean(torch.cat(deletion_membership_losses))

            # # Batched forward pass through the UNet
            # all_model_output = self.unet(all_noisy_images_flat, golden_timestep_expanded, return_dict=False)[0]
            # deletion_model_output = self.unet(deletion_noisy_images_flat, golden_timestep_expanded, return_dict=False)[0]

            # # Compute membership loss for all images
            # all_loss = torch.mean((all_model_output - noise_flat) ** 2, dim=[1, 2, 3])
            # all_membership_loss = torch.mean(all_loss)

            # # Compute membership loss for deletion images
            # deletion_loss = torch.mean((deletion_model_output - noise_flat) ** 2, dim=[1, 2, 3])
            # deletion_membership_loss = torch.mean(deletion_loss)
            membership_losses.append([all_membership_loss, deletion_membership_loss])
        return membership_losses