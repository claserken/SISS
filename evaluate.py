import os
import hydra
from diffusers.training_utils import EMAModel
import torch
from diffusers import DDPMPipeline, DDIMPipeline
import torchvision.utils as utils
import numpy as np
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image

class Evaluator:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def load_model(self, unet, noise_scheduler):
        self.unet = unet
        self.noise_scheduler = noise_scheduler

    def load_fpath(self, unet_fpath, is_ema: bool):
        self.unet = hydra.utils.get_class(self.cfg.unet._target_).from_pretrained(
                unet_fpath,
                subfolder="unet",
                # revision=self.cfg.unet.revision,
                # variant=self.cfg.unet.variant,
        )
        if is_ema:
            ema_unet = EMAModel.from_pretrained(
                os.path.join(unet_fpath, "unet_ema"),
                hydra.utils.get_class(self.cfg.unet._target_),
            )
            ema_unet.copy_to(self.unet.parameters())   
                 
        self.noise_scheduler = hydra.utils.instantiate(self.cfg.scheduler)


    def sample_images(self, num_samples, num_inference_steps=None, set_generator=False):
        # ddpm_pipeline = DDIMPipeline.from_pretrained('google/ddpm-celebahq-256').to('cuda')
        pipeline = DDPMPipeline(
            unet=self.unet,
            scheduler=self.noise_scheduler,
        ) # pipeline auto-converts from range [-1, 1] to [0, 1] with clamping
        generator = torch.Generator(device=pipeline.device).manual_seed(self.cfg.random_seed) if set_generator else None
        images = pipeline(
            generator=generator,
            batch_size=num_samples,
            num_inference_steps=num_inference_steps if num_inference_steps is not None else self.cfg.pipeline.num_inference_steps,
            output_type="numpy",
        ).images # shape is channel-last (N, H, W, C) which wandb prefers 
        return images
        
        # just for google
        # breakpoint()
        # model_id = "google/ddpm-cifar10-32"
        # ddim = DDIMPipeline.from_pretrained(model_id).to('cuda')
        # ddim.unet = self.unet
        # generator = torch.Generator(device=ddim.device).manual_seed(
        #      seed if seed is not None else self.cfg.random_seed
        # )
        # images = ddim(generator=generator, batch_size=num_samples, num_inference_steps=self.cfg.pipeline.num_inference_steps, output_type="numpy").images
        # return images

    # Injects noise then denoises
    def denoise_images(self, noisy_image_batch, timestep, num_inference_steps=None, set_generator=True):
        device = self.unet.device
        ddpm_pipeline = DDPMPipeline.from_pretrained(self.cfg.checkpoint_path).to(device)
        generator = torch.Generator(device=device).manual_seed(self.cfg.random_seed) if set_generator else None
        with torch.no_grad():
            for t in tqdm(reversed(range(timestep + 1))):
                model_output = self.unet(noisy_image_batch, t)["sample"]
                # model_output2 = self.unet(noisy_image_batch, t)["sample"]
                # print('Unet is same' if torch.all(model_output == model_output2) else 'Unet output is diff')
                noisy_image_batch = ddpm_pipeline.scheduler.step(model_output, t, noisy_image_batch, generator=generator)["prev_sample"]
                # noisy_image_batch2 = self.noise_scheduler.step(model_output, t, noisy_image_batch, generator=torch.Generator(device='cuda').manual_seed(self.cfg.random_seed))["prev_sample"]
                # breakpoint()
                # print('Batch is same' if torch.all(noisy_image_batch == noisy_image_batch2) else 'Batch output is diff')
        noisy_image_batch = (noisy_image_batch + 1) / 2
        noisy_image_batch = noisy_image_batch.clamp(0, 1)
        return noisy_image_batch.permute(0, 2, 3, 1)

    @staticmethod
    def make_grid_from_images(images):
        num_samples = images.shape[0]
        num_channels = images.shape[-1]
        
        # make_grid has input and output as channel-first 
        grid = utils.make_grid(torch.from_numpy(images).permute(0, 3, 1, 2), nrow=int(np.sqrt(num_samples)))
        if num_channels == 1:
            grid = np.array(grid.permute(1, 2, 0)[:, :, :1]) # make_grid adjusts num_channels to 3 so slice first dim here
        return grid