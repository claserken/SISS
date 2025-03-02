import inspect
import logging
import math
import os
import shutil
import datetime
from pathlib import Path

import accelerate
from accelerate.utils import set_seed
import transformers
import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
from PIL import Image
from torch import linalg

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from evaluate import Evaluator

from main import Task
from losses.ddpm_deletion_loss import DDPMDeletionLoss
from data.utils.infinite_sampler import InfiniteSampler
from data.utils.repeat_sampler import RepeatedSampler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)    

class DeleteCeleb(Task):
    def __init__(
        self,
        cfg: DictConfig
    ):
        self.cfg = cfg        

    def run(self):
        # Logging
        logging_dir = os.path.join(self.cfg.output_dir, self.cfg.logging.logging_dir)

        if self.cfg.logging.logger == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError(
                    "Make sure to install tensorboard if you want to use it for logging during training."
                )
        elif self.cfg.logging.logger == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )
            import wandb

        # Set up accelerator
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.output_dir, logging_dir=logging_dir
        )
        kwargs = InitProcessGroupKwargs(
            timeout=datetime.timedelta(seconds=7200)
        )  # a big number for high resolution or big dataset
        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.logging.logger,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            wandb_conf = OmegaConf.to_container(self.cfg) # fully converts to dict (even nested keys)
            accelerator.init_trackers(project_name=self.cfg.project_name, config=wandb_conf)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed)

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if self.cfg.ema.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                
                # breakpoint()
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                

        # def load_model_hook(models, input_dir):
        #     if self.cfg.ema.use_ema:
        #         load_model = EMAModel.from_pretrained(
        #             os.path.join(input_dir, "unet_ema"),
        #             hydra.utils.get_class(self.cfg.unet._target_),
        #         )
        #         ema_model.load_state_dict(load_model.state_dict())
        #         ema_model.to(accelerator.device)
        #         del load_model

        #     for i in range(len(models)):
        #         # pop models so that they are not loaded again
        #         model = models.pop()

        #         # load diffusers style into model
        #         load_model = hydra.utils.get_class(
        #             self.cfg.unet._target_
        #         ).from_pretrained(input_dir, subfolder="unet")
        #         model.register_to_config(**load_model.config)

        #         model.load_state_dict(load_model.state_dict())
        #         del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

        # Instantiate unet
        # if self.cfg.unet._type == "initialize":
        #     unet = hydra.utils.instantiate(self.cfg.unet, _convert_="all")
        # elif self.cfg.unet._type == "pretrained":
        print(f'Loading checkpoint {self.cfg.checkpoint_path}')
        ddpm_pipeline = DDPMPipeline.from_pretrained(self.cfg.checkpoint_path)
        # unet = hydra.utils.get_class(self.cfg.unet._target_).from_pretrained(
        #     self.cfg.checkpoint_path,
        #     subfolder=self.cfg.subfolders.unet_ema if not self.cfg.ema.use_ema else self.cfg.subfolders.unet # load unet_ema if ema is disabled for deletion
        # )
        unet = ddpm_pipeline.unet

        # Create EMA for the model.
        if self.cfg.ema.use_ema:
            # ema_model = EMAModel(
            #     unet.parameters(),
            #     decay=self.cfg.ema.ema_max_decay,
            #     use_ema_warmup=True,
            #     inv_gamma=self.cfg.ema.ema_inv_gamma,
            #     power=self.cfg.ema.ema_power,
            #     model_cls=hydra.utils.get_class(self.cfg.unet._target_),
            #     model_config=unet.config,
            # )
            ema_model = EMAModel.from_pretrained(
                os.path.join(self.cfg.checkpoint_path, self.cfg.subfolders.unet_ema),
                hydra.utils.get_class(self.cfg.unet._target_),
            )

        # if self.cfg.enable_xformers_memory_efficient_attention:
        #     if is_xformers_available():
        #         import xformers

        #         xformers_version = version.parse(xformers.__version__)
        #         if xformers_version == version.parse("0.0.16"):
        #             logger.warn(
        #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
        #             )
        #         model.enable_xformers_memory_efficient_attention()
        #     else:
        #         raise ValueError(
        #             "xformers is not available. Make sure it is installed correctly"
        #         )

        # Instantiate noise scheduler
        # if self.cfg.scheduler._type == "initialize":
        #     noise_scheduler = hydra.utils.instantiate(self.cfg.scheduler)
        # elif self.cfg.scheduler._type == "pretrained":
        #     noise_scheduler = hydra.utils.get_class(
        #         self.cfg.scheduler._target_
        #     ).from_pretrained(
        #         self.cfg.checkpoint_path, subfolder=self.cfg.subfolders.noise_scheduler
        #     )
        # noise_scheduler = hydra.utils.instantiate(self.cfg.scheduler)
        noise_scheduler = ddpm_pipeline.scheduler

        # Instantiate optimizer
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, unet.parameters())

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.cfg.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.cfg.mixed_precision = accelerator.mixed_precision

        # ----- END SHARED CODE -----

        # Instantiate dataset
        # if self.cfg.dataset._type == "huggingface":
        #     dataset = load_dataset(
        #         self.cfg.dataset.dataset_name, self.cfg.dataset.dataset_config_name, split=self.cfg.dataset.split
        #     )
        # elif self.cfg.dataset._type == "custom":

        # Preprocessing the datasets and DataLoaders creation.
        transform = hydra.utils.instantiate(self.cfg.transform)
        dataset_all = hydra.utils.instantiate(self.cfg.dataset_all, transform=transform)
        dataset_deletion = hydra.utils.instantiate(self.cfg.dataset_deletion, transform=transform)

        # def transform_images(examples):
        #     if "image" in examples:
        #         image_key = "image"
        #     elif "img" in examples:
        #         image_key = "img"
        #     else:
        #         raise Exception(
        #             "You need to have a key 'image' or 'img' in your dataset"
        #         )
        #     images = [
        #         transformations(image.convert("RGB") if self.cfg.dataset.convert_to_rgb else image) for image in examples[image_key]
        #     ]
        #     return {"input": images}

        # def shapes_transform(image):
        #     image = augmentations(image.convert("RGB"))
        #     return {"input": image}

        infinite_sampler_all = InfiniteSampler(dataset_all)
        # sampler_deletion = InfiniteSampler(dataset_deletion)
        sampler_deletion = RepeatedSampler(dataset_deletion, num_repeats=self.cfg.training_steps*self.cfg.gradient_accumulation_steps*self.cfg.train_batch_size)

        dataloader_all = torch.utils.data.DataLoader(
            dataset_all,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            sampler=infinite_sampler_all
        )
        dataloader_deletion = torch.utils.data.DataLoader(
            dataset_deletion,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            sampler=sampler_deletion
        )
        dataset_all_iterator = iter(dataloader_all)
        dataset_deletion_iterator = iter(dataloader_deletion)
        
        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.training_steps
        )

        # Prepare everything with our `accelerator` (except optimizer and lr_scheduler)
        unet, optimizer, lr_scheduler, dataset_all_iterator, dataset_deletion_iterator = accelerator.prepare(
            unet, optimizer, lr_scheduler, dataset_all_iterator, dataset_deletion_iterator
        )

        if self.cfg.ema.use_ema:
            ema_model.to(accelerator.device)

        total_batch_size = (
            self.cfg.train_batch_size
            * accelerator.num_processes
            * self.cfg.gradient_accumulation_steps
        )
        # num_update_steps_per_epoch = math.ceil(
        #     len(train_dataloader) / self.cfg.gradient_accumulation_steps
        # )
        # max_train_steps = self.cfg.num_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"Num all examples: {len(dataset_all)}")
        logger.info(f"Num deletion examples: {len(dataset_deletion)}")
        logger.info(f"  Num training steps = {self.cfg.training_steps}")
        logger.info(
            f"  Instantaneous batch size per device = {self.cfg.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        # logger.info(f"  Total optimization steps = {max_train_steps}")

        global_step = 0
        # first_epoch = 0

        # Potentially load in the weights and states from a previous save
        # if self.cfg.checkpoint_path:
            # if self.cfg.resume_from_checkpoint != "latest":
            #     path = os.path.basename(self.cfg.resume_from_checkpoint)
            # else:
            #     # Get the most recent checkpoint
            #     dirs = os.listdir(self.cfg.output_dir)
            #     dirs = [d for d in dirs if d.startswith("checkpoint")]
            #     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            #     path = dirs[-1] if len(dirs) > 0 else None

            # if path is None:
            #     accelerator.print(
            #         f"Checkpoint '{self.cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            #     )
            #     self.cfg.resume_from_checkpoint = None
            # else:
            # path = self.cfg.checkpoint_path
            # accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(path)
            # global_step = int(path.split("-")[1])

            # resume_global_step = global_step * self.cfg.gradient_accumulation_steps
            # first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step % (
            #     num_update_steps_per_epoch * self.cfg.gradient_accumulation_steps
            # )

        gamma = noise_scheduler.alphas_cumprod ** 0.5
        gamma = gamma.to(accelerator.device)

        sigma = (1 - noise_scheduler.alphas_cumprod) ** 0.5
        sigma = sigma.to(accelerator.device)

        loss_class = DDPMDeletionLoss(gamma=gamma, sigma=sigma) #, frac_deletion=self.cfg.deletion.frac_deletion)
        loss_fn = getattr(loss_class, self.cfg.deletion.loss_fn)

        def evaluate_model(unet, num_imgs_to_generate=self.cfg.eval_batch_size, batch_size=self.cfg.eval_batch_size, set_generator=False):
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler)
            
            generated_images = []
            remaining_images = num_imgs_to_generate

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)
                images = evaluator.sample_images(current_batch_size, set_generator=set_generator)
                generated_images.append(images)
                remaining_images -= current_batch_size
            
            generated_images = np.concatenate(generated_images, axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters())

            unet.train()
            return generated_images
        
        def evaluate_unlearning_timestep(unet, timestep, clean_image, num_imgs_to_generate=self.cfg.eval_batch_size, batch_size=self.cfg.eval_batch_size):
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler)
            
            generated_images = []
            remaining_images = num_imgs_to_generate
            generator = torch.Generator(device=accelerator.device).manual_seed(self.cfg.random_seed)

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)

                noise = torch.randn((current_batch_size, *clean_image.shape), generator=generator, device=accelerator.device)
                timestep_expanded = torch.full((current_batch_size,), timestep, device=accelerator.device)
                noisy_image_batch = noise_scheduler.add_noise(clean_image, noise, timestep_expanded)
                images = evaluator.denoise_images(noisy_image_batch, timestep)
                # breakpoint()
                generated_images.append(images)
                remaining_images -= current_batch_size
            
            generated_images = np.concatenate([image.cpu() for image in generated_images], axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters())

            unet.train()
            return generated_images

        if self.cfg.metrics.membership_loss: 
            membership_class = hydra.utils.instantiate(self.cfg.metrics.membership_loss.class_cfg, dataset_all=dataset_all, dataset_deletion=dataset_deletion, noise_scheduler=noise_scheduler, unet=unet, device=accelerator.device)
            membership_class.sample_images()
            membership_class.sample_noises()

            plot_params = self.cfg.metrics.membership_loss.plot_params
            if plot_params is not None:
                time_freq = plot_params.time_frequency
                timesteps = list(range(0, self.cfg.scheduler.num_train_timesteps, time_freq))
                losses = membership_class.compute_membership_losses(timesteps)
                all_membership_losses = [loss[0].cpu() for loss in losses]

                # Create plot
                fig, ax = plt.subplots()
                ax.plot(timesteps, all_membership_losses)

                # Set the labels and title
                ax.set_xlabel('Timestep')
                ax.set_ylabel('All membership loss')
                ax.set_title('Loss over time')

                # Save the plot to a file
                plt.savefig(plot_params.save_path)

                exit()

                    

        if self.cfg.metrics.classifier_cfg:
            classifier = hydra.utils.instantiate(self.cfg.metrics.classifier_cfg, device=accelerator.device)
        
        if self.cfg.metrics.fid:
            fid_calculator = hydra.utils.instantiate(self.cfg.metrics.fid.class_cfg, device=accelerator.device)
            fid_calculator.load_celeb()

        if self.cfg.metrics.denoising_injections:
            injection_timestep = torch.tensor(self.cfg.metrics.denoising_injections.timestep, device=accelerator.device)
            target_image = transform(Image.open(self.cfg.metrics.denoising_injections.img_path)).to(accelerator.device)

        if self.cfg.metrics.likelihood:
            likelihood_calculator = hydra.utils.instantiate(self.cfg.metrics.likelihood.class_cfg)

        deletion_tracker = {
            'deletion_reached': False,
            'deletion_steps': None
        }
        def log_metrics(unet):
            wandb_batch = {}
            if global_step % self.cfg.sampling_steps == 0:
                images = evaluate_model(unet, set_generator=True) # fix randomness 
                grid = Evaluator.make_grid_from_images(images)
                wandb_batch["Sampled Images"] = wandb.Image(grid)

                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format
                if self.cfg.metrics.fraction_deletion:
                    frac_deletion = classifier.compute_class_frequency(images, self.cfg.deletion.class_label) 
                    wandb_batch["metrics/deletion_class_fraction"] = frac_deletion
                    if frac_deletion == 0 and not deletion_tracker['deletion_reached']:
                        wandb.run.summary['deletion_steps'] = global_step
                        deletion_tracker['deletion_steps'] = global_step
                        deletion_tracker['deletion_reached'] = True
                
                if self.cfg.metrics.denoising_injections:
                    targeted_image_generations = evaluate_unlearning_timestep(unet, injection_timestep, target_image)
                    target_image_grid = Evaluator.make_grid_from_images(targeted_image_generations)
                    wandb_batch[f"Target Image Generations (t={injection_timestep})"] = wandb.Image(target_image_grid)
                
                if self.cfg.metrics.likelihood and (global_step % self.cfg.metrics.likelihood.step_frequency == 0):
                    print('Computing likelihood...')
                    likelihood = likelihood_calculator.evaluate_likelihood(unet, target_image.unsqueeze(0))[0][0].item()
                    wandb_batch["metrics/likelihood"] = likelihood
                    print(f'Likelihood: {likelihood}')
            
                
            if self.cfg.metrics.membership_loss and (global_step % self.cfg.metrics.membership_loss.step_frequency == 0):
                timesteps = self.cfg.metrics.membership_loss.timesteps
                membership_losses = membership_class.compute_membership_losses(timesteps)
                for idx, timestep in enumerate(timesteps):
                    wandb_batch[f"membership_loss/all_membership_loss_t={timestep}"] = membership_losses[idx][0]
                    wandb_batch[f"membership_loss/deletion_membership_loss_t={timestep}"] = membership_losses[idx][1]
                    wandb_batch[f"membership_loss/membership_ratio_t={timestep}"] = membership_losses[idx][1] / membership_losses[idx][0]

            if self.cfg.metrics.inception_score and (global_step % self.cfg.metrics.inception_score.step_frequency == 0 or deletion_tracker['deletion_steps'] == global_step):
                images = evaluate_model(unet, num_imgs_to_generate=self.cfg.metrics.inception_score.num_imgs_to_generate, batch_size=self.cfg.metrics.inception_score.batch_size)
                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format

                inception_calculator = hydra.utils.instantiate(self.cfg.metrics.inception_score.class_cfg, classifier=classifier)
                inception_calculator.update(images)
                is_mean, is_std = inception_calculator.compute()
                wandb_batch['metrics/is_mean'] = is_mean.item()
                wandb_batch['metrics/is_std'] = is_std.item()
                print(f'IS_mean: {is_mean}')
                print(f'IS_std: {is_std}')
            
            if self.cfg.metrics.fid:
                should_compute_fid = self.cfg.metrics.fid.step_frequency and (global_step % self.cfg.metrics.fid.step_frequency == 0)
                if should_compute_fid or deletion_tracker['deletion_steps'] == global_step:
                    print('Computing FID...')
                    images = evaluate_model(unet, num_imgs_to_generate=self.cfg.metrics.fid.num_imgs_to_generate, batch_size=self.cfg.metrics.fid.batch_size)
                    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format

                    fid_calculator.add_fake_images(images)
                    fid = fid_calculator.compute(verbose=True)
                    wandb_batch['metrics/fid'] = fid.item()
                    print(f'FID: {fid}')

            if self.cfg.logging.logger == "wandb":
                accelerator.get_tracker("wandb").log(wandb_batch, step=global_step)

        # Train!
        progress_bar = tqdm(
            total=self.cfg.training_steps * len(self.cfg.deletion.img_name),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Steps")
        log_metrics(unet)

        accum_loss_x = {}
        accum_loss_a = {}
        while global_step < self.cfg.training_steps * len(self.cfg.deletion.img_name):
            unet.train()

            all_images = next(dataset_all_iterator)
            all_images = all_images.to(device=accelerator.device, dtype=weight_dtype)

            deletion_images = next(dataset_deletion_iterator)
            deletion_images = deletion_images.to(device=accelerator.device, dtype=weight_dtype)

            target_image = deletion_images[0].squeeze()
            
            # for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
            # if (
            #     self.cfg.resume_from_checkpoint
            #     and epoch == first_epoch
            #     and step < resume_step
            # ):
            #     if step % self.cfg.gradient_accumulation_steps == 0:
            #     continue

            # all_images = batch.to(weight_dtype)
            # Sample noise that we'll add to the images (same one for both all and deletion!)
            assert all_images.shape == deletion_images.shape
            noise = torch.randn(
                all_images.shape, dtype=weight_dtype, device=all_images.device
            )
            # deletion_noise = torch.randn(
            #     deletion_images.shape, dtype=weight_dtype, device=deletion_images.device
            # )

            bsz = all_images.shape[0]
            # Sample a random timestep for each image (same for all and deletion pairs)
            if self.cfg.deletion.timestep_del_window is None and self.cfg.deletion.loss_fn == "modified_noise_obj":
                raise ValueError('Timestep deletion window length must be provided for the modified_noise_obj loss function.')
            start_timestep = noise_scheduler.config.num_train_timesteps - self.cfg.deletion.timestep_del_window if self.cfg.deletion.loss_fn == "modified_noise_obj" else 0
            timesteps = torch.randint(
                999, #start_timestep,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=all_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_all_images = noise_scheduler.add_noise(all_images, noise, timesteps)
            noisy_deletion_images = noise_scheduler.add_noise(deletion_images, noise, timesteps)
            
            all_samples_dict = {
                "og_latents": all_images,
                # "noise": all_noise,
                "noisy_latents": noisy_all_images,
            }

            deletion_samples_dict = {
                "og_latents": deletion_images,
                # "noise": deletion_noise,
                "noisy_latents": noisy_deletion_images,
            }

            with accelerator.accumulate(unet):
                # Predict the noise residual
                # model_output = unet(noisy_all_images, timesteps).sample

                if self.cfg.scheduler.prediction_type == "epsilon":
                    items = loss_fn(unet, timesteps, noise, {}, all_samples_dict, deletion_samples_dict, **self.cfg.deletion.loss_params)
                    loss, loss_x, loss_a, importance_weight_x, importance_weight_a, weighted_loss_x, weighted_loss_a = (
                        items
                    )
                    batch_stats = {}
                    if loss is not None:
                        print("loss:", loss.mean().item())
                        batch_stats["loss/mean"] = loss.mean().item()
                        batch_stats["loss/max"] = loss.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss/min"] = loss.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss/std"] = loss.mean(dim=[1, 2, 3]).std().item()

                    if loss_x is not None:
                        batch_stats["loss_x/mean"] = loss_x.mean().item()
                        batch_stats["loss_x/max"] = loss_x.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_x/min"] = loss_x.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_x/std"] = loss_x.mean(dim=[1, 2, 3]).std().item()
                    
                    if loss_a is not None:
                        batch_stats["loss_a/mean"] = loss_a.mean().item()
                        batch_stats["loss_a/max"] = loss_a.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_a/min"] = loss_a.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_a/std"] = loss_a.mean(dim=[1, 2, 3]).std().item()
                    
                    if importance_weight_x is not None:
                        batch_stats["importance_weight_x/mean"] = importance_weight_x.mean().item()
                        batch_stats["importance_weight_x/max"] = importance_weight_x.max().item()
                        batch_stats["importance_weight_x/min"] = importance_weight_x.min().item()
                        batch_stats["importance_weight_x/std"] = importance_weight_x.std().item()

                    if importance_weight_a is not None:
                        batch_stats["importance_weight_a/mean"] = importance_weight_a.mean().item()
                        batch_stats["importance_weight_a/max"] = importance_weight_a.max().item()
                        batch_stats["importance_weight_a/min"] = importance_weight_a.min().item()
                        batch_stats["importance_weight_a/std"] = importance_weight_a.std().item()

                    if "superfactor" in self.cfg.deletion.loss_params:
                        batch_stats["superfactor"] = self.cfg.deletion.loss_params.superfactor
                        decay = self.cfg.deletion.superfactor_decay
                        if decay is not None:
                            self.cfg.deletion.loss_params.superfactor *= self.cfg.deletion.superfactor_decay
                    wandb.log(batch_stats, step=global_step)
                elif self.cfg.scheduler.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod,
                        timesteps,
                        (all_images.shape[0], 1, 1, 1),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(
                        model_output.float(), all_images.float(), reduction="none"
                    )
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {self.cfg.scheduler.prediction_type}"
                    )
                
                # breakpoint()
                if loss is not None:
                    loss = loss.sum() / self.cfg.train_batch_size
                    accelerator.backward(loss)
                else:
                    weighted_loss_x = weighted_loss_x.sum() / self.cfg.train_batch_size
                    weighted_loss_a = weighted_loss_a.sum() / self.cfg.train_batch_size

                    # only keep graph in case of SISS since double forward and erasediff dont mix noise_preds between loss_x and loss_a
                    retain_graph = self.cfg.deletion.loss_fn == "importance_sampling_with_mixture"
                    accelerator.backward(weighted_loss_x, retain_graph=retain_graph)

                    gradients_loss_x = {}
                    for name, param in unet.named_parameters():
                        # if param.grad is not None:
                        gradients_loss_x[name] = param.grad.clone()
                        # if name not in accum_loss_x:
                        #     accum_loss_x[name] = param.grad.clone()  # First time, initialize with clone of the gradient
                        # else:
                        #     accum_loss_x[name] += param.grad.clone()  # Add gradients for manual accumulation

                    accelerator.backward(weighted_loss_a)

                    # gradients_loss_a = {}
                    for name, param in unet.named_parameters():
                        # if param.grad is not None:
                        true_grad = param.grad.clone() - gradients_loss_x[name]
                        if name not in accum_loss_a:
                            accum_loss_a[name] = true_grad  # First time, initialize with clone of the gradient
                        else:
                            accum_loss_a[name] += true_grad  # Add gradients for manual accumulation
                    # print('accumulating')
                
                if accelerator.sync_gradients:
                    # print('prepare to step!')
                    if loss is None:
                        for name, param in unet.named_parameters():
                            accum_loss_x[name] = param.grad.clone() - accum_loss_a[name]

                        # Initialize variables to hold the sum of squared norms for both accum_loss_x and accum_loss_a
                        total_l2_norm_x = 0.0
                        total_l2_norm_a = 0.0

                        # Compute L2 norm for accum_loss_x
                        for grad in accum_loss_x.values():
                            total_l2_norm_x += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Compute L2 norm for accum_loss_a
                        for grad in accum_loss_a.values():
                            total_l2_norm_a += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Take the square root of the sum to get the final L2 norms
                        total_l2_norm_x = torch.sqrt(total_l2_norm_x)
                        total_l2_norm_a = torch.sqrt(total_l2_norm_a)

                        print(f"L2 norm of accum_loss_x: {total_l2_norm_x}")
                        print(f"L2 norm of accum_loss_a: {total_l2_norm_a}")

                        
                        if self.cfg.deletion.loss_fn == 'erasediff':
                            # scaling_factor = self.cfg.deletion.eta - sum(torch.sum(accum_loss_x[name] * accum_loss_a[name]) for name, _ in unet.named_parameters()) / (total_l2_norm_a ** 2)
                            # scaling_factor = -max(scaling_factor, 0)
                            scaling_factor = -self.cfg.deletion.scaling_norm / total_l2_norm_a
                        else:
                            # gradient norm fixing 
                            scaling_factor = self.cfg.deletion.scaling_norm / total_l2_norm_a # if total_l2_norm_a > self.cfg.deletion.loss_params.clipping_norm else 1
                        
                        wandb.log({'gradient/norm_loss_x': total_l2_norm_x, 'gradient/norm_loss_a': total_l2_norm_a, 'gradient/scaling_factor': scaling_factor}, step=global_step)
                        for name, param in unet.named_parameters():
                            param.grad = accum_loss_x[name] - scaling_factor * accum_loss_a[name]  # Set the gradient to the clipped difference

                        accum_loss_x = {}
                        accum_loss_a = {}

                    # # grads is a list of gradient tensors from the parameters
                    # grads = [p.grad for p in unet.parameters() if p.grad is not None]

                    # # Flatten each gradient tensor and concatenate them into a single tensor
                    # flat_grads = torch.cat([g.view(-1) for g in grads])

                    # # Compute the L2 norm (Euclidean norm) of the concatenated gradients
                    # grad_norm = torch.norm(flat_grads, p=2)
                    # print(grad_norm)

                    # wandb.log({'gradient/norm': grad_norm}, step=global_step)
                    # breakpoint()
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()

                
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if self.cfg.ema.use_ema:
                    ema_model.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # Log metrics
                    log_metrics(unet)


                    # Save model
                    if self.cfg.checkpointing_steps and global_step % self.cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if self.cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(self.cfg.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= self.cfg.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - self.cfg.checkpoints_total_limit
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        self.cfg.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            self.cfg.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            if loss is not None:
                logs["loss"] = loss.detach().item()
            if self.cfg.ema.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()

            # Runs at the end of every step!
            # if accelerator.is_main_process:
                # print(f"Global step: {global_step}")
                # print(f"Checkpointing steps: {self.cfg.checkpointing_steps}")
                # if (
                #     # epoch % self.cfg.save_images_epochs == 0
                #     # or epoch == self.cfg.num_epochs - 1
                #     global_step % self.cfg.checkpointing_steps == 0
                # ):
                #     breakpoint()
                    

                # if (
                #     epoch % self.cfg.save_model_epochs == 0
                #     or epoch == self.cfg.num_epochs - 1
                # ):
                #     # save the model
                #     unet = accelerator.unwrap_model(unet)

                #     if self.cfg.ema.use_ema:
                #         ema_model.store(unet.parameters())
                #         ema_model.copy_to(unet.parameters())

                #     pipeline = DDPMPipeline(
                #         unet=unet,
                #         scheduler=noise_scheduler,
                #     )

                #     pipeline.save_pretrained(self.cfg.output_dir)

                #     if self.cfg.ema.use_ema:
                #         ema_model.restore(unet.parameters())
        
        # The end
        progress_bar.close()
        accelerator.end_training()
