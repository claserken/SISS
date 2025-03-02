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


class TrainUnconditional(Task):
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
                    ema_model.save_pretrained(os.path.join(output_dir, self.cfg.subfolders.unet_ema))
                
                # breakpoint()
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, self.cfg.subfolders.unet))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                

        def load_model_hook(models, input_dir):
            if self.cfg.ema.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    hydra.utils.get_class(self.cfg.unet._target_),
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = hydra.utils.get_class(
                    self.cfg.unet._target_
                ).from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # Instantiate unet
        # if self.cfg.unet._type == "initialize":
        unet = hydra.utils.instantiate(self.cfg.unet, _convert_="all")
        # elif self.cfg.unet._type == "pretrained":
            # unet = hydra.utils.get_class(self.cfg.unet._target_).from_pretrained(
            #     self.cfg.unet.pretrained_model_name_or_path,
            #     subfolder="unet",
            #     revision=self.cfg.unet.revision,
            #     variant=self.cfg.unet.variant,
            # )

        # Create EMA for the model.
        if self.cfg.ema.use_ema:
            ema_model = EMAModel(
                unet.parameters(),
                decay=self.cfg.ema.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.cfg.ema.ema_inv_gamma,
                power=self.cfg.ema.ema_power,
                model_cls=hydra.utils.get_class(self.cfg.unet._target_),
                model_config=unet.config,
            )

        if self.cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        # Instantiate noise scheduler
        # if self.cfg.scheduler._type == "initialize":
        noise_scheduler = hydra.utils.instantiate(self.cfg.scheduler)
        # elif self.cfg.scheduler._type == "pretrained":
        #     noise_scheduler = hydra.utils.get_class(
        #         self.cfg.scheduler._target_
        #     ).from_pretrained(
        #         self.cfg.scheduler.pretrained_model_name_or_path, subfolder="scheduler"
        #     )
        

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
        dataset = hydra.utils.instantiate(self.cfg.dataset, transform=transform)

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

        logger.info(f"Dataset size: {len(dataset)}")

        # dataset.set_transform(transform_images)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader_num_workers,
        )

        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps * self.cfg.gradient_accumulation_steps,
            num_training_steps=(len(train_dataloader) * self.cfg.num_epochs),
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        if self.cfg.ema.use_ema:
            ema_model.to(accelerator.device)

        total_batch_size = (
            self.cfg.train_batch_size
            * accelerator.num_processes
            * self.cfg.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.cfg.gradient_accumulation_steps
        )
        max_train_steps = self.cfg.num_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {self.cfg.num_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.cfg.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.cfg.checkpoint_path:
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
            path = self.cfg.checkpoint_path
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * self.cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * self.cfg.gradient_accumulation_steps
            )

        # Train!
        for epoch in range(first_epoch, self.cfg.num_epochs):
            unet.train()
            progress_bar = tqdm(
                total=num_update_steps_per_epoch,
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                    self.cfg.checkpoint_path
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.cfg.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                clean_images = batch.to(weight_dtype)
                # Sample noise that we'll add to the images
                noise = torch.randn(
                    clean_images.shape, dtype=weight_dtype, device=clean_images.device
                )
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=clean_images.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(unet):
                    # Predict the noise residual
                    model_output = unet(noisy_images, timesteps).sample

                    if self.cfg.scheduler.prediction_type == "epsilon":
                        loss = F.mse_loss(
                            model_output.float(), noise.float()
                        )  # this could have different weights!
                    elif self.cfg.scheduler.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod,
                            timesteps,
                            (clean_images.shape[0], 1, 1, 1),
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(
                            model_output.float(), clean_images.float(), reduction="none"
                        )
                        loss = loss.mean()
                    else:
                        raise ValueError(
                            f"Unsupported prediction type: {self.cfg.scheduler.prediction_type}"
                        )

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # breakpoint()
                    if self.cfg.ema.use_ema:
                        ema_model.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        # Sample from model
                        if global_step % self.cfg.sampling_steps == 0:
                            unet = accelerator.unwrap_model(unet)

                            if self.cfg.ema.use_ema:
                                ema_model.store(unet.parameters())
                                ema_model.copy_to(unet.parameters())

                            # pipeline = DDPMPipeline(
                            #     unet=unet,
                            #     scheduler=noise_scheduler,
                            # )

                            # generator = torch.Generator(device=pipeline.device).manual_seed(0)
                            # # run pipeline in inference (sample random noise and denoise)
                            # images = pipeline(
                            #     generator=generator,
                            #     batch_size=self.cfg.eval_batch_size,
                            #     num_inference_steps=self.cfg.scheduler.num_inference_steps,
                            #     output_type="numpy",
                            # ).images
                            evaluator = Evaluator(self.cfg)
                            evaluator.load_model(unet, noise_scheduler)
                            images = evaluator.sample_images(self.cfg.eval_batch_size)
                            grid = Evaluator.make_grid_from_images(images)
                            if self.cfg.ema.use_ema:
                                ema_model.restore(unet.parameters())

                            # denormalize the images and save to tensorboard
                            # images_processed = (images * 255).round().astype("uint8")

                            # if self.cfg.logging.logger == "tensorboard":
                            #     if is_accelerate_version(">=", "0.17.0.dev0"):
                            #         tracker = accelerator.get_tracker(
                            #             "tensorboard", unwrap=True
                            #         )
                            #     else:
                            #         tracker = accelerator.get_tracker("tensorboard")
                            #     tracker.add_images(
                            #         "Sampled Images",
                            #         images_processed.transpose(0, 3, 1, 2),
                            #         epoch,
                            #     )
                            if self.cfg.logging.logger == "wandb":
                                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                                accelerator.get_tracker("wandb").log(
                                    {
                                        "Sampled Images": wandb.Image(grid),
                                        "epoch": epoch,
                                    },
                                    step=global_step,
                                )

                        # Save model
                        if global_step % self.cfg.checkpointing_steps == 0:
                            # Create checkpoint dir in case it doesn't exist yet (e.g. when resuming from another checkpoint)
                            os.makedirs(self.cfg.output_dir, exist_ok=True)
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
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                if self.cfg.ema.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            progress_bar.close()

            accelerator.wait_for_everyone()

            # Runs at the end of every epoch!
            # if accelerator.is_main_process:
                # print(f"Global step: {global_step}")
                # print(f"Checkpointing steps: {self.cfg.checkpointing_steps}")
                # if (
                #     # epoch % self.cfg.save_images_epochs == 0
                #     # or epoch == self.cfg.num_epochs - 1
                #     global_step % self.cfg.checkpointing_steps == 0
                # ):
                #     breakpoint()
                    
                # Save pipeline every epoch (not necessary with checkpointing steps)
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

        accelerator.end_training()
