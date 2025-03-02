#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

from torchmetrics.multimodal import CLIPImageQualityAssessment

import accelerate
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import json
import joblib
from PIL import Image
import torchvision.transforms.functional as F


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from data.src.local_sd_pipeline import LocalStableDiffusionPipeline

from main import Task
from omegaconf import DictConfig, OmegaConf
import hydra

from data.src.sd_dataset import SDData
# from data.src.config import IMG_DIR, LABELS_PATH, PROMPT
from data.utils.infinite_sampler import InfiniteSampler
from losses.ddpm_deletion_loss import DDPMDeletionLoss
# from classifier import classify_all_images, get_average_image

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class DeleteSD(Task):
    def __init__(
        self,
        cfg: DictConfig
    ):
        self.cfg = cfg

    def save_model_card(
        self,
        repo_id: str,
        images=None,
        repo_folder=None,
    ):
        img_str = ""
        if len(images) > 0:
            image_grid = make_image_grid(images, 1, len(self.cfg.validation_prompts))
            image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
            img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

        yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {self.cfg.pretrained_model_name_or_path}
    datasets:
    - {self.cfg.dataset_name}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - text-to-image
    - diffusers
    inference: true
    ---
        """
        model_card = f"""
    # Text-to-image finetuning - {repo_id}

    This pipeline was finetuned from **{self.cfg.pretrained_model_name_or_path}** on the **{self.cfg.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {self.cfg.validation_prompts}: \n
    {img_str}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
    prompt = "{self.cfg.validation_prompts[0]}"
    image = pipeline(prompt).images[0]
    image.save("my_image.png")
    ```

    ## Training info

    These are the key hyperparameters used during training:

    * Epochs: {self.cfg.num_train_epochs}
    * Learning rate: {self.cfg.learning_rate}
    * Batch size: {self.cfg.train_batch_size}
    * Gradient accumulation steps: {self.cfg.gradient_accumulation_steps}
    * Image resolution: {self.cfg.resolution}
    * Mixed-precision: {self.cfg.mixed_precision}

    """
        wandb_info = ""
        if is_wandb_available():
            wandb_run_url = None
            if wandb.run is not None:
                wandb_run_url = wandb.run.url

        if wandb_run_url is not None:
            wandb_info = f"""
    More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
    """

        model_card += wandb_info

        with open(os.path.join(repo_folder, "README.md"), "w") as f:
            f.write(yaml + model_card)

    def log_validation(
        self, vae, text_encoder, tokenizer, unet, accelerator, weight_dtype, img_count
    ):
        logger.info("Running validation... ")
        with torch.no_grad():
            if self.cfg.using_augmented_prompt:
                aug_prompt_embeds = torch.load(self.cfg.validation_prompts[0])
                
            if self.cfg.using_augmented_prompt or self.cfg.metrics.noise_norm:
                pipeline = LocalStableDiffusionPipeline.from_pretrained(
                    self.cfg.pretrained_model_name_or_path,
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=accelerator.unwrap_model(unet),
                    safety_checker=None,
                    revision=self.cfg.revision,
                    variant=self.cfg.ema_variant,
                    torch_dtype=weight_dtype,
                    device_map=accelerator.device
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.cfg.pretrained_model_name_or_path,
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=accelerator.unwrap_model(unet),
                    safety_checker=None,
                    revision=self.cfg.revision,
                    variant=self.cfg.ema_variant,
                    torch_dtype=weight_dtype,
                    device_map=accelerator.device
                )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            # pipeline = pipeline.to(accelerator.device)

                
            # pipeline.set_progress_bar_config(disable=True)

            if self.cfg.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(
                    self.cfg.seed
                )

            images = []
            # num_inference_steps=20 creates issues with low quality especially for memorized image
            if self.cfg.metrics.clip_iqa:
                clip_computer = CLIPImageQualityAssessment().to(accelerator.device)
            if self.cfg.metrics.fraction_deletion:
                kmeans_classifier = joblib.load(self.cfg.metrics.fraction_deletion.classifier_path)
            if self.cfg.metrics.sscd:
                sscd_computer = torch.jit.load(self.cfg.metrics.sscd.model_path).to(accelerator.device)
                sscd_transform = hydra.utils.instantiate(self.cfg.metrics.sscd.data_transforms)

            for i in range(len(self.cfg.validation_prompts)):
                with torch.autocast("cuda"):
                    images_for_prompt = []
                    noise_norms = []
                    for _ in range(self.cfg.eval_batches):
                        if self.cfg.using_augmented_prompt and i == 0:
                            pipe_out, noise_stats = pipeline(
                                prompt_embeds=aug_prompt_embeds,
                                generator=generator,
                                num_images_per_prompt=self.cfg.eval_batch_size,
                                track_noise_norm=self.cfg.metrics.noise_norm
                            )
                        else:
                            pipe_out, noise_stats = pipeline(
                                self.cfg.validation_prompts[i],
                                generator=generator,
                                num_images_per_prompt=self.cfg.eval_batch_size,
                                track_noise_norm=self.cfg.metrics.noise_norm
                            )
                        images_for_prompt.extend(pipe_out.images)
                        noise_norms.extend(noise_stats['text_noise_norm'])
                        # images.extend([(new_image, self.cfg.validation_prompts[i]) for new_image in new_images])
                    # Convert images to tensors
                    to_tensor = transforms.ToTensor()
                    image_tensors = torch.stack([to_tensor(image) for image in images_for_prompt])

                    # Create a grid of images
                    grid = make_grid(image_tensors, nrow=int(math.sqrt(self.cfg.eval_batches * self.cfg.eval_batch_size)))

                    # Append the grid and caption to the images list
                    images.append((grid, self.cfg.validation_prompts[i]))

                    metrics = {}
                    image_tensors = image_tensors.to(accelerator.device)
                    if self.cfg.metrics.clip_iqa:
                        clip_scores = clip_computer(image_tensors)
                        avg_clip_score = clip_scores.mean().item()
                        metrics[f"metrics/clip_iqa_{i}"] = avg_clip_score
                    
                    if self.cfg.metrics.fraction_deletion:
                        SCALE_FACTOR = 255 
                        preds = kmeans_classifier.predict(SCALE_FACTOR * image_tensors.to('cpu').permute(0, 2, 3, 1).flatten(start_dim=1))
                        fraction_memorized = preds.mean()
                        metrics[f"metrics/deletion_fraction_{i}"] = fraction_memorized
                        if fraction_memorized == 0 and (f"deletion_steps_{i}" not in dict(wandb.run.summary)):
                            wandb.run.summary[f"deletion_steps_{i}"] = img_count / self.cfg.imgs_per_gradient
                    
                    if self.cfg.metrics.sscd:
                        mem_img = F.to_tensor(Image.open(self.cfg.data_files.mem_img_path).convert('RGB')).to(accelerator.device)
                        mem_embedding = sscd_computer(sscd_transform(mem_img).unsqueeze(0))
                        all_img_embeddings = sscd_computer(sscd_transform(image_tensors))
                        sscd_scores = torch.matmul(mem_embedding, all_img_embeddings.T).squeeze()
                        avg_sscd = sscd_scores.mean().item()
                        metrics[f"metrics/sscd_{i}"] = avg_sscd

                    if self.cfg.metrics.noise_norm:
                        if not hasattr(self, 'noise_norms'):
                            self.noise_norms = [[] for _ in range(len(self.cfg.validation_prompts))]
                        noise_norm_tensor = torch.tensor(noise_norms)
                        avg_noise_norms = noise_norm_tensor.mean(dim=0).tolist()
                        avg_noise_norms.reverse() 
                        self.noise_norms[i].append(avg_noise_norms)
                        wandb.log({f"noise_norms/noise_norms_{i}" : wandb.plot.line_series(
                                    xs=list(range(0, 1020, 20)), 
                                    ys=self.noise_norms[i],
                                    keys=list(range(len(self.noise_norms[i]))),
                                    title=f"Text-conditional noise norm (prompt {i})",
                                    xname="Timestep")},
                                    step=img_count)


                    wandb.log(metrics, step=img_count)

        # average_image = get_average_image()
        # percent_label_1 = classify_all_images(images, average_image)
        # print(f"Percentage of images labeled '1': {percent_label_1}%")

        # images_tensor = torch.stack([torch.tensor(np.array(img).transpose(2, 0, 1)).float() for img in images])/255.0
        # clip_iqa_metric = CLIPImageQualityAssessment(prompts=("quality",))
        # with torch.no_grad():
        #     clip_scores = clip_iqa_metric(images_tensor)
        #     average_clip_score = torch.mean(clip_scores).item()
        # print(f"Average CLIP-IQA Score for 'high quality': {average_clip_score}")

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(
                    "validation", np_images, img_count, dataformats="NHWC"
                )
            elif tracker.name == "wandb":
                wandb.log(
                    {
                        "Generated Images": [
                            wandb.Image(
                                image, caption=caption
                            )
                            for image, caption in images
                        ],
                        # "percent_label_1": percent_label_1,
                        # "average_clip_score_high_quality": average_clip_score
                    },
                    step=img_count,
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        torch.cuda.empty_cache()

        return images

    def fill_cfg(self):
        if self.cfg.deletion.frac_deletion is None:
           with open(self.cfg.data_files.clustering_info_path, "r", encoding='utf-8') as file:
              clustering_info = json.load(file)
           self.cfg.deletion.frac_deletion = clustering_info['frac_deletion']
           mem_img_name = self.cfg.images_name + '_' + str(clustering_info['mem_idx']).zfill(3) + '.png'
           self.cfg.data_files.mem_img_path = os.path.join(self.cfg.data_files.img_dir, mem_img_name)

        
        if self.cfg.validation_prompts is None:
            self.cfg.validation_prompts = []
            with open(self.cfg.modified_prompts_path, "r", encoding='utf-8') as file:
              modified_prompts = json.load(file)
              self.cfg.validation_prompts.append(modified_prompts[self.cfg.images_name])

            with open(self.cfg.og_prompts_path, "r", encoding='utf-8') as file:
              og_prompts = json.load(file)
              self.cfg.validation_prompts.append(og_prompts[self.cfg.images_name])

        # Check for modified prompt 
        self.cfg.using_augmented_prompt = self.cfg.validation_prompts[0].endswith('.pt') 

    def run(self):
        self.fill_cfg() 
        logging_dir = os.path.join(self.cfg.output_dir, self.cfg.logging_dir)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.output_dir, logging_dir=logging_dir
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.report_to,
            project_config=accelerator_project_config,
        )

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
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.cfg.output_dir is not None:
                os.makedirs(self.cfg.output_dir, exist_ok=True)

            if self.cfg.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.cfg.hub_model_id or Path(self.cfg.output_dir).name,
                    exist_ok=True,
                    token=self.cfg.hub_token,
                ).repo_id

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.cfg.revision,
        )

        def deepspeed_zero_init_disabled_context_manager():
            """
            returns either a context list that includes one that will disable zero.Init or an empty context list
            """
            deepspeed_plugin = (
                AcceleratorState().deepspeed_plugin
                if accelerate.state.is_initialized()
                else None
            )
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
        # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.cfg.revision,
                variant=self.cfg.frozen_variant,
            )
            vae = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.cfg.revision,
                variant=self.cfg.frozen_variant,
            )

        unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.cfg.revision,
            variant=self.cfg.non_ema_variant if self.cfg.use_ema else self.cfg.ema_variant
        )

        # Freeze vae and text_encoder and set unet to trainable
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()

        # Create EMA for the unet.
        if self.cfg.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="unet",
                revision=self.cfg.revision,
                variant=self.cfg.ema_variant
            )
            ema_unet = EMAModel(
                ema_unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=ema_unet.config,
            )

        if self.cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if self.cfg.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):
                if self.cfg.use_ema:
                    load_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                    )
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

        if self.cfg.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate
                * self.cfg.gradient_accumulation_steps
                * self.cfg.train_batch_size
                * accelerator.num_processes
            )

        # Initialize the optimizer
        if self.cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            unet.parameters(),
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        # if self.cfg.dataset_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     dataset = load_dataset(
        #         self.cfg.dataset_name,
        #         self.cfg.dataset_config_name,
        #         cache_dir=self.cfg.cache_dir,
        #         data_dir=self.cfg.train_data_dir,
        #     )
        # else:
        #     data_files = {}
        #     if self.cfg.train_data_dir is not None:
        #         data_files["train"] = os.path.join(self.cfg.train_data_dir, "**")
        #     dataset = load_dataset(
        #         "imagefolder",
        #         data_files=data_files,
        #         cache_dir=self.cfg.cache_dir,
        #     )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        # column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        # dataset_columns = DATASET_NAME_MAPPING.get(self.cfg.dataset_name, None)
        # if self.cfg.image_column is None:
        #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        # else:
        #     image_column = self.cfg.image_column
        #     if image_column not in column_names:
        #         raise ValueError(
        #             f"--image_column' value '{self.cfg.image_column}' needs to be one of: {', '.join(column_names)}"
        #         )
        # if self.cfg.caption_column is None:
        #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        # else:
        #     caption_column = self.cfg.caption_column
        #     if caption_column not in column_names:
        #         raise ValueError(
        #             f"--caption_column' value '{self.cfg.caption_column}' needs to be one of: {', '.join(column_names)}"
        #         )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(caption: str, repeat: int):
            # captions = []
            # for caption in examples[caption_column]:
            #     if isinstance(caption, str):
            #         captions.append(caption)
            #     elif isinstance(caption, (list, np.ndarray)):
            #         # take a random caption if there are multiple
            #         captions.append(random.choice(caption) if is_train else caption[0])
            #     else:
            #         raise ValueError(
            #             f"Caption column `{caption_column}` should contain either strings or lists of strings."
            #         )
            inputs = tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids.repeat(repeat, 1)

        # Preprocessing the datasets.
        # train_transforms = transforms.Compose(
        #     [
        #         # transforms.Resize(self.cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        #         # transforms.CenterCrop(self.cfg.resolution) if self.cfg.center_crop else transforms.RandomCrop(self.cfg.resolution),
        #         # transforms.RandomHorizontalFlip() if self.cfg.random_flip else transforms.Lambda(lambda x: x),
        #         transforms.Normalize([127.5], [127.5])
        #     ]
        # )

        train_transforms = hydra.utils.instantiate(self.cfg.data_transforms)

        # def preprocess_train(examples):
        #     images = [image.convert("RGB") for image in examples[image_column]]
        #     examples["pixel_values"] = [train_transforms(image) for image in images]
        #     examples["input_ids"] = tokenize_captions(examples)
        #     return examples

        # with accelerator.main_process_first():
        # if self.cfg.max_train_samples is not None:
        #     dataset["train"] = dataset["train"].shuffle(seed=self.cfg.seed).select(range(self.cfg.max_train_samples))
        # # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train)

        # def collate_fn(examples):
        #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
        #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        #     input_ids = torch.stack([example["input_ids"] for example in examples])
        #     return {"pixel_values": pixel_values, "input_ids": input_ids}

        # DataLoaders creation:
        # train_dataset_all = SDData(
        #     IMG_DIR, LABELS_PATH, "nondeletion", transform=train_transforms
        # )
        # breakpoint()
        train_dataset_all = hydra.utils.instantiate(self.cfg.all_data, img_dir=self.cfg.data_files.img_dir, labels_fpath=self.cfg.data_files.labels_path, transform=train_transforms)
        train_dataset_memorized = hydra.utils.instantiate(self.cfg.memorized_data, img_dir=self.cfg.data_files.img_dir, labels_fpath=self.cfg.data_files.labels_path, transform=train_transforms)
        # train_dataset_memorized = SDData(
        #     IMG_DIR, LABELS_PATH, "deletion", transform=train_transforms
        # )

        infinite_sampler_all = InfiniteSampler(train_dataset_all)
        infinite_sampler_memorized = InfiniteSampler(train_dataset_memorized)

        train_dataloader_all = torch.utils.data.DataLoader(
            train_dataset_all,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            sampler=infinite_sampler_all,
        )
        data_all_iterator = iter(train_dataloader_all)

        train_dataloader_memorized = torch.utils.data.DataLoader(
            train_dataset_memorized,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            sampler=infinite_sampler_memorized,
        )
        data_deletion_iterator = iter(train_dataloader_memorized)
        self.cfg.imgs_per_gradient = self.cfg.train_batch_size * accelerator.num_processes * self.cfg.gradient_accumulation_steps

        # Scheduler and math around the number of training steps.
        # overrode_max_train_steps = False
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.cfg.gradient_accumulation_steps)
        # if self.cfg.max_train_steps is None:
        #     self.cfg.max_train_steps = self.cfg.num_train_epochs * num_update_steps_per_epoch
        #     overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=self.cfg.training_steps
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, data_all_iterator, data_deletion_iterator, lr_scheduler = (
            accelerator.prepare(
                unet, optimizer, data_all_iterator, data_deletion_iterator, lr_scheduler
            )
        )

        if self.cfg.use_ema:
            ema_unet.to(accelerator.device)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.cfg.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.cfg.mixed_precision = accelerator.mixed_precision

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        # if overrode_max_train_steps:
        #     self.cfg.max_train_steps = self.cfg.num_train_epochs * num_update_steps_per_epoch
        # # Afterwards we recalculate our number of training epochs
        # self.cfg.num_train_epochs = math.ceil(self.cfg.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        
        if accelerator.is_main_process:
            wandb_conf = OmegaConf.to_container(self.cfg) # fully converts to dict (even nested keys)
            accelerator.init_trackers(project_name=self.cfg.project_name, config=wandb_conf)

        # Function for unwrapping if model was compiled with `torch.compile`.
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        # Train!

        logger.info("***** Running training *****")
        logger.info(f"  Num total examples = {len(train_dataset_all)}")
        logger.info(f"  Num memorized examples = {len(train_dataset_memorized)}")
        logger.info(f"  Num training steps = {self.cfg.training_steps}")
        logger.info(f"  Batch size = {self.cfg.train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        # logger.info(f"  Total optimization steps = {self.cfg.max_train_steps}")
        # global_step = 0
        # first_epoch = 0
        img_count = 0

        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint != "latest":
                path = os.path.basename(self.cfg.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.cfg.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self.cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.cfg.resume_from_checkpoint = None
                # kimg_count = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self.cfg.output_dir, path))
                img_count = int(path.split("-")[1])
                # initial_global_step = global_step
                # first_epoch = global_step // num_update_steps_per_epoch

        progress_bar = tqdm(
            range(0, self.cfg.training_steps * self.cfg.imgs_per_gradient),
            initial=img_count,
            desc="Images",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def log_with_ema(
            vae,
            text_encoder,
            tokenizer,
            unet,
            # args,
            accelerator,
            weight_dtype,
            img_count,
            ema_unet,
        ):
            if self.cfg.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            self.log_validation(
                vae,
                text_encoder,
                tokenizer,
                unet,
                accelerator,
                weight_dtype,
                img_count,
            )
            if self.cfg.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
        breakpoint()
        gamma = noise_scheduler.alphas_cumprod**0.5
        gamma = gamma.to(accelerator.device)

        sigma = (1 - noise_scheduler.alphas_cumprod) ** 0.5
        sigma = sigma.to(accelerator.device)

        loss_class = DDPMDeletionLoss(gamma=gamma, sigma=sigma)
        loss_fn = getattr(loss_class, self.cfg.deletion.loss_fn)
        # Log at step 0 before training starts
        # log_with_ema(
        #     vae,
        #     text_encoder,
        #     tokenizer,
        #     unet,
        #     # args,
        #     accelerator,
        #     weight_dtype,
        #     img_count,
        #     unet,
        # )
        if "superfactor" in self.cfg.deletion.loss_params:
            wandb.log({"superfactor": self.cfg.deletion.loss_params.superfactor}, step=img_count)

        accum_loss_x = {}
        accum_loss_a = {}
        while img_count < self.cfg.training_steps * self.cfg.imgs_per_gradient:
            # train_loss = 0.0
            # for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                all_images, all_labels = next(data_all_iterator)
                all_images = all_images.to(accelerator.device)
                all_labels = all_labels.to(accelerator.device)

                deletion_images, deletion_labels = next(data_deletion_iterator)
                deletion_images = deletion_images.to(accelerator.device)
                deletion_labels = deletion_labels.to(accelerator.device)

                # print(f"Images shape: {images.shape}")
                # print(f"Labels shape: {labels.shape}")
                all_latents = vae.encode(
                    all_images.to(weight_dtype)
                ).latent_dist.sample()
                # breakpoint()
                all_latents = all_latents * vae.config.scaling_factor

                deletion_latents = vae.encode(
                    deletion_images.to(weight_dtype)
                ).latent_dist.sample()
                deletion_latents = deletion_latents * vae.config.scaling_factor

                # print(f"Latents shape: {latents.shape}")
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(all_latents)
                if self.cfg.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += self.cfg.noise_offset * torch.randn(
                        (all_latents.shape[0], all_latents.shape[1], 1, 1),
                        device=all_latents.device,
                    )

                if self.cfg.input_perturbation:
                    new_noise = (
                        noise
                        + self.cfg.input_perturbation * torch.randn_like(noise)
                    )

                bsz = all_latents.shape[0]
                # Sample a random timestep for each image
                if self.cfg.deletion.timestep_del_window is None and self.cfg.deletion.loss_fn == "modified_noise_obj":
                    raise ValueError('Timestep deletion window length must be provided for the modified_noise_obj loss function.')
                start_timestep = noise_scheduler.config.num_train_timesteps - self.cfg.deletion.timestep_del_window if self.cfg.deletion.loss_fn == "modified_noise_obj" else 0
                timesteps = torch.randint(
                    999,#start_timestep,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=all_latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if self.cfg.input_perturbation:
                    all_noisy_latents = noise_scheduler.add_noise(
                        all_latents, new_noise, timesteps
                    )
                    deletion_noisy_latents = noise_scheduler.add_noise(
                        deletion_latents, new_noise, timesteps
                    )
                else:
                    all_noisy_latents = noise_scheduler.add_noise(
                        all_latents, noise, timesteps
                    )
                    deletion_noisy_latents = noise_scheduler.add_noise(
                        deletion_latents, noise, timesteps
                    )

                # Get the text embedding for conditioning
                if self.cfg.using_augmented_prompt:
                    encoder_hidden_states = torch.load(self.cfg.validation_prompts[0])
                else:
                    # Tokenize caption
                    input_ids = tokenize_captions(self.cfg.validation_prompts[0], all_latents.shape[0]).to(
                        accelerator.device
                    )
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if self.cfg.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=self.cfg.prediction_type
                    )

                # if noise_scheduler.config.prediction_type == "epsilon":
                #     target = all_noise
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(all_latents, all_noise, timesteps)
                # else:
                #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                if self.cfg.snr_gamma is None:
                    # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    all_samples_dict = {
                        "og_latents": all_latents,
                        "noisy_latents": all_noisy_latents
                    }

                    deletion_samples_dict = {
                        "og_latents": deletion_latents,
                        "noisy_latents": deletion_noisy_latents
                    }

                    conditioning = {
                        'encoder_hidden_states': encoder_hidden_states
                    }
                    items = loss_fn(
                        unet=unet,
                        timesteps=timesteps,
                        noise=noise,
                        conditioning=conditioning,
                        all_samples_dict=all_samples_dict,
                        deletion_samples_dict=deletion_samples_dict,
                        **self.cfg.deletion.loss_params
                    )
                    continue
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

                    wandb.log(batch_stats, step=img_count)
                # else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                # snr = compute_snr(noise_scheduler, timesteps)
                # if noise_scheduler.config.prediction_type == "v_prediction":
                #     # Velocity objective requires that we add one to SNR values before we divide by them.
                #     snr = snr + 1
                # mse_loss_weights = (
                #     torch.stack([snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                # )

                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                # loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                # loss = loss.mean()

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
                            scaling_factor = self.cfg.deletion.eta - sum(torch.sum(accum_loss_x[name] * accum_loss_a[name]) for name, _ in unet.named_parameters()) / (total_l2_norm_a ** 2)
                            scaling_factor = -max(scaling_factor, 0)
                        else:
                            # gradient norm fixing 
                            scaling_factor = self.cfg.deletion.scaling_norm / total_l2_norm_a # if total_l2_norm_a > self.cfg.deletion.loss_params.clipping_norm else 1
                        
                        wandb.log({'gradient/norm_loss_x': total_l2_norm_x, 'gradient/norm_loss_a': total_l2_norm_a, 'gradient/scaling_factor': scaling_factor}, step=img_count + self.cfg.train_batch_size)
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

                # Update image bar
                progress_bar.update(self.cfg.train_batch_size)
                img_count += self.cfg.train_batch_size

                if self.cfg.checkpointing_steps and img_count % (self.cfg.checkpointing_steps * self.cfg.imgs_per_gradient) == 0:
                    if accelerator.is_main_process:
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
                                    len(checkpoints) - self.cfg.checkpoints_total_limit + 1
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
                            self.cfg.output_dir, f"checkpoint-{img_count}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Checks if the accelerator has performed an optimization step behind the scenes
                # Occurs every time gradients accumulate and a step is done
                if accelerator.sync_gradients:
                    if self.cfg.use_ema:
                        ema_unet.step(unet.parameters())

                    torch.cuda.empty_cache()
                    log_with_ema(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        # args,
                        accelerator,
                        weight_dtype,
                        img_count,
                        unet,
                    )

                    if "superfactor" in self.cfg.deletion.loss_params:
                        decay = self.cfg.deletion.superfactor_decay
                        if decay is not None:
                            self.cfg.deletion.loss_params.superfactor *= self.cfg.deletion.superfactor_decay
                        
                        wandb.log({
                                    "superfactor": self.cfg.deletion.loss_params.superfactor,
                                  }, step=img_count)

                    # wandb.log({"train_loss": train_loss}, step=img_count)
                    # train_loss = 0.0
                    # self.cfg.superfactor *= self.cfg.superfactor_decay
                    # wandb.log({"superfactor": self.cfg.superfactor}, step=img_count)
                    # wandb.log(
                    #     {"superfactor_decay": self.cfg.superfactor_decay}, step=img_count
                    # )
                
                latest_lr = lr_scheduler.get_last_lr()[0]
                logs = {
                    "lr": latest_lr
                }
                if loss is not None:
                    logs["loss"] = loss.detach().item()
                accelerator.log(logs, step=img_count)
                progress_bar.set_postfix(**logs)

            # if accelerator.is_main_process:
            #     if self.cfg.validation_prompts is not None and tick_count % self.cfg.validation_ticks == 0:

        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        #     unet = unwrap_model(unet)
        #     if self.cfg.use_ema:
        #         ema_unet.copy_to(unet.parameters())

        #     pipeline = StableDiffusionPipeline.from_pretrained(
        #         self.cfg.pretrained_model_name_or_path,
        #         text_encoder=text_encoder,
        #         vae=vae,
        #         unet=unet,
        #         revision=self.cfg.revision,
        #         variant=self.cfg.ema_variant,
        #         safety_checker=None
        #     )
        #     pipeline.save_pretrained(self.cfg.output_dir)

        #     # Run a final round of inference.
        #     all_images = []
        #     if self.cfg.validation_prompts is not None:
        #         logger.info("Running inference for collecting generated images...")
        #         pipeline = pipeline.to(accelerator.device)
        #         pipeline.torch_dtype = weight_dtype
        #         pipeline.set_progress_bar_config(disable=True)

        #         if self.cfg.enable_xformers_memory_efficient_attention:
        #             pipeline.enable_xformers_memory_efficient_attention()

        #         if self.cfg.seed is None:
        #             generator = None
        #         else:
        #             generator = torch.Generator(device=accelerator.device).manual_seed(
        #                 self.cfg.seed
        #             )

        #         # generator = None
        #         for i in range(len(self.cfg.validation_prompts)):
        #             with torch.autocast("cuda"):
        #                 image = pipeline(
        #                     self.cfg.validation_prompts[i],
        #                     # num_inference_steps=20,
        #                     generator=generator,
        #                 ).images[0]
        #             all_images.append(image)

        #     if self.cfg.push_to_hub:
        #         self.save_model_card(repo_id, all_images, repo_folder=self.cfg.output_dir)
        #         upload_folder(
        #             repo_id=repo_id,
        #             folder_path=self.cfg.output_dir,
        #             commit_message="End of training",
        #             ignore_patterns=["step_*", "epoch_*"],
        #         )

        # The end
        progress_bar.close()
        accelerator.end_training()