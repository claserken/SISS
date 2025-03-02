# Data Unlearning in Diffusion Models (ICLR 2025)

This codebase implements various fine-tuning methods for forgetting specific training datapoints (data unlearning) from diffusion models without having to retrain from scratch. Our main contribution is SISS (Subtracted Importance Sampled Scores), a principled and efficient data unlearning objective. More details can be found in our paper at https://openreview.net/pdf?id=SuHScQv5gP.

## Setup

1. Clone the repo.
    ```sh
    git clone https://github.com/claserken/SISS.git
    cd SISS
    ```

2. Create the conda environment.
    ```sh
    conda env create -f environment.yml
    ```

3. Download pretrained checkpoints and datasets.
    ```sh
    curl -L -o checkpoints.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-checkpoints
    
    curl -L -o data/datasets.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-datasets
    ```

4. After creating and activating the environment, you can run our wandb-compatible experiments with Hydra as follows:
    ```sh
    python main.py --config-name=[delete_celeb, delete_sd, delete_tshirt]
    ```

## Hydra Config Details

Each Hydra config (``delete_celeb.yaml, delete_sd.yaml, delete_tshirt.yaml``) defines:
- **Model & Dataset**: Paths to checkpoints/data.
- **Unlearning Method**: Which loss function to apply.
- **Training Hyperparameters**: Batch size, learning rate, etc.
- **Metrics to compute during fine-tuning**.

### Unlearning Methods (``losses/ddpm_deletion_loss.py``)

1. **SISS (importance_sampling_with_mixture)**
   - Uses a defensive mixture for importance sampling (one forward pass per batch).
   - **Key hyperparams**:
     - `lambd ∈ [0, 1]`: Balances sampling between kept data and unlearn set.
     - `scaling_norm`: Clips NegGrad term's gradient to this norm. Recommended to be tuned to 10% of naive deletion term's gradient norm.

2. **SISS (No IS) (double_forward_with_neg_del)**
   - Same concept but no importance sampling.
   - Requires two forward passes per batch (keep + forget).
   - **Key hyperparams**:
     - `scaling_norm`: Clips NegGrad term's gradient to this norm. Recommended to be tuned to 10% of naive deletion term's gradient norm.

3. **EraseDiff (erasediff)**
   - Forces the model to predict random noise for the unlearn set.
   - **Key hyperparams**:
     - `eta`: EraseDiff learning rate.

4. **Naive Deletion (naive_del)**
   - Fine-tune only on kept data (X \ A).
   - **Key hyperparams**:
     - None

5. **NegGrad (simple_neg_del)**
   - Gradient ascent on unlearn set.
   - **Key hyperparams**:
     - `superfactor`: Scales learning rate.


### Monitored Metrics

- **FID**: Measures image quality for CelebA.
- **Inception Score (IS)**: Measures quality for MNIST using a digit classifier.
- **SSCD**: Self-Supervised Copy Detection (https://arxiv.org/pdf/2202.10261) for image similarity to quantify unlearning. 
- **CLIP-IQA**: CLIP-based image quality assessment for Stable Diffusion.
- **Negative Log Likelihood (NLL)**: Likelihood of the unlearned data to quantify unlearning. Calculated according to exact likelihood formula in https://arxiv.org/pdf/2011.13456.
