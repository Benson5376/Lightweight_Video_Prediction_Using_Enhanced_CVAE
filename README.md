# Lightweight_Video_Prediction_Using_Enhanced_CVAE

---

## Introduction

This project implements a Conditional Variational Autoencoder (CVAE) for video prediction. The CVAE leverages past video frames and conditions (action and end-effector position) to predict future frames. It includes custom modules for data loading, training with KL annealing, teacher forcing, and a reparameterization trick. The model is trained and evaluated on the BAIR Robot Pushing dataset.  
![Ground Truth](https://github.com/Benson5376/Lightweight_Video_Prediction_Using_Enhanced_CVAE/blob/main/resourses/ground_truth.gif)
![Prediction](https://github.com/Benson5376/Lightweight_Video_Prediction_Using_Enhanced_CVAE/blob/main/resourses/prediction.gif)
![image](https://github.com/Benson5376/Lightweight_Video_Prediction_Using_Enhanced_CVAE/blob/main/resourses/gt_12frame.PNG)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Dependencies](#dependencies)
5. [Configuration](#configuration)
6. [Training Details](#training-details)
7. [Evaluation](#evaluation)
8. [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   - Download the BAIR Robot Pushing dataset.
   - Extract the dataset and set the path in `--data_root` argument during execution.

2. **Train the model:**
   ```bash
   python train_fixed_prior.py --data_root ./path_to_dataset --log_dir ./logs --niter 40
   ```

3. **Evaluate the model:**
   - Use the `validate_loader` in the training script to compute PSNR or save predictions for visualization.

4. **Generate videos or GIFs for predictions:**
   - Modify the evaluation section to save predicted sequences as video or GIF files for visualization.

## Features

- **Conditional Variational Autoencoder (CVAE):**
  - Encoder-decoder structure with latent space sampling using a reparameterization trick.
  - Conditional input for action and end-effector positions.
  - VAE code reference: https://github.com/pytorch/examples/tree/main/vae  
- **KL Annealing:**
  - Supports monotonic and cyclical KL annealing schedules.

- **Teacher Forcing:**
  - Configurable teacher forcing ratio with scheduled decay.

- **Metrics:**
  - Implements PSNR, SSIM, and MSE metrics for evaluation.

- **Visualization:**
  - Capable of generating visual predictions as videos or GIFs.

## Dependencies

- Python 3.8+
- PyTorch 1.8+
- torchvision
- NumPy
- scikit-image
- Matplotlib
- tqdm

## Configuration

All configurations can be set via command-line arguments. Key arguments include:

- `--lr`: Learning rate (default: 0.002).
- `--batch_size`: Batch size for training (default: 16).
- `--n_past`: Number of past frames to condition on (default: 2).
- `--n_future`: Number of future frames to predict (default: 10).
- `--log_dir`: Directory to save logs and models.
- `--tfr`: Teacher forcing ratio (default: 1.0).
- `--kl_anneal_cyclical`: Use cyclical KL annealing (default: True).

See `train_fixed_prior.py` for a full list of configurable options.

## Training Details

- **Loss Function:**
  - Combines frame reconstruction loss (MSE) and KL divergence loss with a weight (`beta`) for balancing.
- **Optimization:**
  - Adam optimizer with configurable learning rate and momentum parameters.
- **KL Annealing:**
  - Both monotonic and cyclical annealing methods are supported.

## Evaluation

- **Metrics:**
  - PSNR: Measures perceptual quality of predicted frames.
  - SSIM: Evaluates structural similarity between frames.
  - MSE: Quantifies pixel-level differences.

- **Evaluation Procedure:**
  - The `finn_eval_seq` function computes metrics over the test dataset.
  - Visualize predictions by saving output sequences as images or videos.


## License

This project is licensed under the [MIT License](LICENSE).

---

