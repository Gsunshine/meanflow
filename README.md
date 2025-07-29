# Mean Flows for One-step Generative Modeling

<div align="center">
    <img src="./assets/velocity.png" width="1000" alt="1-NFE sample with MeanFlow.">
</div>

<div align="center">
    <img src="./assets/samples.png" width="1000" alt="1-NFE sample with MeanFlow.">
</div>

This is the official JAX implementation for the paper [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447). This code is written and tested on TPUs.

## Initialization

Run `install.sh` to install the dependencies (JAX+TPUs).

## Inference

You can quickly verify your setup with a provided MF-B/4 checkpoint. 

#### Sanity Check (MF-B/4)

1. **Download the checkpoint and FID stats:**  
   [MF-B/4 Checkpoint (Google Drive)](https://drive.google.com/file/d/19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu/view?usp=sharing)

   [FID stats (Google Drive)](https://drive.google.com/file/d/1sESI-bE4SpB0noJ6VQ_OE6XAI_o4c-qK/view?usp=sharing)

   **Note:** The FID stats followed the [ADM GitHub repository](https://github.com/openai/guided-diffusion).

2. **Unzip the checkpoint:**
   ```bash
   unzip <downloaded_file.zip> -d <your_ckpt_dir>
   ```
   Replace `<downloaded_file.zip>` with the downloaded file name, and `<your_ckpt_dir>` with your target checkpoint directory.

3. **Set up the config:**
   - Add `load_from` to `configs/run_b4.yml` and set it to the path of `<your_ckpt_dir>`.
   - Set `fid.cache_ref` to the path of the downloaded FID stats file.
   - Set `eval_only` to `True` in the same config.

4. **Launch evaluation:**
   ```bash
   bash scripts/launch.sh EVAL_JOB_NAME
   ```
   The expected FID is **11.4** for this checkpoint.

## Data Preparation

Before training, you need to prepare the ImageNet dataset and compute latent representations:

#### 1. Download ImageNet

Download the ImageNet dataset and extract it to your desired location. The dataset should have the following structure:
```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

#### 2. Configure Data Paths

Update the data paths in `scripts/prepare_data.sh`:

```bash
IMAGENET_ROOT="YOUR_IMGNET_ROOT"
OUTPUT_DIR="YOUR_OUTPUT_DIR"
LOG_DIR="YOUR_LOG_DIR"
```

#### 3. Launch Data Preparation

Run the data preparation script to compute latent representations:

```bash
IMAGE_SIZE=256 COMPUTE_LATENT=True bash ./scripts/prepare_data.sh
```

**Parameters:**
- `IMAGE_SIZE`: Image size for processing (256, 512, or 1024). Latent sizes will be 32x32, 64x64, or 128x128 respectively.
- `COMPUTE_LATENT`: Whether to compute and save the latent dataset (True/False)
- `COMPUTE_FID`: Whether to compute FID statistics (True/False)

The script will:
- Encode ImageNet images to latent representations using a VAE model
- Save the latent dataset to `OUTPUT_DIR/`
- Compute FID statistics and save to `OUTPUT_DIR/imagenet_{IMAGE_SIZE}_fid_stats.npz`
- Log progress to `LOG_DIR/$USER/`

### Configuration Setup

After data preparation, you need to configure your FID cache reference in the config files:

#### 1. Update Config Files

Edit your config file (e.g., `configs/run_b4.yml`) and replace the placeholder values:

```yaml
dataset:
    root: YOUR_DATA_ROOT  # Path to your prepared latent dataset

fid:
    cache_ref: YOUR_FID_CACHE_REF  # Path to your FID statistics file
```

#### 2. Available Config Files

- `configs/run_b4.yml` - Configuration for MF-B/4 model training (recommended)
- `configs/default.py` - Default configuration (Python format, used as base)

**Configuration Hierarchy:**
The system uses a hierarchical approach where `run_b4.yml` overrides specific parameters from `default.py`. This allows you to customize only the parameters you need while keeping sensible defaults.

Make sure to update both the dataset root path and the FID cache reference path according to your data preparation output.

## Training

Run the following commands to launch training:
```bash
bash scripts/launch.sh JOB_NAME
```

**Note:** Update the environment variables in `scripts/launch.sh` before running:
- `DATA_ROOT`: Path to your prepared data directory
- `LOG_DIR`: Path where to save training logs

#### Config System

The training system uses two config files:

- **`configs/default.py`** - Base configuration with all default hyperparameters
- **`configs/run_b4.yml`** - Model-specific overrides for MF-B/4 training

The system merges these files, allowing you to customize only the parameters you need.

#### Customizing Training

To create a custom experiment:

1. **Create a new config file** (e.g., `configs/my_exp.yml`)
2. **Update the launch script** to use your config:
   ```bash
   # In launch.sh, change the config line to:
   --config=configs/load_config.py:my_exp
   ```

**Example custom config:**
```yaml
training:
    num_epochs: 80                  # Train for fewer epochs

method:
    guidance_eq: 'none'             # Disable guidance
```

#### Training Monitoring

During training, the code log training metrics to `LOG_DIR/$USER/$JOBNAME/`. You can use `tensorboard` to monitor the training progress.

```bash
tensorboard --logdir LOG_DIR --port 12666 
```

## Performance

The table below shows the generative performance under the model size of MF-B/4.

| Settings  | FID@80ep | FID@240ep |
|-----------|----------|-----------|
| guidance_eq=`none` | 61.09/60.75 | 48.16 |
| guidance_eq=`cfg`, $\omega=2.0$, $\kappa=0.0$ | 20.15/20.24 | 13.74 |
| guidance_eq=`cfg`, $\omega=1.0$, $\kappa=0.5$ | 19.15/18.70 | 11.35 |

**Note:** Numbers in FID@80ep are in format "reported in paper / this repo". 
The 2nd and 3nd row correspond to Table 1. (f) and Table 5, using the same effective guidance scale as $\omega/(1-\kappa)$.

## TODO

- [x] Dependencies and sanity check for GPU. (FID=11.29 with default training setting)

## License

This repo is under the MIT license. See [LICENSE](./LICENSE) for details.

## Bibtex

```bib
@article{meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```

## Contributors

This repository is a collaborative effort by Kaiming He, Runqian Wang, Qiao Sun, Zhicheng Jiang, Hanhong Zhao, Yiyang Lu, Xianbang Wang, and Zhengyang Geng, developed in support of several research projects. 

## Acknowledgement

We gratefully acknowledge the Google TPU Research Cloud (TRC) for granting TPU access.
We hope this work will serve as a useful resource for the open-source community.
