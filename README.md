# MBGd-YOLO

This is an implementation of the Mosquitoes Breeding Grounds Detection (MBGd) using YOLOv8. The objective is to analyze the performance of this neural network in an attempt to improve upon the results of previous experiments.

The baseline for this work is the code and workflow implemented at https://github.com/misabellerv/MBGd.git. That project introduced a novel methodology and utilized Faster-RCNN for the detection task.

The present work is part of a long-running research project involving several undergraduate, Master’s, and Ph.D. students, alongside esteemed professors from the Signal, Multimedia, and Telecommunications Lab (SMT) at the Federal University of Rio de Janeiro (UFRJ). This repository contains code developed by me as an undergraduate research student, with help from many colleagues.

The code presented here was used in [@felipe-brrt](https://github.com/felipe-brrt) Bachelor's Thesis to evaluate the advantages of YOLOv8 over Faster-RCNN. Another significant contribution from the group was a new approach to processing the dataset. High-resolution images are now tiled into smaller ones, preventing the neural network from having to downscale them. As a result, the objects of interest are no longer reduced in size, making their identification easier.

This repository implements the YOLOv8 model and introduces a key methodological innovation: nested k-fold cross-validation. This approach provides a more rigorous and unbiased evaluation of the model's performance, ensuring our results are more realistic and generalizable to unseen data.

## Installation

This project was developed and tested using the following specifications. To ensure the full reproducibility of the results, it is strongly recommended to use the same version of Python and the libraries listed in the `requirements.txt` file. However, the code does not rely on specific features that are prone to breaking in newer versions. The project is therefore likely to work with newer versions of Python (e.g., 3.9+) and the listed libraries. If you choose to use newer versions, please be aware that unexpected behavior may occur, or that minor code adjustments may be required.

### Requirements
- CUDA 11.8+
- Python 3.8+
- Conda
- Linux system to run bash script files

### Installation Steps
Check if you have the necessary requirements and run the following command:
1.  **Create and activate the Conda environment:**

    ```bash
    conda create -n mbgd-yolo python=3.8
    conda activate mbgd-yolo
    ```

2.  **Install PyTorch (Important!)**

    You must install the PyTorch version that matches your system's CUDA driver. Check your CUDA version by running `nvidia-smi` in the terminal.

    * **For CUDA 11.8 (used in this project):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

    * **For CUDA 12.1 (for newer GPUs):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    * **For CPU-only installation:**
        ```bash
        pip install torch torchvision torchaudio
        ```
    > For other versions, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/).

3.  **Install remaining project dependencies:**

    Once PyTorch is correctly installed, install the other packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Configurations File

### Dataset Configs
These files follow the official **YOLO documentation format**. A unique dataset configuration (`.yaml`) file is required for each data split. Since this repository contains two different training methodologies, please check the requirements for the script you intend to use:

* **For the Nested K-Fold Script:** This methodology uses a 5x4 nested cross-validation setup, which results in 20 distinct training/validation combinations. Therefore, it requires **20 separate configuration files** to run correctly.

* **For the Simple K-Fold Script (Legacy):** This script runs a standard 5-fold cross-validation and requires only **5 configuration files**, one for each fold.

### hparams.yaml
This file contains the **hyperparameters** for the YOLO model, such as data augmentation, image size, and batch size. For a complete list of options, please visit the official [Ultralytics documentation](https://docs.ultralytics.com/).

### config.yaml
This is the main configuration file for the **k-fold methodology**.

* `OUTPUT_DIR`: Defines the directory where all outputs will be saved. Inside this path, a new folder will be created with the name specified in `EXPERIMENT_NAME`.

* `EPOCHS`:Sets the maximum number of iterations for training.

* `PATIENCE`: Defines the number of consecutive epochs with no significant improvement to wait before stopping the training early. For example, if `PATIENCE` is set to 20, the training will halt after 20 epochs without improvement.

* `PRE_TRAINED_MODELFeel`: Feel free to change the pre-trained model for a new experiment. However, be aware that this may require additional changes in other files.

* `NUM_FOLDS`: Sets the total number of runs to perform. Each run consists of a training and a validation phase.

* `OBJECT` Specifies the object class you want to detect. Currently, only one class is supported at a time.

* `DATASET`: **Pay attention** to this variable. Ensure that it points to the correct path of the dataset's configuration `.yaml` file; otherwise, the code will fail to run.

### nested_config.yaml
This is the main configuration file for the **nested k-fold methodology**. Its structure is nearly identical to `config.yaml`, sharing most of the same parameters.

The primary difference is in how the dataset configuration paths are defined, which is structured to support the multiple data splits required by the nested k-fold process.

## Running the Project
This project features two distinct scripts. The first, `train_val.sh`, follows the original methodology, performing only training and validation. The second script, `nested_workflow.sh`, introduces an improved approach with nested k-fold cross-validation.

### 1. Make the Script Executable
First, make sure the bash scripts have the correct permissions. You only need to do this once.
```bash
chmod +x <full_path_to_train_val.sh>
chmod +x <full_path_to_nested_workflow.sh>
```

### 2. (Optional) Fix Line Endings
If you have cloned or edited the project on a Windows machine, you might encounter issues with line endings when running the script on Linux. To fix this, run the following command:
```bash
sed -i 's/\r$//' scripts/train_val.sh
sed -i 's/\r$//' scripts/nested_workflow.sh
```

### 3. Start the Training
Now you can run the code from the main directory. To run the training process in the background (so it continues even if you close the terminal), use the `nohup` command.
```bash
#if you want to run train and validation:
nohup bash scripts/train_val.sh > log_out.log 2>&1 &

#if you want to do train, validation and test:
nohup bash scripts/nested_workflow.sh > log_out.log 2>&1 &
```
And the outputs will be saved at `log_out.log` file in real time. You can monitore it.

### 4. Stop the Training Process
If you need to interrupt the workflow, just run the following command.
```
killall -9 python
```
Warning: This command will forcefully terminate all Python processes running under your user account, not just the ones from this script. Use it with caution.
It may be necessary to run this command multiple times because the `training.sh` script launches different Python processes for the training and validation of each fold.
