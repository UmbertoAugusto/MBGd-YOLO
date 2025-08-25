# MBGd-YOLO

This is an implementation of the Mosquitoes Breeding Grounds Detection (MBGd) using YOLOv8. The objective is to analyze the performance of this neural network in an attempt to improve upon the results of previous experiments.

The baseline for this work is the code and workflow implemented at https://github.com/misabellerv/MBGd.git. That project introduced a novel methodology and utilized Faster-RCNN for the detection task.

The present work is part of a long-running research project involving several undergraduate, Master’s, and Ph.D. students, alongside esteemed professors from the Signal, Multimedia, and Telecommunications Lab (SMT) at the Federal University of Rio de Janeiro (UFRJ). This repository contains code developed by me as an undergraduate research student, with help from many colleagues.

The code presented here was used in [@felipe-brrt](https://github.com/felipe-brrt) Bachelor's Thesis to evaluate the advantages of YOLOv8 over Faster-RCNN. Another significant contribution from the group was a new approach to processing the dataset. High-resolution images are now tiled into smaller ones, preventing the neural network from having to downscale them. As a result, the objects of interest are no longer reduced in size, making their identification easier.

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
In progress...


## Running the Project

### 1. Make the Script Executable
First, make sure the bash script has the correct permissions. You only need to do this once.
```bash
chmod +x <full_path_to_training.sh>
```

### 2. (Optional) Fix Line Endings
If you have cloned or edited the project on a Windows machine, you might encounter issues with line endings when running the script on Linux. To fix this, run the following command:
```bash
sed -i 's/\r$//' scripts/training_mosquitoes.sh
```

### 3. Start the Training
Now you can run the code from the main directory. To run the training process in the background (so it continues even if you close the terminal), use the `nohup` command.
```bash
nohup bash scripts/training.sh > train+test.log 2>&1 &
```
And the outputs will be saved at `train+test.log` file in real time. You can monitore it.

### 4. Stop the Training Process
If you need to interrupt the workflow, just run the following command.
```
killall -9 python
```
Warning: This command will forcefully terminate all Python processes running under your user account, not just the ones from this script. Use it with caution.
It may be necessary to run this command multiple times because the `training.sh` script launches different Python processes for the training and validation of each fold.
