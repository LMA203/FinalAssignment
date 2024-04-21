Codalab username: LucManders
TU/e email: l.manders@student.tue.nl

## Getting Started

### Dependencies

This repo is based on the following project/packages:
- Pytorch
- Albumentations

### Installing

To use this project you need to clone the repository. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/LMA203/FinalAssignment.git
```

After cloning the repository, navigate to the project directory:

```bash
cd FinalAssignment
```

- Step 1: Create virtual environment:
  
```
conda create -n experiments python=3.12
conda activate experiments
```

- Step 2: run this command to install the nessecery packages:

```
pip install -r requirements.txt
```

### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.

  
- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  

- **model.py:** Defines the neural network architecture.
  
- **train.py:** Contains the code for training the neural network on snellius.

- **train_locally.ipynb:** Contains the code for training the neural network local.

- **image_augmentation.ipynb** Contains the code to generate the image augmentation.

