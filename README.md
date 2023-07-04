# Prism-diffusion

This repository contains the implementation of the Prism method for deep learning, along with the necessary classes and dependencies. The Prism method is a novel approach for training deep learning models, and this repository provides the scripts and environment setup instructions to run the method effectively.

## Repository Structure

The repository is structured as follows:

- `prism_training.ipynb`: This Jupyter Notebook script contains the implementation of the Prism training method. It is the main script that needs to be executed to train a deep learning model using Prism. Before running the script, certain hyperparameters and logging information need to be filled in.

- `prism.yml`: This file specifies the Conda environment required to run the code in `prism_training.ipynb`. It contains a list of dependencies and their versions to ensure a consistent and reproducible environment.

- `prism_classes.py`: This Python file contains the necessary classes and functions used in the Prism training script. These classes implement the core functionality of the Prism method and provide the required functionality for training the deep learning model.

## Getting Started

To run the Prism training code and train a deep learning model using Prism, follow the instructions below:

### Environment Setup

1. Make sure you have Conda installed on your system. If not, follow the instructions for your operating system to install Conda from the official website.

2. Clone this Git repository to your local machine using the following command:
   ```
   git clone https://github.com/your-username/Prism-diffusion.git
   ```
   Replace `your-username` with your GitHub username.

3. Navigate to the cloned repository:
   ```
   cd Prism-diffusion
   ```

4. Create a Conda environment with the required dependencies by running the following command:
   ```
   conda env create -f prism.yml
   ```

5. Activate the created Conda environment:
   ```
   conda activate prism
   ```

### Running the Prism Training Script

1. Open the `prism_training.ipynb` Jupyter Notebook in a Jupyter-compatible environment, such as Jupyter Notebook or JupyterLab.

2. Before running the script, fill in the necessary hyperparameters and logging information in the designated code cell. Modify the following variables according to your requirements:

   - `logging_name`: The name to initialize the WandB tracker.
   - `args.validation_prompt`: The prompt for validation (e.g., "a beautiful image of a flower").
   - `logging_training`: Whether to log training information.
   - `train_check_every_global_steps`: Log the train information every x steps.
   - `valid_check_every_global_steps`: Log the validation information every x steps.
   - `args.checkpointing_steps`: Save the checkpoint every x steps.
   - `args.learning_rate`: Learning rate for the training process.
   - `args.max_train_steps`: Maximum number of training steps.
   - `image_to_get_palette_from`: Path to the image to get the palette from.
   - `coco_root`: Path to the COCO dataset root directory.
   - `coco_annotation_path`: Path to the COCO annotation file.
   - `args.report_to`: Specify the reporting platform ("wandb" or None).
   - `path_to_weights`: Path to pre-trained weights if you want to load existing weights.

3. Execute the cells in the Jupyter Notebook sequentially to run the Prism training script. The script will train the deep learning model using the Prism method based on the provided configurations.

4. Monitor the training progress and logging information in the Jupyter Notebook output. The training script will log the necessary information according to the specified configurations.

## Contributing

If you wish to contribute to this project, you can follow

.
