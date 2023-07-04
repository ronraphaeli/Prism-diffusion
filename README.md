# Prism-diffusion

![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/5130e9f2-6207-48a7-9435-e1a9530210e2)



This repository contains the implementation of the Prism method for controling colors in diffusion generated images, using a conditional LoRA, along with the necessary classes and dependencies.This repository provides the scripts and environment setup instructions to run the method effectively.


below are some examples of generations using the same prompt and seed, but with a change of color palette.
| Red palette | Blue palette |
|-------------|---------|
|![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/b74df56c-1c75-4c2d-bb3b-b3ab9409a388)|![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/38c39325-f604-4ce4-8b6a-d779ab66cee2)|
|![prompt_a beautiful image of a colorful , vibrant flower_seed_1337red](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/326d09fb-8c98-4ea1-b601-1f2c4f9144b3)|![prompt_a beautiful image of a colorful , vibrant flower_seed_1337blue](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/052862e2-4ad6-4e8a-ba36-75db807e8577) |
|![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/40926b08-c19e-499c-8b98-1944974b12d8)| ![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/cf48f584-dd8d-4f52-943c-7a93fe0727ab)|
|![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/f2105f98-bcd1-4bb1-af7a-f29f60a0b341)| ![image](https://github.com/ronraphaeli/Prism-diffusion/assets/102682845/3a6a8146-7979-4bff-895b-e4b79856530d)|


## Repository Structure

The repository is structured as follows:

- `prism_training.ipynb`: This Jupyter Notebook script contains the implementation of the Prism training method. It is the main script that needs to be executed to train a Prism LoRA. Before running the script, certain hyperparameters and logging information need to be filled in.

- `prism.yml`: This file specifies the Conda environment required to run the code in `prism_training.ipynb`. It contains a list of dependencies and their versions to ensure a consistent and reproducible environment.

- `prism_classes.py`: This Python file contains the necessary classes and functions used in the Prism training script. These classes implement the core functionality of the Prism method and provide the required functionality for training the deep learning model.

## Getting Started

To run the Prism training code and train a deep learning model using Prism, follow the instructions below:

### Environment Setup

0. Make sure you have Conda installed on your system. If not, follow the instructions for your operating system to install Conda from the official website.

1. Fork the repository by clicking the "Fork" button on the top-right corner of this repository page. This will create a copy of the repository in your GitHub account.

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

3. Execute the cells in the Jupyter Notebook sequentially to run the Prism training script. The script will train the model using the Prism method based on the provided configurations.

4. Monitor the training progress and logging information in WandB. The training script will log the necessary information according to the specified configurations.

## machine

in order to run this method, using a GPU with at least 20 GB RAM is a must

## based on

this notebook code is based on the following code from the huggingface diffusers code:
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
with changes relavent to our specific method and architecture, CoLoRA.

## contact info

for any questions regarding the project, you can contact me at ronraphaeli at technion.ac.il
.
