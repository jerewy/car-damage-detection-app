# Car Damage Detection with YOLOv8

This project develops and evaluates a YOLOv8-based object detection model to identify various types of damage on vehicles. The entire workflow, from data preparation to model evaluation and packaging, is documented in the `notebooks/main.ipynb` notebook.

## Project Structure

- `/.gitignore`: Specifies files and directories to be ignored by Git.
- `/cardd.yaml`: The dataset configuration file used by YOLOv8 for training.
- `/deploy.py`: A Python script to run inference on new images using the final trained model.
- `/requirements.txt`: A list of Python dependencies required for the project.
- `/datasets/`: This directory is initially empty and will be populated with the CarDD dataset after running the notebook.
- `/models/`: This directory is initially empty and will contain the final, packaged production model (`cardd_yolo11s_640_final.pt`) after the notebook is fully executed.
- `/notebooks/`: Contains the main Jupyter Notebook (`main.ipynb`) with the full experimentation and evaluation workflow.
- `/runs/`: This directory is initially empty and will contain the output of all training experiments, including weights, logs, and visualizations.

## Setup Instructions

Follow these steps to set up the project environment.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**2. Create and Activate a Python Virtual Environment**
This project was developed and tested with **Python 3.12**. It is highly recommended to use the same version. This command isolates the project dependencies from your system's Python installation.

```bash
# Create the environment with Python 3.12 (ensure it's on your system PATH)
python3.12 -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

**3. Install PyTorch with CUDA**
This project requires PyTorch compatible with CUDA 12.1. This version is suitable for modern NVIDIA GPUs like the RTX 30/40 series (including the RTX 4060 used for development).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Install Project Dependencies**
Install the remaining required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**5. Set Up Kaggle API Token**
To download the dataset, you need a Kaggle API token.

1.  Go to your Kaggle account page: `https://www.kaggle.com/<your-username>/account`.
2.  Click on "Create New API Token". This will download a `kaggle.json` file.
3.  Place this `kaggle.json` file in your user's home directory under a `.kaggle` folder (e.g., `C:\Users\<YourUser>\.kaggle\kaggle.json` on Windows or `~/.kaggle/json` on macOS/Linux). The notebook will automatically detect it.

## Workflow: Training and Evaluation

The core of this project is in the `notebooks/main.ipynb` notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open `main.ipynb`:** Navigate to the `notebooks/` directory and open `main.ipynb`.
3.  **Run the Cells:** Execute the notebook cells sequentially from top to bottom. The notebook will:
    - Download and prepare the CarDD dataset.
    - Run all four training and evaluation experiments.
    - Analyze the results and select the best model.
    - Package the final model and its metadata into the `/models` directory.
    - _Note: The training process will take a significant amount of time._

## Deployment: Running Inference

After you have run the `main.ipynb` notebook and the final model (`cardd_yolo11s_640_final.pt`) has been generated in the `/models` directory, you can use the `deploy.py` script to detect damages on your own images.

**Usage:**

```bash
python deploy.py --img-path "/path/to/your/image.jpg"
```

The script will load the final model, run inference on the specified image, and save an annotated version of the image in a `deployment_output/` directory.
