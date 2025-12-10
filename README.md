# Car Damage Detection with YOLOv11

This project develops and evaluates a YOLOv11-based object detection model to identify multiple types of vehicle damage. The complete workflow—from dataset preparation to model evaluation and final deployment—is implemented in the `notebooks/main.ipynb` notebook.

YOLOv11 is chosen due to its strong trade-off between accuracy and inference speed, along with its ease of use through the Ultralytics API.

---

## Dataset Overview

This project uses the **Car Damage Detection (CarDD)** dataset from Kaggle:

- Approximately **4,000 annotated images**
- **6 damage categories**:

  - dent
  - scratch
  - crack
  - glass_shatter
  - lamp_broken
  - tire_flat

- Provided in **COCO format** and converted to **YOLO format** during preprocessing

Dataset link:
[https://www.kaggle.com/datasets/issamjebnouni/cardd](https://www.kaggle.com/datasets/issamjebnouni/cardd)

The exact number of images may vary depending on the dataset version.
After preprocessing, the dataset structure used in this project is:

- Train: 2,816 images
- Validation: 810 images
- Test: 374 images

---

## Project Structure

```
project_root/
│
├── cardd.yaml                # YOLOv11 dataset configuration
├── deploy.py                 # CLI/Web inference script
├── requirements.txt          # Project dependencies
│
├── datasets/                 # Populated after running the notebook
│   ├── cardd_raw/            # Raw Kaggle COCO dataset
│   └── cardd/                # YOLO-formatted dataset
│
├── models/                   # Final exported model + metadata
│   ├── cardd_yolo11s_1024_final.pt
│   └── cardd_model_info.json
│
├── notebooks/
│   └── main.ipynb            # Full training and evaluation pipeline
│
└── runs/                     # YOLO training outputs
```

---

## Setup Instructions

Follow these steps to set up the environment.

### 1. Clone the Repository

```
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create and Activate a Python Virtual Environment

This project was developed using **Python 3.12**.

```
python3.12 -m venv .venv
```

Activate the environment:

- Windows:

  ```
  .\.venv\Scripts\activate
  ```

- macOS/Linux:

  ```
  source .venv/bin/activate
  ```

### 3. Install PyTorch with CUDA 12.1

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Project Dependencies

```
pip install -r requirements.txt
```

This project uses `ultralytics==8.3.0` for YOLOv11.

### 5. Configure Kaggle API Token

1. Visit: [https://www.kaggle.com/](https://www.kaggle.com/)<your-username>/account
2. Click **Create New API Token**
3. Place the downloaded `kaggle.json` file in:

- Windows:
  `C:\Users\<User>\.kaggle\kaggle.json`
- macOS/Linux:
  `~/.kaggle/kaggle.json`

The notebook will automatically detect it.

---

## Workflow: Training and Evaluation

All core steps of this project are executed in:

```
notebooks/main.ipynb
```

Running the notebook performs:

- Dataset download and preparation
- COCO → YOLO annotation conversion
- Four controlled YOLOv11 experiments
- Validation and test-set evaluation
- Visualizations (mAP curves, confusion matrix, per-class metrics)
- Model selection
- Export of the final production-ready model

### Steps to Run

1. Open the notebook located at:

   ```
   notebooks/main.ipynb
   ```

2. Run all cells sequentially from top to bottom.

### GPU Guidelines

These experiments were conducted on an NVIDIA RTX 4060.

- **640px models** (Exp1, Exp2): batch size ≈ 16
- **1024px models** (Exp3, Exp4): batch size ≈ 4
- Approximate training times:

  - 640px models: 20–30 minutes
  - 1024px models: 45–60 minutes

---

## Experiments

| Experiment | Model   | Resolution |
| ---------- | ------- | ---------- |
| Exp1       | YOLO11n | 640        |
| Exp2       | YOLO11s | 640        |
| Exp3       | YOLO11n | 1024       |
| Exp4       | YOLO11s | 1024       |

These experiments isolate the effect of model size and input resolution.

### Final Selected Model

**Exp4 – YOLO11s @ 1024px**

Reason:

- Highest **mAP50** among all experiments
- Strong per-class accuracy, especially on fine-grained damage categories
- Acceptable inference latency for offline or batch evaluation pipelines

Although Exp2 is significantly faster (~2.5×), Exp4 provides the best _overall detection accuracy_, which was prioritized for this project.

---

## Deployment

### Web Application (FastAPI + Uvicorn)

To run the web-based damage detection UI:

1. Ensure the final model exists:

```
models/cardd_yolo11s_1024_final.pt
```

2. (Optional) Configure **Groq API Key** to enable AI-generated damage summaries.
   Create `.env` in project root:

```
GROQ_API_KEY=your_api_key_here
```

3. Start the server:

```
uvicorn app.main:app --reload
```

4. Open in browser:

```
http://127.0.0.1:8000
```

### CLI Inference Script

```
python deploy.py --img-path "/path/to/image.jpg"
```

Saves annotated output to `deployment_output/`.

To run the webcam demo:

```
python deploy.py
```

Press `Q` to exit.

---

## Results Summary

| Model            | Resolution | mAP50     | mAP50-95 | Precision | Recall | Inference Time |
| ---------------- | ---------- | --------- | -------- | --------- | ------ | -------------- |
| **Exp4 (Final)** | 1024       | **0.752** | 0.586    | 0.781     | 0.701  | 10.07 ms       |
| Exp2             | 640        | 0.732     | 0.588    | 0.768     | 0.692  | **4.01 ms**    |

### Interpretation

- Exp4 provides the **highest detection accuracy**, especially important for subtle damage such as cracks and scratches.
- Exp2 is faster but slightly less accurate.
- For this project, accuracy is prioritized, making **Exp4** the recommended production model.

---

## Reproducibility

This repository contains everything required to reproduce the results:

- Full preprocessing pipeline
- Reproducible training scripts with fixed seeds
- Complete experiment logs in `/runs`
- Final model and metadata in `/models`
