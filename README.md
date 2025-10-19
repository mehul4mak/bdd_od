# 🛣️ BDD Object Detection - Data Analysis & Modeling

## 📌 Project Overview

This project focuses on analyzing the [Berkeley DeepDrive (BDD100K)](https://bdd-data.berkeley.edu/) dataset for the task of **object detection**. The pipeline includes:

1. **Data Exploration & Visualization**
2. **Model Building & Training**
3. **Evaluation & Visualization**

> ⚠️ Note: Only the **object detection** part of the dataset is used (bounding box annotations for 10 classes such as traffic light, sign, person, car, etc.). Semantic segmentation data (drivable area, lane marking) is **not required**.

---

## 📂 Directory Structure

```
.
├── data/ # Placeholder for dataset (not included in repo)
├── notebooks/ # Jupyter notebooks for analysis and experimentation
├── src/ # Source code: data loaders, parsers, utilities
├── .Dockerfile # Dockerfile and config to run analysis in container
├── results/ # Visualizations and reports
├── .pylintrc # Pylint configuration file for code style
├── requirements.txt # Python dependencies
├── README.md # This file
└── analysis_report.md # Detailed data analysis documentation
```


---

## 1. 📊 Data Analysis 

### ✅ Objectives

- Parse and process the BDD100K detection dataset (images + labels).
- Analyze class distributions, train/val splits, and identify anomalies or patterns.
- Visualize statistics via plots or dashboards.
- Highlight unique or interesting samples.

### 🛠️ Implementation Details

- Parsers written to read image files and JSON annotations.
- Train/Val split statistics and distributions computed and visualized.
- Visualizations include bar plots, sample images, class heatmaps, etc.

📄 **See**: [`analysis_report.md`](./analysis_report.md) for detailed findings and visuals.  
📁 **Code**: Available in [`src/eda.py`](./src/eda.py)

---

## 2. 🧠 Model 

### ✅ Objectives

- Select or fine-tune an object detection model (e.g., YOLO, Faster R-CNN).
- Document rationale for model selection.
- Build data loaders compatible with chosen model.
- (Optional for extra points) Train on a subset of data for 1 epoch.

### 🛠️ Implementation Details

- Model loading pipeline created using PyTorch / TensorFlow (user-defined).
- Sample training pipeline included to demonstrate model integration.
- Pretrained model used: **[Specify model here]**, chosen for [rationale].

📁 **Code**: Available in [`src/model/`](./src/model/)

---

## 3. 📈 Evaluation & Visualization 

### ✅ Objectives

- Evaluate the model on validation set.
- Perform **quantitative** analysis (e.g., mAP, precision/recall).
- Perform **qualitative** analysis (visual comparisons of predictions vs. ground truth).
- Identify failure cases and clusters of poor performance.

### 🛠️ Implementation Details

- Evaluation metrics computed and explained.
- Visualization tools implemented to overlay bounding boxes.
- Model performance insights linked back to dataset characteristics.

📁 **Code**: Available in [`src/eval.py`](./src/eval.py)

---

## 🐳 Dockerized Setup

To ensure consistent and reproducible results, the full pipeline is containerized with Docker.

### ✅ Docker Instructions

```bash
# Build the Docker image
docker build -t bdd-object-detection .

# Run the container
docker run -it --rm -v $(pwd):/app bdd-object-detection
