# AI Annotation Workbench 🚀

### AI-Assisted Image Annotation Tool with Human-in-the-Loop Workflow

---

## 📌 Overview

AI Annotation Workbench is an advanced image annotation system designed to accelerate dataset creation for computer vision tasks.

The system integrates **pretrained object detection models (YOLO .pt files)** with an interactive GUI, enabling fast and efficient annotation through **AI suggestions + human correction**.

This project is developed as part of an MSc thesis focused on **traffic and pedestrian behavior analysis using computer vision**.

---

## 🎯 Key Features

### 🤖 AI-Assisted Annotation

* Load pretrained YOLO `.pt` models
* Automatically generate bounding box suggestions
* Adjustable confidence and IoU thresholds
* Filter suggestions by class

---

### 🧠 Human-in-the-Loop Workflow

* Accept / reject AI-generated suggestions
* Drag, resize, and refine bounding boxes
* Full manual annotation support

---

### 🎯 ROI-Based Annotation

* Polygon-based Region of Interest (ROI)
* Apply annotation or review within selected regions
* Useful for focusing on traffic zones or pedestrian crossings

---

### ⚡ High-Speed Annotation System

* Keyboard-driven workflow for fast labeling
* Smart shortcuts for navigation and editing
* Batch operations for efficiency

---

### 📊 Dataset Management

* YOLO format annotation support (`.txt`)
* Load and save labels automatically
* Track reviewed vs unreviewed images

---

## 🏗️ Tech Stack

* **Python**
* **PyQt5** (GUI)
* **OpenCV**
* **NumPy**
* **Ultralytics YOLO**

---

## 🖥️ System Highlights

* Interactive desktop-based annotation tool
* AI-assisted suggestion pipeline using pretrained models
* Efficient annotation state management
* Designed for real-world dataset creation workflows

---

## 🚀 Installation

```bash
git clone https://github.com/trewto/ai-annotation-workbench.git
cd ai-annotation-workbench
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python annotator_v0.78.py
```

### Steps:

1. Open an image folder
2. (Optional) Load a YOLO `.pt` model
3. Generate annotation suggestions
4. Review and refine annotations
5. Save labels in YOLO format

---

## 🧪 Use Cases

* Traffic analysis datasets
* Pedestrian detection and behavior studies
* Custom object detection dataset creation
* Academic research and thesis work

---

## 📈 Research Motivation

Manual annotation is:

* Time-consuming
* Repetitive
* Prone to human error

This system explores how **AI-assisted workflows** can:

* Reduce annotation effort
* Improve efficiency
* Maintain human-level accuracy

---

## 🔮 Future Work

* Active learning integration
* Multi-object tracking assisted annotation
* Semi-supervised labeling
* Cloud-based collaborative annotation
