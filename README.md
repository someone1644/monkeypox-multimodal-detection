# Monkeypox Detection using Multi-Modal Deep Learning

## Overview
This project implements a multi-modal deep learning system that combines image-based features and clinical metadata to simulate real-world medical diagnosis of Monkeypox.

The model integrates a pre-trained Xception convolutional neural network with structured clinical inputs (fever and lymphadenopathy) using a late fusion architecture.

---

## Key Features
- Multi-input neural network combining image and tabular data
- Transfer learning using Xception
- Custom data pipeline for synchronized multi-modal inputs
- Achieves ~91–92% accuracy and 100% recall on test data
- Interactive Gradio-based interface for real-time predictions

---

## Project Structure
- `app/` – Gradio application  
- `model/` – Training scripts and saved model  
- `evaluation/` – Evaluation and testing scripts  
- `utils/` – Data preprocessing scripts  
- `data/` – Dataset and metadata CSV files  

---

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python app/app.py
```
## Model Training
```bash
python model/train.py
```
## Evaluation
```bash
python evaluation/evaluate.py
```

## Results
* **Accuracy:** ~91–92%
* **Recall (Monkeypox):** 100%
* No false negatives observed in test set

## Notes
* Clinical metadata used in this project is synthetically generated to simulate real-world diagnostic conditions.
* This project is intended for academic and research purposes only and should not be used for medical diagnosis.