# Brain Tumor MRI Classifier — Proof of Concept

Proof of concept for classifying brain MRI images into four categories using a machine-learning model and a Streamlit interface.

## Classes

The application works with the following output classes:

- glioma tumor;
- meningioma tumor;
- no tumor;
- pituitary tumor.

## Application flow

```text
MRI image
   |
   v
Upload in Streamlit
   |
   v
Resize + normalize image
   |
   v
TensorFlow Lite model
   |
   v
Inference
   |
   v
Class probabilities
   |
   v
Interactive Plotly chart
```

## Tech stack

- Python
- TensorFlow Lite
- PyTorch experiments
- Streamlit
- NumPy
- Pandas
- Plotly
- Pillow

## Repository structure

```text
.
├── app.py
├── model.pth
├── requirements.txt
├── teste.py
├── trusted/
└── README.md
```

`app.py` is the primary Streamlit application using the TensorFlow Lite inference path.

`teste.py` contains a PyTorch-based experimental path retained for model experimentation.

## Running locally

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Model loading

The Streamlit application downloads the TensorFlow Lite model when it is not available locally and then initializes a TFLite interpreter for inference.

## Current status

This repository is a **proof of concept**, not a medical device.

The model output must not be used for diagnosis, treatment decisions, or clinical decision-making.

## Engineering improvements planned

- separate inference, preprocessing, and UI modules;
- include model-training documentation;
- document dataset provenance and licensing;
- add evaluation metrics such as precision, recall, F1-score, and confusion matrix;
- add automated tests for preprocessing and inference;
- containerize the application;
- pin dependency versions for reproducibility.

## Portfolio value

The project demonstrates an applied computer-vision workflow that connects image preprocessing, model inference, a user-facing interface, and result visualization.
