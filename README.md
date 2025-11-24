# Brain Tumor Classification

This repository repackages the convolutional neural network developed in `brain/cancer_Cnn.ipynb` into a modular training pipeline and Flask inference service. Upload brain MRI slices to detect tumor types using the same TensorFlow architecture as the notebook.

## Project layout
- `src/components` holds ingestion, transformation, and training stages for the CNN workflow.
- `src/pipeline/train_pipeline.py` orchestrates an end-to-end training run.
- `src/pipeline/predict_pipeline.py` powers the Flask app to score MRI images.
- `templates/` contains the minimal frontend for uploading scans.
- `brain/cancer_Cnn.ipynb` remains unchanged for exploration and experimentation.

## Getting started
1. Place the dataset under either `brain/Brain-Tumor-Classification-DataSet-master` **or** `Brain-Tumor-Classification-DataSet-master/` at the repo root (or set `BRAIN_TUMOR_DATA_DIR`). The folder must contain `Training/` and `Testing/` subdirectories.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train the model: `python -m src.pipeline.train_pipeline`.
4. Launch the web app: `python app.py` and open the provided URL to upload MRI images.

The trained model along with class labels and metrics are written to the `artifacts/` directory.