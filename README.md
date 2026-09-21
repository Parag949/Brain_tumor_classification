# Brain Tumor Classification

A modular deep learning pipeline that classifies brain MRI scans into four categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** — using a custom CNN built with TensorFlow/Keras, served through a Flask web app.

## Highlights

- **Modular pipeline architecture**: separate `DataIngestion`, `DataTransformation`, and `ModelTrainer` components, each with its own config, orchestrated by `train_pipeline.py`
- **Custom exception handling**: every error is wrapped with automatic file name and line number capture for fast debugging
- **Structured logging**: timestamped run logs for every training session
- **Data augmentation**: random flip, rotation, and zoom applied only to training data (not leaked into validation/test)
- **Training optimizations**: early stopping on validation accuracy, learning-rate reduction on plateau
- **`tf.data` performance tuning**: AUTOTUNE, dataset caching, prefetching, and manual CPU thread-pool configuration
- **Lazy-loaded inference pipeline**: model and class labels load once and are reused across prediction requests, not reloaded per call

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **77%** |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Training Epochs | 10 (with early stopping) |


## Project Structure
