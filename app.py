import os
from flask import Flask, request, render_template
from src.logger import logging
from src.pipeline.predict_pipeline import BrainScan, PredictPipeline
from src.pipeline.train_pipeline import run_training

application=Flask(__name__)

app=application

MODEL_PATH = os.path.join("artifacts", "model.keras")
CLASS_NAMES_PATH = os.path.join("artifacts", "data_transformation", "class_names.json")

predictor = PredictPipeline()


def ensure_model_artifacts():
    missing = [path for path in (MODEL_PATH, CLASS_NAMES_PATH) if not os.path.exists(path)]
    if not missing:
        logging.info("Detected existing model artifacts; skipping training.")
        return

    logging.info("Missing model artifacts %s. Starting training pipeline.", missing)
    run_training()


#Route for Home Page
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')

    uploaded_file = request.files.get('scan')
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template('home.html', error="Please upload an MRI image to continue.")

    try:
        scan = BrainScan.from_file_storage(uploaded_file, image_size=predictor.image_size)
        prediction = predictor.predict(scan)

        return render_template(
            'home.html',
            results=prediction.label,
            confidence=f"{prediction.confidence * 100:.2f}%",
            filename=uploaded_file.filename
        )
    except Exception as exc:
        return render_template('home.html', error=str(exc))

if __name__=="__main__":
    ensure_model_artifacts()
    app.run(host="0.0.0.0")