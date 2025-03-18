import io
import os
import logging
import traceback
import numpy as np
import joblib
import tensorflow as tf
import gdown
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, MaxPooling2D

# Initialize Flask App
app = Flask(__name__)

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CNN Model File Info
CNN_MODEL_FILENAME = "cnn_model2.keras"
CNN_MODEL_GDRIVE_ID = "1L1w9g7Z9jqDJ8IdrtPaZyTopaVRe5T2E"

# Function to Download CNN Model from Google Drive if Missing
def download_cnn_model():
    if not os.path.exists(CNN_MODEL_FILENAME):
        try:
            logger.info("CNN model not found. Downloading...")
            gdown.download(f"https://drive.google.com/uc?id={CNN_MODEL_GDRIVE_ID}&confirm=t",
                           CNN_MODEL_FILENAME, quiet=False, fuzzy=True)
            logger.info("CNN model downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download CNN model: {str(e)}")
            raise Exception("Critical Error: CNN model could not be downloaded.")

# Load Models with Error Handling
def load_models():
    try:
        model_tree = joblib.load("decision_tree_model_smote.pkl")
        
        # Ensure CNN model is available before loading
        download_cnn_model()
        model_cnn = load_model(CNN_MODEL_FILENAME)

        model_svm = joblib.load("svm_prob_model2.pkl")
        model_stacking = joblib.load("stacking_model.pkl")
        
        logger.info("All Models Loaded Successfully!")
        return model_tree, model_cnn, model_svm, model_stacking
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Load all models
model_tree, model_cnn, model_svm, model_stacking = load_models()

# Exit if models are not loaded properly
if None in [model_tree, model_cnn, model_svm, model_stacking]:
    raise Exception("Critical error: One or more models failed to load.")

# Preprocess Text Input for Decision Tree
def preprocess_input(form_data):
    try:
        text_data = np.array([[  
            float(form_data.get("age", 0)), 
            int(form_data.get("chronic_disease", 0)), 
            int(form_data.get("allergy", 0)), 
            int(form_data.get("fatigue", 0)), 
            int(form_data.get("anxiety", 0)),
            int(form_data.get("swallowing_difficulty", 0)), 
            int(form_data.get("smoking", 0)), 
            int(form_data.get("chest_pain", 0)), 
            int(form_data.get("shortness_of_breath", 0)), 
            int(form_data.get("peer_pressure", 0)),
            int(form_data.get("alcohol_consuming", 0)), 
            int(form_data.get("gender", 0)), 
            int(form_data.get("wheezing", 0)), 
            int(form_data.get("yellow_fingers", 0)), 
            int(form_data.get("coughing", 0))
        ]])
        return text_data
    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}")
        return None

# Preprocess Image for CNN & SVM
def preprocess_img(file_obj):
    try:
        if file_obj is None or file_obj.filename == "":
            return None

        img_bytes = io.BytesIO(file_obj.read())
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        return img_array
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return None

# Load Pre-trained VGG16 for Feature Extraction
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = MaxPooling2D(pool_size=(2, 2))(base_model.output)
x = Flatten()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Extract Features for SVM
def extract_svm_features(img_array):
    try:
        features = feature_extractor.predict(img_array)
        return features
    except Exception as e:
        logger.error(f"Error extracting SVM features: {str(e)}")
        return None

def make_prediction(features, model, model_name):
    try:
        logger.info(f"Making prediction with {model_name}...")
        if model_name == "SVM":
            prediction = model.predict_proba(features)  
        elif model_name == "CNN":
            prediction = model.predict(features)  
            prediction = tf.nn.softmax(prediction).numpy()  
        else:
            prediction = model.predict(features)  
        
        return prediction
    except Exception as e:
        logger.error(f"Error in {model_name} prediction: {str(e)}\n{traceback.format_exc()}")
        return None

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        logger.info("Processing request...")
        
        if request.method == "POST":
            features = preprocess_input(request.form)
            scan_img_file = request.files.get("scan_img")
            img_data = preprocess_img(scan_img_file)

            prediction_text = "No data provided"
            if features is not None:
                prediction_text = make_prediction(features, model_tree, "Decision Tree")
                if prediction_text is not None:
                    prediction_text = prediction_text.tolist()

            prediction_img = "No image uploaded"
            cnn_probs = None
            if img_data is not None:
                cnn_probs = make_prediction(img_data, model_cnn, "CNN")
                if cnn_probs is not None:
                    prediction_img = np.argmax(cnn_probs, axis=1).tolist()

            prediction_svm = "No image uploaded"
            svm_probs = None
            if img_data is not None:
                svm_features = extract_svm_features(img_data)
                if svm_features is not None:
                    svm_probs = make_prediction(svm_features, model_svm, "SVM")
                    if svm_probs is not None:
                        prediction_svm = np.argmax(svm_probs, axis=1).tolist()

            prediction_stacking = "Stacking Model could not make a prediction"
            if cnn_probs is not None and svm_probs is not None:
                stacking_input = np.hstack((cnn_probs, svm_probs))
                stacking_prediction = make_prediction(stacking_input, model_stacking, "Stacking Model")
                if stacking_prediction is not None:
                    prediction_stacking = stacking_prediction.tolist()

            return render_template("result.html", 
                                   decision_tree_prediction=prediction_text, 
                                   cnn_prediction=prediction_img, 
                                   svm_prediction=prediction_svm,
                                   stacking_prediction=prediction_stacking)

        return render_template("form.html")   
    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return render_template("error.html", error_message=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
