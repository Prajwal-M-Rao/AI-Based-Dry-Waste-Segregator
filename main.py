from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

app = FastAPI()

# Directory to save uploaded images
UPLOAD_DIRECTORY = "uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Load updated custom model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("? Model loaded successfully.")
        return model
    except Exception as e:
        print(f"? Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load the model.")

# Load the new model
MODEL_PATH = "new_updated_waste_classifier.h5"
custom_model = load_model(MODEL_PATH)

# Updated class labels
class_labels = ['metal', 'paper', 'plastic']  # Removed "glass"

# Preprocess the image for the model
def preprocess_image(image):
    try:
        image_resized = cv2.resize(image, (224, 224)) / 255.0  # Resize and normalize
        print(f"?? Image resized to: {image_resized.shape}")
        return np.expand_dims(image_resized, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"? Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed.")

# Classify waste using the model
def classify_waste(image):
    try:
        processed_image = preprocess_image(image)
        predictions = custom_model.predict(processed_image)
        print(f"?? Model predictions: {predictions}")

        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]
        waste_type = class_labels[predicted_class]

        print(f"? Predicted class: {waste_type}, Confidence: {confidence:.2f}")
        return waste_type, confidence
    except Exception as e:
        print(f"? Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during classification.")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image file, classify it, and return the result.

    :param file: The image file to upload.
    :return: JSON response with the waste type and confidence score.
    """
    try:
        # Validate file input
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file or no file provided.")

        # Save the file correctly
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Open the image file properly
        image = Image.open(file_path).convert("RGB")  # Convert to RGB to avoid format issues
        image = np.array(image)

        # Ensure OpenCV format (convert RGB to BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Classify the waste
        waste_type, confidence = classify_waste(image)

        # Return the result
        return JSONResponse(content={
            "message": "File processed successfully.",
            "filename": file.filename,
            "waste_type": waste_type,
            "confidence": float(confidence)
        })

    except Exception as e:
        print(f"? Error during file upload or processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

@app.get("/")
def root():
    return {"message": "FastAPI server is running. Use the /upload endpoint to upload images."}