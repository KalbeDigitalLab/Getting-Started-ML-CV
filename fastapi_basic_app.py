import base64

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.special import softmax

# Dataset Metadata
RGB_MEAN = [0.51442681, 0.43435301, 0.33421855]
RGB_STD = [0.24099932, 0.246478, 0.23652802]

# Transformation pipeline using Albumentations
transformation_pipeline = A.Compose([
    A.CenterCrop(width=384, height=384),
    A.Normalize(mean=RGB_MEAN, std=RGB_STD)
])

# Load the ONNX model to onnxruntime
onnx_model_path = 'food101_resenet18.onnx'
model = ort.InferenceSession(onnx_model_path)  # Update with the correct model path

# Get model input/output names
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

class_names = ['apple_pie', 'bibimbap', 'cannoli', 'edamame', 'falafel', 'french_toast', 'ice_cream', 'ramen', 'sushi', 'tiramisu']
class_names.sort()

app = FastAPI()

class ImageInput(BaseModel):
    image_base64: str

def preprocess_image(image: np.ndarray):
    """Preprocess the input image.

    Note that the input image is in RGB mode.

    Parameters
    ----------
    image: np.ndarray
        Input image from callback.
    """

    image = transformation_pipeline(image=image)['image']
    image = np.transpose(image, (2, 1, 0))
    image = np.expand_dims(image, axis=0)

    return image

@app.post("/classify")
def classify_image(image_input: ImageInput):
    # Decode the base64 image string
    image_data =  np.fromstring(base64.b64decode(image_input.image_base64), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If input not valid, return dummy data or raise error
    if image is None:
        return {"cat": 0.8, "dog": 0.2}

    # Preprocess image
    processed_image = preprocess_image(image)

    # Run inference using the ONNX model
    prediction = model.run([output_name], {input_name: processed_image})[0] # takes the first output

    # Postprocess result
    prediction = softmax(prediction, axis=1)[0] # Apply softmax to normalize the output
    labeled_result = {name:score for name, score in zip(class_names, prediction.tolist())}

    return labeled_result
