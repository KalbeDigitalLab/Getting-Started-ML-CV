from fastapi import FastAPI
from pydantic import BaseModel
import base64

app = FastAPI()

class ImageInput(BaseModel):
    image_base64: str

@app.post("/classify")
def classify_image(image: ImageInput):
    # Decode the base64 image string
    image_data = base64.b64decode(image.image_base64)

    # Process the image (dummy code)
    # Replace this with your actual machine learning model prediction code
    # Here, we assume the image is classified as 80% cat and 20% dog
    classification_results = {"cat": 0.8, "dog": 0.2}

    return classification_results
