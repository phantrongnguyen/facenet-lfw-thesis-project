from deepface import DeepFace
import numpy as np

def get_embedding(image_path, model):
    if "_" in model:
        model_name, detector = model.split("_", 1)
    else:
        model_name = model
        detector = "mtcnn"

    result = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=True,
        detector_backend=detector,
        align=True
    )

    embedding = np.array(result[0]["embedding"])

    return embedding