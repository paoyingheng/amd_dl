import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load the trained YOLOv8 classification model
model = YOLO('best.pt')

def predict_image(image):
    # Perform classification
    results = model.predict(source=image, save=False)

    # Debugging: Print the raw `probs` object and its attributes
    print(f"Raw probs object: {results[0].probs}")
    print(f"Probs data (tensor): {results[0].probs.data}")

    # Extract probabilities and class names
    probs = results[0].probs.data.tolist()  # Access the data attribute and convert to list
    class_names = model.names  # Class names

    # Format and return results as a dictionary
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


# Gradio interface
title = "Age-Related Macular Degeneration Classification with YOLOv8"
description = "This application uses YOLOv8, a deep learning model trained on Optical Coherence Tomography (OCT) images. YOLOv8 analyzes the uploaded OCT image and classifies it into one of three categories: Dry AMD, Wet AMD, or Normal. Please upload an OCT image to test the model."
gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
).launch()