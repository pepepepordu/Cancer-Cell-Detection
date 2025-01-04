import os
from roboflow import Roboflow
from PIL import Image

# Initialize Roboflow
rf = Roboflow(api_key="TMcBStNa8S94JRFn5YJo")
project = rf.workspace("ise-ai").project("cancer-cell-detection-dkkhm")
model = project.version(5).model

# Infer on a local image
prediction_response = model.predict("Image_0013_0017.png", confidence=40)
predictions = prediction_response.json()

# Open the original image
original_image = Image.open("Image_0019_0012.jpg")

# Create output directory if it doesn't exist
output_dir = "cropped"

# Loop through each prediction to crop the image
for i, prediction in enumerate(predictions['predictions']):
    # Get the bounding box coordinates
    x1 = int(prediction['x'] - (prediction['width'] / 2))
    y1 = int(prediction['y'] - (prediction['height'] / 2))
    x2 = int(prediction['x'] + (prediction['width'] / 2))
    y2 = int(prediction['y'] + (prediction['height'] / 2))
    
    # Define the crop box (left, upper, right, lower)
    crop_box = (x1, y1, x2, y2)

    # Crop the image
    cropped_image = original_image.crop(crop_box)

    # Save the cropped image in the output directory with a unique name
    cropped_image.save(os.path.join(output_dir, f"cropped_image_{i}.jpg"))  # Save with index in filename

print(f"Cropped images saved in '{output_dir}' directory.")