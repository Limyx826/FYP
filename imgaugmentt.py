import imgaug.augmenters as iaa
import cv2
import os

# Define the augmentation pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.Rotate((-45, 45)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.2))
])

# Define the input and output directories
input_dir = "input"
output_dir = "output"

# Create the output directory if it doesn't already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each image in the input directory, apply the augmentation pipeline, and save the augmented image to the output directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(input_dir, filename))
        for i in range(10):
            augmented_image = augmentation_pipeline(image=image)
            output_filename = f"{filename}_{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, augmented_image)
            print(f"Saved augmented image to {output_path}")
