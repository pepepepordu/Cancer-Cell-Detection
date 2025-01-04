import cv2
import numpy as np
import os

# Define the directory containing the images
image_directory = r"D:\Documents\CU ISE\Y3T1\Exploring Robotics & AI\test\output"

# Define the kernel for erosion and dilation
kernel = np.ones((3, 3), np.uint8)

# Loop through all files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other extensions if needed
        # Load the image
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range of black color in HSV
        lower_black = np.array([0, 0, 0], dtype="uint8")
        upper_black = np.array([255, 50, 204], dtype="uint8")

        # Threshold the HSV image to get a mask of black color
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Apply erosion and dilation on the black mask
        mask_black = cv2.erode(mask_black, kernel, iterations=2)
        mask_black = cv2.dilate(mask_black, kernel, iterations=2)

        # Find contours in the mask
        cnts_black = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_black = cnts_black[0] if len(cnts_black) == 2 else cnts_black[1]

        # Define the range of red color in HSV
        lower_red = np.array([150, 100, 100], dtype="uint8")
        upper_red = np.array([179, 255, 255], dtype="uint8")

        # Threshold the HSV image to get a mask of red color
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Apply erosion and dilation on the red mask
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=1)

        # Find contours in the mask
        cnts_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_red = cnts_red[0] if len(cnts_red) == 2 else cnts_red[1]

        # Count the number of black and red dots
        num_black_dots = len(cnts_black)
        num_red_dots = len(cnts_red)

        # Print the results for the current image
        print(f"Results for {filename}:")
        print("Number of black dots:", num_black_dots)
        print("Number of red dots:", num_red_dots)

        # Draw the contours on the original image
        for c in cnts_black:
            cv2.drawContours(image, [c], -1, (0, 0, 0), 2)
        for c in cnts_red:
            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

        # Save the processed image
        output_path = os.path.join(image_directory, f"processed_{filename}")
        cv2.imwrite(output_path, image)

print("Processing completed.")
