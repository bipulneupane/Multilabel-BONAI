import cv2
import numpy as np
import os
import glob

def remove_small_blobs(binary_mask, min_area):
    """
    Remove small blobs from the segmented binary mask based on the minimum area threshold.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):  # Skip the background label (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = 255
    return filtered_mask

def fill_holes(binary_mask):
    """
    Fill holes inside the binary mask using the flood fill technique.
    """
    inverted_mask = cv2.bitwise_not(binary_mask)
    h, w = binary_mask.shape[:2]
    flood_filled = inverted_mask.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    filled_mask = cv2.bitwise_or(binary_mask, flood_filled_inv)
    return filled_mask

def process_image(image_path, output_folder):
    """
    Process a single image: erode, open, remove small blobs, fill holes, and save the result.
    """
    # Load binary mask from PNG image
    binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to ensure it's binary
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Define kernel size for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical kernel

    # Apply erosion (erode operation)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # Apply opening operation (erosion followed by dilation)
    opened_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)

    # Apply Connected Component Analysis to remove small blobs
    min_area = 600  # Adjust the threshold for small blob removal
    filtered_mask = remove_small_blobs(opened_mask, min_area)

    # Apply fill holes operation
    filled_mask = fill_holes(filtered_mask)

    # Save the result
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, filled_mask)

def process_images_in_folder(input_folder, output_folder):
    """
    Process all images in the input folder and save the final output in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PNG images in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))

    for image_path in image_paths:
        print(f"Processing {image_path}...")
        process_image(image_path, output_folder)

# Paths for the input and output folders
input_folder = './data/BONAI-shape/val/results/singlelabel-f-umaxvit/'
output_folder = './data/BONAI-shape/val/results/processed-singlelabel-f-umaxvit/'

# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)

print("Processing complete. Final images saved in:", output_folder)