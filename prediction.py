import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

import matplotlib.pyplot as plt
import cv2

from utils import *


import torch

def load_model(saved_ckpt, model_name, encoder_name, encoder_weights, num_classes, activation, device):
    """
    Loads a segmentation model from a saved checkpoint and sets it to evaluation mode.

    Args:
        saved_ckpt (str): Path to the saved model checkpoint file.
        model_name (str): The name of the model architecture (e.g., 'Unet', 'FPN') from the segmentation models pytorch (SMP) library.
        encoder_name (str): The name of the encoder backbone used in the model (e.g., 'resnet34').
        encoder_weights (str or None): Pretrained weights for the encoder (e.g., 'imagenet') or None for random initialization.
        num_classes (int): The number of output classes for the segmentation task.
        activation (str or callable): The activation function to apply to the model's output (e.g., 'sigmoid', 'softmax').
        device (torch.device): The device to load the model on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model set to evaluation mode.
    
    Example:
        model = load_model('checkpoint.pth', 'Unet', 'resnet34', 'imagenet', 3, 'sigmoid', torch.device('cuda'))

    This function:
        - Initializes the model using the specified architecture, encoder, and other configurations.
        - Loads the saved model weights from the checkpoint.
        - Transfers the model to the specified device (e.g., GPU or CPU).
        - Sets the model to evaluation mode (`model.eval()`).
    """
    
    # Load the model with the specified architecture and configurations
    model = get_model_from_smp(model_name, encoder_name, encoder_weights, num_classes, activation)
    
    # Load the saved checkpoint
    checkpoint = torch.load(saved_ckpt, map_location=device)
    
    # Load model state dictionary from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Transfer the model to the specified device (CPU/GPU)
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def predict_and_save_masks_multilabel_frs(model, input_folder, output_folder, device, img_size=(512, 512)):
    """
    Predict multilabel segmentation masks (footprint, roof, and shape) for each image in the input folder using the provided model
    and save the predicted masks to the specified output folder.

    Args:
        model (torch.nn.Module): The trained segmentation model used for prediction.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where predicted masks will be saved.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        img_size (tuple): Size (width, height) to resize the input images to before prediction (default: (512, 512)).

    Returns:
        None: The function saves the predicted masks (footprint, roof, and shape) as PNG files in the output folder.

    Example:
        predict_and_save_masks_multilabel_frh(model, 'input_images/', 'output_masks/', torch.device('cuda'), img_size=(512, 512))

    This function:
        - Loads each image from the `input_folder`.
        - Resizes the image to the specified `img_size` to match the model's input size.
        - Predicts segmentation masks using the provided model.
        - Converts predicted masks to binary masks using a threshold (0.5).
        - Resizes the masks back to the original image size.
        - Saves the predicted masks for footprint, roof, and shape with appropriate filenames in the `output_folder`.
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(input_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  # Store the original image size
            image = image.resize(img_size)  # Resize image to match model input size
            image_tensor = ToTensor()(image).unsqueeze(0).to(device)  # Convert image to tensor and move to device

            # Predict masks
            with torch.no_grad():
                output = model(image_tensor)  # Get model output
                preds = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid activation and move to CPU

            # Convert predictions to binary masks
            pred_masks_np = (preds > 0.5).astype(np.uint8).squeeze()

            # Resize masks back to original size
            pred_masks_np = [cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST) for mask in pred_masks_np]

            # Save masks with appropriate file names
            base_filename = os.path.splitext(img_file)[0]
            foot_mask_path = os.path.join(output_folder, f"{base_filename}_foot.png")
            roof_mask_path = os.path.join(output_folder, f"{base_filename}_roof.png")
            shape_mask_path = os.path.join(output_folder, f"{base_filename}_shape.png")

            cv2.imwrite(foot_mask_path, pred_masks_np[0] * 255)  # Convert mask to 0-255 range for saving
            cv2.imwrite(roof_mask_path, pred_masks_np[1] * 255)
            cv2.imwrite(shape_mask_path, pred_masks_np[2] * 255)

            print(f"Saved masks for {img_file}")


def predict_and_save_masks_multilabel_fr(model, input_folder, output_folder, device, img_size=(512, 512)):
    """
    Predict multilabel segmentation masks (footprint and roof) for each image in the input folder using the provided model
    and save the predicted masks to the specified output folder.

    Args:
        model (torch.nn.Module): The trained segmentation model used for prediction.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where predicted masks will be saved.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        img_size (tuple): Size (width, height) to resize the input images to before prediction (default: (512, 512)).

    Returns:
        None: The function saves the predicted masks (footprint and roof) as PNG files in the output folder.

    Example:
        predict_and_save_masks_multilabel_fr(model, 'input_images/', 'output_masks/', torch.device('cuda'), img_size=(512, 512))

    This function:
        - Loads each image from the `input_folder`.
        - Resizes the image to the specified `img_size` to match the model's input size.
        - Predicts segmentation masks using the provided model.
        - Converts predicted masks to binary masks using a threshold (0.5).
        - Resizes the masks back to the original image size.
        - Saves the predicted masks for footprint and roof with appropriate filenames in the `output_folder`.
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(input_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  # Store the original image size
            image = image.resize(img_size)  # Resize image to match model input size
            image_tensor = ToTensor()(image).unsqueeze(0).to(device)  # Convert image to tensor and move to device

            # Predict masks
            with torch.no_grad():
                output = model(image_tensor)  # Get model output
                preds = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid activation and move to CPU

            # Convert predictions to binary masks
            pred_masks_np = (preds > 0.5).astype(np.uint8).squeeze()

            # Resize masks back to original size
            pred_masks_np = [cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST) for mask in pred_masks_np]

            # Save masks with appropriate file names
            base_filename = os.path.splitext(img_file)[0]
            foot_mask_path = os.path.join(output_folder, f"{base_filename}_foot.png")
            roof_mask_path = os.path.join(output_folder, f"{base_filename}_roof.png")

            cv2.imwrite(foot_mask_path, pred_masks_np[0] * 255)  # Convert mask to 0-255 range for saving
            cv2.imwrite(roof_mask_path, pred_masks_np[1] * 255)

            print(f"Saved masks for {img_file}")


def predict_and_save_masks_singlelabel_f(model, input_folder, output_folder, device, img_size=(512, 512)):
    """
    Predict single-label footprint segmentation masks for each image in the input folder using the provided model
    and save the predicted masks to the specified output folder.

    Args:
        model (torch.nn.Module): The trained segmentation model used for prediction.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where predicted footprint masks will be saved.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        img_size (tuple): Size (width, height) to resize the input images to before prediction (default: (512, 512)).

    Returns:
        None: The function saves the predicted footprint masks as PNG files in the output folder.

    Example:
        predict_and_save_masks_singlelabel_f(model, 'input_images/', 'output_masks/', torch.device('cuda'), img_size=(512, 512))

    This function:
        - Loads each image from the `input_folder`.
        - Resizes the image to the specified `img_size` to match the model's input size.
        - Predicts a footprint segmentation mask using the provided model.
        - Converts the predicted mask to a binary mask using a threshold (0.5).
        - Resizes the mask back to the original image size.
        - Saves the predicted footprint mask with an appropriate filename in the `output_folder`.
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(input_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  # Store original image size
            image = image.resize(img_size)  # Resize image to match model input size
            image_tensor = ToTensor()(image).unsqueeze(0).to(device)  # Convert image to tensor and move to device

            # Predict mask
            with torch.no_grad():
                output = model(image_tensor)  # Get model output
                preds = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid activation and move to CPU

            # Convert prediction to binary mask
            pred_mask_np = (preds > 0.5).astype(np.uint8).squeeze()

            # Resize mask back to original size
            pred_mask_np = cv2.resize(pred_mask_np, original_size, interpolation=cv2.INTER_NEAREST)

            # Save mask with appropriate file name
            base_filename = os.path.splitext(img_file)[0]
            footprint_mask_path = os.path.join(output_folder, f"{base_filename}_foot.png")

            cv2.imwrite(footprint_mask_path, pred_mask_np * 255)  # Convert mask to 0-255 range for saving

            print(f"Saved footprint mask for {img_file}")

# Examples:

# Defining model checkpoint path
# multilabel_frs = './trained_models/BONAI-multilabel-train1-Unet-tu-maxvit_base_tf_512-BONAI-shape-200ep-Adam-dice_loss.pth'
# roof_foot_model = './trained_models/BONAI-multilabel-nohull-Unet-tu-maxvit_base_tf_512-BONAI-shape-50ep-Adam-dice_loss.pth'
# single_foot_model = './trained_models/BONAI-single-footprint-Unet-tu-maxvit_base_tf_512-BONAI-shape-50ep-Adam-dice_loss.pth'

# Load the model into a device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# multilabel_frs = load_model(multilabel_frs, 'Unet', 'tu-maxvit_base_tf_512', None, 3, 'sigmoid', device)
# roof_foot_model = load_model(roof_foot_model, 'Unet', 'tu-maxvit_base_tf_512', None, 2, 'sigmoid', device)
# single_foot_model = load_model(single_foot_model, 'Unet', 'tu-maxvit_base_tf_512', None, 1, 'sigmoid', device)

# Run predictions and save masks
# predict_and_save_masks_multilabel_frs(multilabel_frs, './data/BONAI-shape/val/image/', './data/BONAI-shape/val/results/multilabel-all-umaxvit/', device)
# predict_and_save_masks_multilabel_fr(roof_foot_model, './data/BONAI-shape/val/image/', './data/BONAI-shape/val/results/multilabel-fr-umaxvit/', device)
# predict_and_save_masks_singlelabel_f(single_foot_model, './data/BONAI-shape/val/image/', './data/BONAI-shape/val/results/singlelabel-f-umaxvit/', device)

