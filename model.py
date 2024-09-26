import os, cv2
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
import albumentations as albu

import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from utils import *

def evaluate_singleclass_edns(saved_ckpt, edn, bb, data_type, classes, loss_func):
    """
    Evaluates a single-class segmentation model on a validation dataset.

    Args:
        saved_ckpt (str): Path to the saved model checkpoint.
        edn (str): The model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the model.
        data_type (str): The type of data (train, val, test) used for validation.
        classes (list): List of classes for segmentation.
        loss_func (str): The name of the loss function to be used for evaluation.
    
    Returns:
        None: Prints evaluation metrics including loss, precision, recall, IoU, and F1 score for the validation dataset.
    
    Example:
        evaluate_singleclass_edns('checkpoint.pth', 'Unet', 'resnet34', 'val', ['building'], 'jaccard_loss')
    
    This function:
        - Loads the validation dataset using a specific preprocessing function.
        - Loads the model from SMP, sets the pretrained state from the provided checkpoint, and prepares it for evaluation.
        - Defines evaluation metrics such as precision, recall, IoU, and F1 score.
        - Runs the evaluation on the validation dataset and prints the evaluation metrics.
    """
    ENCODER = bb
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = None
    ACTIVATION = 'sigmoid'
    
    # Define transformations (here only ToTensor)
    transform = Compose([
        ToTensor(),  
    ])
    
    print("********************************************************************************")
    print("********************************************************************************")

    # Setup directories for dataset
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir, x_valid_dir, x_test_dir = (
        os.path.join(DATA_DIR, 'train', 'image'), 
        os.path.join(DATA_DIR, 'val', 'image'), 
        os.path.join(DATA_DIR, 'test', 'image')
    )
    y_train_dir, y_valid_dir, y_test_dir = (
        os.path.join(DATA_DIR, 'train', 'label'), 
        os.path.join(DATA_DIR, 'val', 'label'), 
        os.path.join(DATA_DIR, 'test', 'label')
    )
    
    # Create validation dataset and dataloader
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    valid_dataset = SegmentationDatasetSinlgeClass(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # Load model and pretrained weights
    model = get_model_from_smp(edn, bb, ENCODER_WEIGHTS, len(classes), ACTIVATION)
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()

    # Count model parameters
    params = sum(p.numel() for p in model.parameters())

    # Define loss function and metrics
    loss = get_loss(loss_func)
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    
    # Setup evaluation epoch
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Print model and evaluation details
    print("Encoder: ", bb)
    print("Checkpoint: ", saved_ckpt)
    print("Validated on: ", data_type)
    print("Class: ", classes)
    print("Net Params: ", params)

    # Run evaluation
    valid_logs = test_epoch.run(valid_dataloader)

    # Print evaluation results
    print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}".format(
        list(valid_logs.items())[0][1],
        list(valid_logs.items())[1][1],
        list(valid_logs.items())[2][1],
        list(valid_logs.items())[3][1],
        list(valid_logs.items())[4][1])
    )

    print("********************************************************************************")
    print("********************************************************************************")


def evaluate_multilabel_model(saved_ckpt, edn, bb, data_type, classes, loss_func):
    """
    Evaluates a multilabel segmentation model on a validation dataset and prints class-wise metrics.

    Args:
        saved_ckpt (str): Path to the saved model checkpoint.
        edn (str): The model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the model.
        data_type (str): The type of data (train, val, test) used for validation.
        classes (list): List of classes for multilabel segmentation.
        loss_func (str): The name of the loss function to be used for evaluation.

    Returns:
        None: Prints evaluation metrics (Precision, Recall, IoU, F1 score) for each class in the multilabel task.

    Example:
        evaluate_multilabel_model('checkpoint.pth', 'Unet', 'resnet34', 'val', ['footprint', 'roof'], 'dice_loss')

    This function:
        - Loads the validation dataset using specific transformations.
        - Loads the model from SMP, sets pretrained weights from the provided checkpoint, and prepares it for evaluation.
        - Defines metrics such as Precision, Recall, IoU, and F1 score.
        - Runs the evaluation on the validation dataset for each class and prints the average metrics.
    """
    
    # Set encoder and device
    ENCODER = bb
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations (e.g., ToTensor)
    transform = Compose([
        ToTensor(),  
    ])
    
    ####### DATASET GENERATOR
    # Set directories for validation data
    DATA_DIR = './data/' + data_type + '/'
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')

    # Create validation dataset and dataloader
    valid_dataset = SegmentationDatasetMultiClass(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### LOAD MODEL
    n_classes = 1 if len(classes) == 1 else len(classes)  # For multilabel segmentation
    ACTIVATION = 'sigmoid'
    model = get_model_from_smp(edn, ENCODER, None, n_classes, ACTIVATION)
    
    # Load model state from checkpoint
    checkpoint = torch.load(saved_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Count model parameters
    params = sum(p.numel() for p in model.parameters())

    # Define metrics
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5), 
        smp.utils.metrics.Recall(threshold=0.5), 
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    ####### PRINTING SOME DETAILS
    print("Encoder: ", bb)
    print("Checkpoint: ", saved_ckpt)
    print("Validated on: ", data_type)
    print("Classes: ", classes)
    print("Net Params: ", params)

    # Initialize dictionaries to store metrics for each class
    class_metrics = {cls: {metric.__name__: 0 for metric in metrics} for cls in classes}

    # Iterate over the validation dataset
    with torch.no_grad():
        for images, true_masks in valid_dataloader:
            images = images.to(DEVICE)
            true_masks = [mask.to(DEVICE) for mask in torch.unbind(true_masks, dim=1)]
            outputs = model(images)
            
            # Assuming `outputs` is of shape [batch_size, num_classes, height, width]
            for i, class_name in enumerate(classes):
                output_class = outputs[:, i, :, :]
                for metric in metrics:
                    class_metrics[class_name][metric.__name__] += metric(output_class, true_masks[i]).item()

    # Normalize metrics over the dataset
    num_samples = len(valid_dataloader)
    for class_name in class_metrics:
        for metric_name in class_metrics[class_name]:
            class_metrics[class_name][metric_name] /= num_samples

    # Print the results
    for class_name in class_metrics:
        print(f"Metrics for {class_name.capitalize()}:")
        for metric_name, metric_value in class_metrics[class_name].items():
            print(f"  {metric_name.capitalize()}: {metric_value:.4f}")

    print("********************************************************************************")
    print("********************************************************************************")


def train_single_class_png_labels(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func):
    """
    Trains a segmentation model on a single-class PNG label dataset and saves the best-performing model checkpoint.

    Args:
        model_name (str): The model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the model.
        data_type (str): The type of dataset (e.g., 'WHU', 'custom') used for training.
        CLASSES (list): List of classes for segmentation (in this case, usually a single class like ['building']).
        BATCH_SIZE (int): Batch size for training.
        EPOCHS (int): The number of epochs to train for.
        LR (float): The learning rate for the optimizer.
        optimiser (str): The optimizer to use (e.g., 'Adam', 'SGD').
        loss_func (str): The loss function to be used for training (e.g., 'dice_loss', 'jaccard_loss').

    Returns:
        None: Trains the model and saves the best-performing model based on IoU score during validation.

    Example:
        train_single_class_png_labels('Unet', 'resnet34', 'WHU', ['building'], 8, 50, 0.001, 'Adam', 'dice_loss')

    This function:
        - Loads the dataset using the specified encoder and preprocessing.
        - Initializes the model, loss function, optimizer, and scheduler.
        - Trains the model for a given number of epochs, evaluates it after each epoch, and saves the best model.
        - Prints relevant training metrics (e.g., Precision, Recall, IoU, F1, learning rate).
    """
    
    # Set encoder and device
    ENCODER = bb
    ENCODER_WEIGHTS = None
    CLASSES = ['building']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model save path
    model_save = model_name + ENCODER
    checkpoint_path = './trained_models/MultiTask/' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep' + '-' + str(optimiser) + '-' + str(loss_func) + '.pth'

    ####### DATASET GENERATOR
    # Define data directories for training and validation
    DATA_DIR = './data/' + data_type + '/'  
    x_train_dir, x_valid_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image')
    y_train_dir, y_valid_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label')

    # Dataset for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = SingleClassPNGLabelDataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = SingleClassPNGLabelDataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    # Create segmentation model with pretrained encoder
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # For binary and multiclass segmentation
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)

    # Print model and dataset details
    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    
    print("\n********************************************************************************")
    print("Training model ", model_name)
    print("Encoder: ", ENCODER)
    print("Network params:", params)
    print("Dataset: ", data_type)
    print("Task: SingleClass - PNG Label Dataset")
    print("Classes: ", CLASSES)

    # Define metrics, loss function, optimizer, and scheduler
    metrics = [smp.utils.metrics.Precision(threshold=0.5), smp.utils.metrics.Recall(threshold=0.5), smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(threshold=0.5)]
    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model_params, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    # Initialize training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

    max_score = 0

    # Training loop
    for i in range(0, EPOCHS):
        print('\nEpoch: {}/{}'.format(i+1, EPOCHS))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        
        # Print validation metrics
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
            list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr'])
        )
        
        # Save best model based on IoU score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path) 
            print('Model saved!')
            scheduler.step(max_score)

    print("********************************************************************************\n\n")


def train_unet_singleclass(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_prefix):
    """
    Trains a U-Net or similar segmentation model for single-class segmentation and saves the best model checkpoint.

    Args:
        model_name (str): The model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the U-Net model.
        data_type (str): The type of dataset (e.g., 'WHU', 'custom') used for training.
        CLASSES (list): List of classes for segmentation (in this case, usually a single class like ['building']).
        BATCH_SIZE (int): Batch size for training.
        EPOCHS (int): The number of epochs to train for.
        LR (float): The learning rate for the optimizer.
        optimiser (str): The optimizer to use (e.g., 'Adam', 'SGD').
        loss_func (str): The loss function to be used for training (e.g., 'dice_loss', 'jaccard_loss').
        ckpt_prefix (str): Prefix for the checkpoint file where the trained model will be saved.

    Returns:
        None: Trains the U-Net model and saves the best-performing model based on the IoU score during validation.

    Example:
        train_unet_singleclass('Unet', 'resnet34', 'WHU', ['building'], 8, 50, 0.001, 'Adam', 'dice_loss', 'ckpt_')

    This function:
        - Loads the dataset using specific transformations.
        - Initializes the U-Net model, loss function, optimizer, and scheduler.
        - Trains the model for a given number of epochs, evaluates it after each epoch, and saves the best model.
        - Prints relevant training metrics (e.g., Precision, Recall, IoU, F1, learning rate).
    """
    
    # Set encoder and device
    ENCODER = bb
    ENCODER_WEIGHTS = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformation
    transform = Compose([
        ToTensor(),  
    ])
    
    # Create the name for the checkpoint file
    ckpt_name = ckpt_prefix + model_name + '-' + ENCODER + '-' + data_type + '-' + str(EPOCHS) + 'ep' + '-' + optimiser + '-' + loss_func + '.pth'

    ####### DATASET GENERATOR
    # Define data directories for training, validation, and testing
    DATA_DIR = './data/' + data_type + '/'   
    x_train_dir, x_valid_dir, x_test_dir = (
        os.path.join(DATA_DIR, 'train', 'image'),
        os.path.join(DATA_DIR, 'val', 'image'),
        os.path.join(DATA_DIR, 'test', 'image')
    )
    y_train_dir, y_valid_dir, y_test_dir = (
        os.path.join(DATA_DIR, 'train', 'label'),
        os.path.join(DATA_DIR, 'val', 'label'),
        os.path.join(DATA_DIR, 'test', 'label')
    )

    # Load training and validation datasets
    train_dataset = SegmentationDatasetSinlgeClass(img_dir=x_train_dir, ann_dir=y_train_dir, transform=transform)
    valid_dataset = SegmentationDatasetSinlgeClass(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    # Create segmentation model with the specified encoder
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # Case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    
    # Print model and dataset details
    print("\n********************************************************************************")
    print("Training model:", model_name)
    print("Encoder:", ENCODER)
    print("Network params:", sum(p.numel() for p in model.parameters()))
    print("Dataset:", data_type)
    print("Task: SingleTask")
    print("Classes:", CLASSES)
    
    # Define metrics, loss function, optimizer, and scheduler
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5), 
        smp.utils.metrics.Recall(threshold=0.5), 
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    
    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    # Initialize training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

    max_score = 0
    for i in range(EPOCHS):
        print('\nEpoch: {}/{}'.format(i + 1, EPOCHS))
        
        # Run training and validation
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        
        # Print validation metrics
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
            list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr'])
        )
        
        # Save the model with the best IoU score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
            print('Model saved!')
            scheduler.step(max_score)

    print("********************************************************************************\n\n")


def train_unet_encoders_multilabel(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_prefix):
    """
    Trains a U-Net or similar segmentation model with specific encoders for multilabel segmentation tasks and saves the best model checkpoint.

    Args:
        model_name (str): The model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the U-Net model.
        data_type (str): The type of dataset (e.g., 'WHU', 'custom') used for training.
        CLASSES (list): List of classes for multilabel segmentation.
        BATCH_SIZE (int): Batch size for training.
        EPOCHS (int): The number of epochs to train for.
        LR (float): The learning rate for the optimizer.
        optimiser (str): The optimizer to use (e.g., 'Adam', 'SGD').
        loss_func (str): The loss function to be used for training (e.g., 'jaccard_loss', 'dice_loss').
        ckpt_prefix (str): Prefix for the checkpoint file where the trained model will be saved.

    Returns:
        None: Trains the U-Net model and saves the best-performing model based on the IoU score during validation.

    Example:
        train_unet_encoders_multilabel('Unet', 'resnet34', 'WHU', ['footprint', 'roof'], 8, 50, 0.001, 'Adam', 'dice_loss', 'ckpt_')

    This function:
        - Loads the dataset using specific transformations.
        - Initializes the U-Net model, loss function, optimizer, and scheduler for multilabel segmentation.
        - Trains the model for a given number of epochs, evaluates it after each epoch, and saves the best model.
        - Prints relevant training metrics (e.g., Precision, Recall, IoU, F1, learning rate).
    """
    
    # Set encoder and device
    ENCODER = bb
    ENCODER_WEIGHTS = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations (ToTensor for image preprocessing)
    transform = Compose([
        ToTensor(),
    ])
    
    # Create the name for the checkpoint file
    ckpt_name = ckpt_prefix + model_name + '-' + ENCODER + '-' + data_type + '-' + str(EPOCHS) + 'ep' + '-' + optimiser + '-' + loss_func + '.pth'

    ####### DATASET GENERATOR
    # Define data directories for training, validation, and testing
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir, x_valid_dir, x_test_dir = (
        os.path.join(DATA_DIR, 'train', 'image'),
        os.path.join(DATA_DIR, 'val', 'image'),
        os.path.join(DATA_DIR, 'test', 'image')
    )
    y_train_dir, y_valid_dir, y_test_dir = (
        os.path.join(DATA_DIR, 'train', 'label'),
        os.path.join(DATA_DIR, 'val', 'label'),
        os.path.join(DATA_DIR, 'test', 'label')
    )

    # Load training and validation datasets
    train_dataset = SegmentationDatasetMultiClass(img_dir=x_train_dir, ann_dir=y_train_dir, transform=transform)
    valid_dataset = SegmentationDatasetMultiClass(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle = True for random train data
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) # shuffle = True for random val data

    ####### COMPILE MODEL
    # Create segmentation model with the specified encoder for multilabel tasks
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # Case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid'  # Use sigmoid for multilabel segmentation
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    
    # Print model and dataset details
    print("\n********************************************************************************")
    print("Training model:", model_name)
    print("Encoder:", ENCODER)
    print("Network params:", sum(p.numel() for p in model.parameters()))
    print("Dataset:", data_type)
    print("Task: MultiLabel")
    print("Classes:", CLASSES)
    
    # Define metrics, loss function, optimizer, and scheduler
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5), 
        smp.utils.metrics.Recall(threshold=0.5), 
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    
    loss = get_multilabel_loss(loss_func)
    
    # Update loss function name
    if loss_func == 'jaccard_loss':
        loss.__name__ = 'Jaccard_loss'
    elif loss_func == 'dice_loss':
        loss.__name__ = 'Dice_loss'
    else:
        print('Loss name is wrong.')

    optim = get_optim(optimiser, model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    # Initialize training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

    max_score = 0
    for i in range(EPOCHS):
        print('\nEpoch: {}/{}'.format(i + 1, EPOCHS))
        
        # Run training and validation
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        
        # Print validation metrics
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
            list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr'])
        )
        
        # Save the model with the best IoU score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
            print('Model saved!')
            scheduler.step(max_score)

    print("********************************************************************************\n\n")


def train_multilabel_foot_and_roof(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_prefix):
    """
    Trains a segmentation model for multilabel tasks involving footprint and roof segmentation only, and saves the best model checkpoint.

    Args:
        model_name (str): The name of the model architecture to use from the segmentation models pytorch (SMP) library.
        bb (str): The encoder backbone to use in the model.
        data_type (str): The dataset type (e.g., 'WHU', 'custom') used for training.
        CLASSES (list): List of classes for segmentation (e.g., ['footprint', 'roof']).
        BATCH_SIZE (int): Batch size for training.
        EPOCHS (int): The number of epochs to train the model.
        LR (float): The learning rate for the optimizer.
        optimiser (str): The optimizer to use (e.g., 'Adam', 'SGD').
        loss_func (str): The loss function to use for training (e.g., 'jaccard_loss', 'dice_loss').
        ckpt_prefix (str): Prefix for the checkpoint file where the trained model will be saved.

    Returns:
        None: Trains the model and saves the best-performing model based on the IoU score during validation.

    Example:
        train_multilabel_foot_and_roof('Unet', 'resnet34', 'WHU', ['footprint', 'roof'], 8, 50, 0.001, 'Adam', 'dice_loss', 'ckpt_')

    This function:
        - Loads the training and validation datasets for footprint and roof segmentation.
        - Initializes the model, loss function, optimizer, and learning rate scheduler.
        - Trains the model for a given number of epochs, evaluates it on the validation dataset, and saves the best-performing model.
        - Prints relevant training and validation metrics (Precision, Recall, IoU, F1, loss, learning rate).
    """
    
    # Set encoder and device
    ENCODER = bb
    ENCODER_WEIGHTS = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations (ToTensor for image preprocessing)
    transform = Compose([
        ToTensor(),
    ])
    
    # Create the name for the checkpoint file
    ckpt_name = ckpt_prefix + model_name + '-' + ENCODER + '-' + data_type + '-' + str(EPOCHS) + 'ep' + '-' + optimiser + '-' + loss_func + '.pth'

    ####### DATASET GENERATOR
    # Define data directories for training, validation, and testing
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir, x_valid_dir, x_test_dir = (
        os.path.join(DATA_DIR, 'train', 'image'),
        os.path.join(DATA_DIR, 'val', 'image'),
        os.path.join(DATA_DIR, 'test', 'image')
    )
    y_train_dir, y_valid_dir, y_test_dir = (
        os.path.join(DATA_DIR, 'train', 'label'),
        os.path.join(DATA_DIR, 'val', 'label'),
        os.path.join(DATA_DIR, 'test', 'label')
    )

    # Load training and validation datasets
    train_dataset = SegmentationDatasetFootprintRoof(img_dir=x_train_dir, ann_dir=y_train_dir, transform=transform)
    valid_dataset = SegmentationDatasetFootprintRoof(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    ####### COMPILE MODEL
    # Create segmentation model with the specified encoder for multilabel tasks
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # Case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid'  # Use sigmoid for multilabel segmentation
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    
    # Print model and dataset details
    print("\n********************************************************************************")
    print("Training model:", model_name)
    print("Encoder:", ENCODER)
    print("Network params:", sum(p.numel() for p in model.parameters()))
    print("Dataset:", data_type)
    print("Task: MultiLabel - footprint and roof (no hull)")
    print("Classes:", CLASSES)
    
    # Define metrics, loss function, optimizer, and scheduler
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5), 
        smp.utils.metrics.Recall(threshold=0.5), 
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    
    loss = get_multilabel_loss(loss_func)
    
    # Update loss function name
    if loss_func == 'jaccard_loss':
        loss.__name__ = 'Jaccard_loss'
    elif loss_func == 'dice_loss':
        loss.__name__ = 'Dice_loss'
    else:
        print('Loss name is wrong.')

    optim = get_optim(optimiser, model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    # Initialize training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

    max_score = 0
    for i in range(EPOCHS):
        print('\nEpoch: {}/{}'.format(i + 1, EPOCHS))
        
        # Run training and validation
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        
        # Print validation metrics
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
            list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr'])
        )
        
        # Save the model with the best IoU score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
            print('Model saved!')
            scheduler.step(max_score)

    print("********************************************************************************\n\n")


# Examples:

# Training a model:
# *******************************
# Training on single label/class
#train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-footprint-')
#train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['roof'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-roof-')
#train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['shape'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-shape-')

# Training a multilabel model
#train_unet_encoders_multilabel('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof', 'shape'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-multilabel-')

# Train only on roof and footprint (no shape)
# train_multilabel_foot_and_roof('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-multilabel-nohull-')
# *******************************

# Evaluating a model
# *******************************
# multilabel_model = "./trained_models/BONAI-multilabel-Unet-tu-maxvit_base_tf_512-BONAI-shape-50ep-Adam-dice_loss.pth"
# footprint_model = "./trained_models/MultiTask/Umaxvit/BONAI-single-footprint-Unet-tu-maxvit_base_tf_512-BONAI-shape-200ep-Adam-dice_loss.pth"
# evaluate_multilabel_model(multilabel_model, 'Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof', 'shape'], 'dice_loss')
# evaluate_edns(saved_ckpt=footprint_model, 'Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint'], 'dice_loss')

# Training on other datasets
# *******************************
# Training on Massachusetts Building dataset
# train_single_class_png_labels('Unet', 'tu-maxvit_base_tf_512', 'Massachusetts', ['building'], 2, 50, 0.0001, 'Adam', 'dice_loss')

# Training on WHU Building dataset
# train_single_class_png_labels('Unet', 'tu-maxvit_base_tf_512', 'WHU', ['building'], 2, 50, 0.0001, 'Adam', 'dice_loss')




