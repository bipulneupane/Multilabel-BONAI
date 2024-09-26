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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def polygons_to_mask(shape, polygons, category):
    """
    Convert a set of polygons into a binary mask.

    Args:
        shape (tuple): The shape of the output mask (height, width).
        polygons (list): A list of polygons, where each polygon is a dictionary containing polygon vertices for a specific category.
        category (str): The key in each polygon dictionary that corresponds to the category for which the mask is being generated.

    Returns:
        numpy.ndarray: A binary mask (2D array) with 1s inside the polygons for the given category, and 0s outside.
    
    Example:
        shape = (1024, 1024)
        polygons = [{'building': [(x1, y1), (x2, y2), (x3, y3)]}, {'building': [(x4, y4), (x5, y5), (x6, y6)]}]
        mask = polygons_to_mask(shape, polygons, 'building')
    """
    # Create an empty mask with the given shape
    mask = Image.new('L', (shape[1], shape[0]), 0)
    
    # Iterate through the list of polygons
    for polygon in polygons:
        # Check if the current polygon contains the desired category
        if category in polygon:
            # Draw the polygon on the mask
            ImageDraw.Draw(mask).polygon(polygon[category], outline=1, fill=1)
    
    # Convert the mask to a NumPy array and return
    return np.array(mask)


class SingleClassPNGLabelDataset(Dataset):
    """
    A PyTorch Dataset class for handling image and mask pairs for a single class semantic segmentation task.
    This dataset supports image resizing, augmentations, and preprocessing for both images and masks.
    
    Attributes:
        CLASSES (list): A list of class names. In this case, it only contains 'building'.
        ids (list): A list of image filenames in the specified image directory.
        images_fps (list): Full paths to images.
        masks_fps (list): Full paths to corresponding masks.
        class_values (list): List of class indices corresponding to the selected classes.
        augmentation (albumentations.Compose): Optional augmentations to apply to the images and masks.
        preprocessing (albumentations.Compose): Preprocessing pipeline for resizing and other transformations.
        image_size (tuple): The target size (height, width) for resizing the images and masks.
    
    Args:
        images_dir (str): Directory containing input images.
        masks_dir (str): Directory containing corresponding masks.
        classes (list): List of class names to extract from the masks.
        augmentation (albumentations.Compose, optional): Augmentation pipeline to apply.
        preprocessing (albumentations.Compose, optional): Preprocessing pipeline to apply.
        image_size (tuple, optional): Size to resize images and masks to (default: (1024, 1024)).
    """
    
    CLASSES = ['building']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            image_size=(1024, 1024),  
    ):
        """
        Initializes the dataset with image and mask directories, class labels, augmentations, and preprocessing.

        Args:
            images_dir (str): Path to the directory containing the images.
            masks_dir (str): Path to the directory containing the masks.
            classes (list): List of class names to be extracted from the masks.
            augmentation (albumentations.Compose, optional): Augmentation pipeline to apply to images and masks.
            preprocessing (albumentations.Compose, optional): Preprocessing pipeline to apply to images and masks.
            image_size (tuple): Desired output image size (height, width).
        """
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Convert class names to class values (indices in CLASSES list)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.image_size = image_size
        
        # If no preprocessing is provided, default to resizing
        if self.preprocessing is None:
            self.preprocessing = albu.Compose([
                albu.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST),
                # Additional preprocessing steps can be added here
            ])
        else:
            # Insert resizing at the beginning of the existing preprocessing pipeline
            self.preprocessing.transforms.insert(
                0,
                albu.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST)
            )
    
    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask for a given index, applying augmentations and preprocessing.

        Args:
            i (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed image and mask.
        """
        # Read and process the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read the corresponding mask
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Extract specific classes from the mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Apply augmentations, if any
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing steps
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
        
    def __len__(self):
        """
        Returns the total number of image-mask pairs in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ids)



class SegmentationDatasetSingleClass(Dataset):
    """
    A PyTorch Dataset for loading images and corresponding single-class segmentation masks.
    This dataset reads images and their annotations, which are polygons in a JSON file, 
    and converts the polygons into binary masks for a specified class (e.g., 'footprint').

    Attributes:
        img_dir (str): Path to the directory containing the images.
        ann_dir (str): Path to the directory containing the annotations (JSON files).
        transform (callable, optional): Optional transform to be applied on the image.
        imgs (list): List of image filenames in the image directory.

    Args:
        img_dir (str): Directory path where the images are stored.
        ann_dir (str): Directory path where the annotation JSON files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        """
        Initializes the dataset by loading image paths and setting up transformations.

        Args:
            img_dir (str): Path to the directory containing the images.
            ann_dir (str): Path to the directory containing the annotations (JSON files).
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding binary mask for a given index.

        Args:
            idx (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: A tuple containing the transformed image (Tensor) and corresponding binary mask (Tensor).
        """
        # Construct the paths for the image and its corresponding annotation
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))
        
        # Load the annotation file (JSON)
        with open(ann_path) as f:
            anns = json.load(f)
        
        # Load the image and convert it to RGB
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Generate the binary mask from the polygons for the specified category ('footprint')
        mask = polygons_to_mask((height, width), anns['annotations'], 'footprint').astype(np.float32)
        
        # Add a channel dimension to the mask
        mask = np.expand_dims(mask, axis=0)
        
        # Apply transformations to the image, if any
        if self.transform:
            image = self.transform(image)
        
        # Convert the mask to a PyTorch tensor
        mask = torch.from_numpy(mask).float()

        return image, mask


class SegmentationDatasetMultiClass(Dataset):
    """
    A PyTorch Dataset for loading images and corresponding multi-class segmentation masks.
    This dataset reads images and their annotations (as polygons in a JSON file) and converts the polygons
    into binary masks for multiple categories such as 'footprint', 'roof', and 'shape'.

    Attributes:
        img_dir (str): Path to the directory containing the images.
        ann_dir (str): Path to the directory containing the annotation JSON files.
        transform (callable, optional): Optional transform to be applied on the image.
        imgs (list): List of image filenames in the image directory.

    Args:
        img_dir (str): Directory path where the images are stored.
        ann_dir (str): Directory path where the annotation JSON files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        """
        Initializes the dataset by loading image paths, setting up transformations, and preparing image list.

        Args:
            img_dir (str): Path to the directory containing the images.
            ann_dir (str): Path to the directory containing the annotations (JSON files).
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding multi-class binary masks for a given index.

        Args:
            idx (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: A tuple containing the transformed image (Tensor) and corresponding multi-class masks (Tensor).
        """
        # Construct the paths for the image and its corresponding annotation
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))
        
        # Load the annotation file (JSON)
        with open(ann_path) as f:
            anns = json.load(f)
        
        # Load the image and convert it to RGB
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Generate binary masks for each category ('footprint', 'roof', 'shape')
        masks = {}
        for category in ['footprint', 'roof', 'shape']:
            masks[category] = polygons_to_mask((height, width), anns['annotations'], category)

        # Stack the masks along the first dimension (channels)
        masks = np.stack([masks['footprint'], masks['roof'], masks['shape']], axis=0)
        
        # Apply transformations to the image, if any
        if self.transform:
            image = self.transform(image)
        
        # Convert the masks to a PyTorch tensor
        masks = torch.from_numpy(masks).float()
        
        return image, masks


class SegmentationDatasetFootprintRoof(Dataset):
    """
    A PyTorch Dataset for loading images and corresponding binary segmentation masks
    for two classes: 'footprint' and 'roof'. The dataset reads images and their annotations,
    which are polygons in a JSON file, and converts these polygons into binary masks.

    Attributes:
        img_dir (str): Path to the directory containing the images.
        ann_dir (str): Path to the directory containing the annotation JSON files.
        transform (callable, optional): Optional transform to be applied on the image.
        imgs (list): List of image filenames in the image directory.

    Args:
        img_dir (str): Directory path where the images are stored.
        ann_dir (str): Directory path where the annotation JSON files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        """
        Initializes the dataset by loading image paths, setting up transformations, and preparing the image list.

        Args:
            img_dir (str): Path to the directory containing the images.
            ann_dir (str): Path to the directory containing the annotations (JSON files).
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding binary masks for 'footprint' and 'roof' for a given index.

        Args:
            idx (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: A tuple containing the transformed image (Tensor) and corresponding multi-class masks (Tensor).
        """
        # Construct the paths for the image and its corresponding annotation
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.png', '.json').replace('.jpg', '.json'))
        
        # Load the annotation file (JSON)
        with open(ann_path) as f:
            anns = json.load(f)
        
        # Load the image and convert it to RGB
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Generate binary masks for 'footprint' and 'roof'
        masks = {}
        for category in ['footprint', 'roof']:
            masks[category] = polygons_to_mask((height, width), anns['annotations'], category)

        # Stack the footprint and roof masks into separate channels
        masks = np.stack([masks['footprint'], masks['roof']], axis=0)
        
        # Apply transformations to the image, if any
        if self.transform:
            image = self.transform(image)
        
        # Convert the masks to a PyTorch tensor
        masks = torch.from_numpy(masks).float()
        
        return image, masks


def get_model_from_smp(model_name, enc_name, enc_weight, num_class, act):
    """
    Fetches a segmentation model from the Segmentation Models PyTorch (SMP) library based on the specified model name.

    Args:
        model_name (str): The name of the model architecture to use. 
                          Available options include 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 'Linknet', 'MAnet', 'PAN', 'PSPNet', 'Unet', and 'UnetPlusPlus'.
        enc_name (str): The name of the encoder architecture (e.g., 'resnet34', 'efficientnet-b0', etc.).
        enc_weight (str or None): Pre-trained weights for the encoder (e.g., 'imagenet') or None for random initialization.
        num_class (int): The number of output classes for segmentation.
        act (str or callable): The activation function to apply to the final layer (e.g., 'sigmoid', 'softmax').

    Returns:
        torch.nn.Module: A segmentation model from SMP initialized with the given encoder, weights, number of classes, and activation.
    
    Example:
        model = get_model_from_smp('Unet', 'resnet34', 'imagenet', 3, 'softmax')
    
    Raises:
        ValueError: If an invalid model name is passed.
    """
    print('Model: ', model_name)
    
    model_map = {
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'FPN': smp.FPN,
        'Linknet': smp.Linknet,
        'MAnet': smp.MAnet,
        'PAN': smp.PAN,
        'PSPNet': smp.PSPNet,
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus
    }

    if model_name in model_map:
        return_model = model_map[model_name](
            encoder_name=enc_name, 
            encoder_weights=enc_weight, 
            classes=num_class, 
            activation=act
        )
    else:
        raise ValueError(f"Model name '{model_name}' is not valid. Please choose from {list(model_map.keys())}.")
    
    return return_model


def get_optim(opt, model_params, LR):
    """
    Returns a specified optimizer from PyTorch for the given model parameters and learning rate (LR).

    Args:
        opt (str): The name of the optimizer to use. 
                   Options include 'Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', and 'NAdam'.
        model_params (iterable): The parameters of the model to be optimized.
        LR (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer based on the specified name and configuration.
    
    Example:
        optimizer = get_optim('Adam', model.parameters(), 0.001)
    
    Raises:
        ValueError: If an invalid optimizer name is provided.
    """
    print('Optimizer: ', opt)
    
    optimizer_map = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adamax': torch.optim.Adamax,
        'NAdam': torch.optim.NAdam
    }

    if opt in optimizer_map:
        optim = optimizer_map[opt]([dict(params=model_params, lr=LR)])
    else:
        raise ValueError(f"Optimizer name '{opt}' is not valid. Please choose from {list(optimizer_map.keys())}.")

    return optim


def get_loss(loss_func):
    """
    Returns the specified loss function based on the provided loss function name.

    Args:
        loss_func (str): The name of the loss function to use. 
                         Options include 'jaccard_loss', 'dice_loss', 'binary_crossentropy', 'bce_dice_loss', and 'bce_jaccard_loss'.

    Returns:
        smp.utils.losses: The initialized loss function based on the specified name.
    
    Example:
        loss_function = get_loss('bce_dice_loss')
    
    Raises:
        ValueError: If an invalid loss function name is provided.
    """
    print('Loss Function: ', loss_func)
    
    # Define available loss functions
    jaccard_loss = smp.utils.losses.JaccardLoss()
    dice_loss = smp.utils.losses.DiceLoss()
    binary_crossentropy = smp.utils.losses.BCELoss()
    
    # Combined loss functions
    bce_dice_loss = binary_crossentropy + dice_loss
    bce_jaccard_loss = binary_crossentropy + jaccard_loss

    # Mapping loss function names to actual implementations
    loss_map = {
        'jaccard_loss': jaccard_loss,
        'dice_loss': dice_loss,
        'binary_crossentropy': binary_crossentropy,
        'bce_dice_loss': bce_dice_loss,
        'bce_jaccard_loss': bce_jaccard_loss
    }

    # Return the corresponding loss function, or raise an error for invalid names
    if loss_func in loss_map:
        return_lf = loss_map[loss_func]
    else:
        raise ValueError(f"Loss function name '{loss_func}' is not valid. Please choose from {list(loss_map.keys())}.")

    return return_lf


def get_multilabel_loss(loss_func):
    """
    Returns the specified multilabel loss function based on the provided loss function name.
    This function supports the Jaccard and Dice losses for multilabel segmentation tasks.

    Args:
        loss_func (str): The name of the multilabel loss function to use. 
                         Options include 'jaccard_loss' and 'dice_loss'.

    Returns:
        smp.losses: The initialized multilabel loss function.
    
    Example:
        loss_function = get_multilabel_loss('jaccard_loss')
    
    Raises:
        ValueError: If an invalid loss function name is provided.
    """
    print('Loss Function: ', loss_func)
    
    # Define multilabel loss functions from the SMP library
    jaccard_loss = smp.losses.JaccardLoss('multilabel')
    dice_loss = smp.losses.DiceLoss('multilabel')

    # Mapping loss function names to actual implementations
    loss_map = {
        'jaccard_loss': jaccard_loss,
        'dice_loss': dice_loss
    }

    # Return the corresponding loss function, or raise an error for invalid names
    if loss_func in loss_map:
        return loss_map[loss_func]
    else:
        raise ValueError(f"Loss function name '{loss_func}' is not valid. Please choose from {list(loss_map.keys())}.")
