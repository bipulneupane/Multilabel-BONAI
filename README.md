# Multilabel-BONAI

Welcome to **Multilabel-BONAI**, a comprehensive repository that provides code for data preparation, segmentation model development, and post-processing for the **multi-label segmentation** of buildings' **roof, footprint**, and **shape** on off-nadir aerial images.

This repository is built around the **BONAI dataset** and supports training and evaluation of segmentation models for single and multi-label tasks. Additionally, it provides tools for post-processing and visualization of model predictions.

## Example Usage

### Training a Model

#### Single Label/Class Segmentation

```python
train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-footprint-')
train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['roof'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-roof-')
train_unet_singleclass('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['shape'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-single-shape-')
```

#### Multi-Label Segmentation

```python
train_unet_encoders_multilabel('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof', 'shape'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-multilabel-')
```

#### Roof and Footprint Segmentation (Without Shape)

```python
train_multilabel_foot_and_roof('Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof'], 2, 50, 0.0001, 'Adam', 'dice_loss', './trained_models/MultiTask/'+'BONAI-multilabel-noshape-')
```

### Evaluating a Model

```python
multilabel_model = "./trained_models/BONAI-multilabel-Unet-tu-maxvit_base_tf_512-BONAI-shape-50ep-Adam-dice_loss.pth"
footprint_model = "./trained_models/MultiTask/Umaxvit/BONAI-single-footprint-Unet-tu-maxvit_base_tf_512-BONAI-shape-200ep-Adam-dice_loss.pth"

evaluate_multilabel_model(multilabel_model, 'Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint', 'roof', 'shape'], 'dice_loss')
evaluate_edns(saved_ckpt=footprint_model, 'Unet', 'tu-maxvit_base_tf_512', 'BONAI-shape', ['footprint'], 'dice_loss')
```

### Training on Other Datasets

#### Massachusetts Building Dataset

```python
train_single_class_png_labels('Unet', 'tu-maxvit_base_tf_512', 'Massachusetts', ['building'], 2, 50, 0.0001, 'Adam', 'dice_loss')
```

#### WHU Building Dataset

```python
train_single_class_png_labels('Unet', 'tu-maxvit_base_tf_512', 'WHU', ['building'], 2, 50, 0.0001, 'Adam', 'dice_loss')
```

### Shape Calculation from Roof and Footprint Labels

```bash
python data_preparation.py
```

### Post-Processing the Footprint Outputs

```bash
python post_processing.py
```

### Predictions and Visualization

Refer to `prediction.py` for examples of how to perform predictions and visualize the results.

---

## Required Python Libraries and Packages

The following Python packages are required to run the code in this repository:

```
os
torch
numpy
pillow
torchvision
cv2
matplotlib
json
shapely
segmentation_models_pytorch (https://github.com/qubvel-org/segmentation_models.pytorch)
albumentations (https://pypi.org/project/albumentations/0.0.10/)
```

---

## Collect the BONAI Dataset

The BONAI dataset, used extensively in this repository, can be downloaded from:

[https://github.com/jwwangchn/BONAI](https://github.com/jwwangchn/BONAI)

---

## Credit and Citation

This work is currently under revision in an IEEE journal. The appropriate citation will be provided once the paper is published.

---

## Acknowledgements

The authors would like to acknowledge the creators of the **BONAI dataset** – the only dataset with annotations for roof, footprint, and offset segmentation. This dataset has been invaluable for advancing research in off-nadir aerial image segmentation.

---

This README file introduces your project and provides comprehensive usage instructions, making it easier for others to understand and utilize your repository.
