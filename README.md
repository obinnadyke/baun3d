# BAUN3D: Boundary-Attentive 3D-UNet for Auto-Segmentation of Tumor-Prone Organs in Medical CT Volumes

---

## Overview 

BAUN3D (Boundary-Attentive 3D U-Net) is a unique deep learning-base radiomics framework built for the segmentation of organs and tumors in volumetric CT. Developed specifically for challenging tumor-prone organ segmentation tasks, BAUN3D is implemented with deformable cross attention mechanisms, gated boundary refinement module, and a composite loss objective for handling curriculum learning, extreme class imbalance, small tumor targets, and contour structural continuity.

---

## System requirements
```
- Python ≥ 3.8
- CUDA ≥ 11.8 (for GPU acceleration)
- 10GB+ GPU memory per GPU
```

## Dataset

### Data Directory Structure
```
data/
├── lits/
│   ├── imagesTr/          # Training images (*.nii.gz)
│   ├── labelsTr/          # Training labels (*.nii.gz)
│   └── imagesTs/          # Test images
├── pancreas/
│   ├── imagesTr/
│   ├── labelsTr/
│   └── imagesTs/
└── ...
```
---

## Train | Test | Inference 
Training/inference scripts and command-line will be publicly released soon. 

## Results 

### Numeric outcomes

```
| Dataset  | Organ Dice | Tumor Dice | Avg Dice | HD95 (mm)   |
|----------|------------|------------|----------|-------------|
| LiTS     | 0.96       | 0.74       | 0.84     | 5.82        |
| Pancreas | 0.84       | 0.81       | 0.83     | 4.83        |
```

### Qualitative outcomes

<div align="center">
  <img src="docs/liver_sag.png" alt="Sagittal view1" width="850"/>
  <p><em>Boundary segmentation sample for Liver/Tumor</em></p>
</div>
<div align="center">
  <img src="docs/pancreas_sag.png" alt="Sagittal view2" width="850"/>
  <p><em>Boundary segmentation sample for Pancreas/Tumor</em></p>
</div>
