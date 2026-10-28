# Industrial Surface Defect Detection Using Deep Learning

## Project Overview
This project focuses on identifying and segmenting defects in industrial surface images using deep learning and computer vision. The goal is to detect defect regions at the pixel level and classify them into four defect categories.

The project is based on the **Industrial Surface Defect Detection** Kaggle competition and is part of the IIT Madras BS Degree Diploma-level Deep Learning and Generative AI project.

## Objectives
- Analyze the industrial surface defect dataset.
- Explore defect distributions and segmentation masks.
- Preprocess images and decode Run-Length Encoded (RLE) masks.
- Build a deep learning segmentation model.
- Evaluate predictions using Dice score and the competition's official scoring metric.
- Compare different probability thresholds for segmentation.

## Dataset
The dataset contains industrial surface images with pixel-level defect annotations.

**Dataset structure:**
```text
public/
├── train.csv
├── sample_submission.csv
├── train/
└── test/
```

**Dataset details:**
- Training images: 11,393
- Test images: 3,000
- Image dimensions: 800 × 128 pixels
- Number of defect classes: 4
- Annotation format: Run-Length Encoding (RLE)

### Defect Classes
- `defect_a`
- `defect_b`
- `defect_c`
- `defect_d`

Dataset source: [Industrial Surface Defect Detection — Kaggle](https://www.kaggle.com/competitions/industrial-surface-defect-detection)

## Methodology

### 1. Data Exploration
- Inspected the dataset structure and annotation columns.
- Analyzed the number of images containing each defect class.
- Investigated multi-class images and pixel-level mask overlaps.
- Examined class imbalance and defect distribution.

### 2. Data Preprocessing
- Loaded images using OpenCV.
- Converted images from BGR to RGB format.
- Resized images to 800 × 128 pixels.
- Decoded RLE annotations into binary segmentation masks.
- Prepared four-channel masks corresponding to the four defect classes.
- Split the dataset into training and validation subsets.

### 3. Deep Learning Model
A U-Net-based segmentation model is used as the baseline architecture.

- **Input:** RGB industrial surface image.
- **Output:** Four probability maps, one for each defect class.
- **Activation:** Sigmoid.
- **Loss function:** Binary cross-entropy.
- **Framework:** TensorFlow/Keras.

### 4. Evaluation
The model is evaluated using:
- Dice score for segmentation quality.
- Official competition score.
- Threshold comparison at 0.3, 0.5
