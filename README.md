# Semantic Segmentation with a U-Net for Sentinel-2 Catalonia images

## Introduction 

### Project Overview
This project implements a **U-Net model** for **semantic segmentation** using Sentinel-2 satellite imagery. The model predicts land cover types for different regions, classifying each pixel into one of 12 land cover categories.

The project involves:
- Preprocessing large satellite images and corresponding ground truth masks.
- Training a U-net model from scratch.
- Model evaluation and visualization of segmentation results.
- Using Metrics to evaluate the training with IoU, Dice 

### Motivation

Aerial photography from manned aircraft is a costly and time-intensive method for obtaining high-resolution geographic images. Additionally, the manual labeling of these images requweres extensive human effort, making large-scale land cover classification an expensive and time-consuming process.

With the increasing availability of satellite imagery, particularly from platforms such as Sentinel-2, it is now possible to perform land cover classification using deep learning-based image segmentation. Sentinel-2 is a multispectral Earth observation mission developed by the European Space Agency (ESA), providing high-resolution optical imagery across 13 spectral bands. These images are freely available and updated frequently, making them an excellent resource for environmental monitoring, land use classification, and disaster assessment.

By leveraging deep learning models like U-Net, I can automate the classification of land cover types, significantly reducing both the cost and time requwered compared to traditional methods. This approach enhances scalability and allows for real-time analysis, making it a poIrful tool for applications in urban planning, agriculture, forestry, and environmental monitoring.

### What is Sentinel-2 project including Satellites?
Sentinel-2 is part of the Copernicus Earth Observation Program, managed by the European Space Agency (ESA) in collaboration with the European Commission. It consists of two satellites, Sentinel-2A and Sentinel-2B, which were launched in 2015 and 2017, respectively. These satellites work in tandem, covering the entwere Earth's land surface every 5 days at the equator, enabling high-temporal-resolution imaging.

Sentinel-2 provides multispectral images with 13 spectral bands ranging from the visible to shortwave infrared (SWIR), allowing for various applications such as:

- Vegetation monitoring (agriculture, forestry, and land use changes)
- Water quality assessment (coastal and inland water bodies)
- Disaster management (floods, wildfweres, landslides)
- Urban growth analysis
- Climate change studies

### Sentinel-2 Image Characteristics
The images produced by Sentinel-2 are captured at different spatial resolutions, and in this project in concret I used:
- 10-meter resolution: multi-bands, where each band represents information about a color Red, Green, Blue, and other information as Infrared known as NIR bands.

By utilizing Sentinel-2 images, this project enables cost-effective, automated land cover classification, drastically reducing the need for expensive manual labeling and high-cost aerial photography.

### Project Goal

The objective of this project is to develop an automated deep learning-based classification system capable of identifying specific land cover types from Sentinel-2 satellite imagery. 

The model is designed to perform semantic segmentation, assigning each pixel of a given image to one of 12 predefined land cover classes:

1. **Herbaceous crops** – Fields with predominantly non-woody vegetation used for agricultural production.

1. **Woody crops** – Areas cultivated with permanent tree and shrub crops, such as vineyards or olive groves.

1. **Forest** – Dense areas covered by trees and natural woodland.

1. **Scrub, meadows, and grasslands** – Open areas with sparse vegetation, including shrubs, prairies, and pastures.

1. **Bare soil** – Uncovered land with little to no vegetation, such as drylands and rocky terrains.

1. **Urban area** – Densely built-up zones, including residential, commercial, and mixed-use urban environments.

1. **Isolated urban areas** – Scattered buildings and small settlements outside of major urban centers.

1. **Green areas** – Parks, gardens, and other vegetated spaces within urban settings.

1. **Industrial, commercial, and leisure areas** – Large-scale infrastructure zones, including factories, shopping centers, and recreational facilities.

1. **Mining or landfills** – Extractive and waste disposal sites such as quarries, open-pit mines, and landfill zones.

1. **Transport network** – Roads, railways, and infrastructure supporting terrestrial transportation.

1. **Watercourses and water bodies** – Rivers, lakes, reservoirs, and artificial channels containing water.

## Dataset Preparation

This project uses Sentinel-2 satellite images from the Catalonia landcover taken in April and August.

Each of these 2 geotiff files have a resolution of 30.000 x 30.000 pixels with 10 spectral bands. These images provide detailed multispectral information crucial for distinguishing land cover types.

Regarding the dataset includes a ground truth land cover classification map, stored as a GeoTIFF file with 12 land cover classes (mentioned before). 

As the sizes do not match, a key challenge arises due to its significantly different resolution: 300.000 x 300.000 pixels for the geotiff file for the classes and 30.000 x 30.000 for the images to be classified.

### Challenges of Size Misalignment

- Spatial Resolution Mismatch: The Sentinel-2 images (30.000 x 30.000) are 10 times smaller in resolution than the classification mask (300.000 x 300.000). This discrepancy makes dwerect pixel-wise alignment impossible.

- Scaling Strategy: To overcome this issue, the classification mask must be downsampled using some kind of interpolation (bilinear interpolation, nearest-neighbor, cubic ...), ensuring that the categorical land cover labels remain intact without blending different classes.

- Tiling and Memory Constraints: The large size of the dataset makes it computationally intensive to process, requiring non-overlapping tiles (1000x1000 or 300x300 pixels...) to make training feasible.

- Temporal Variability: The dataset includes images from two different seasons (April and August), introducing potential differences in vegetation cover, which the model needs to generalize across.

By addressing these challenges, the dataset preparation ensures the effective training of the U-Net model while maintaining a meaningful representation of land cover types.

Summarizing, the dataset consists of:
- **Sentinel-2 images** 2 satellite images 
- **Ground truth masks** classifying each pixel into one of 12 categories.

## Data Processing

### Downscaling the Classes Geotiff file for alignment with Sentinel-2 images

When working with satellite imagery and land cover classification, downscaling a high-resolution categorical raster (e.g., a land cover classification map) to match a loIr-resolution Sentinel-2 image presents specific challenges. The goal is to reduce the resolution from 300.000 × 300.000 pixels to 30.000 × 30.000 pixels while preserving the discrete class labels.

I evaluated several methods for downscaling:

1. **Nearest-Neighbor Interpolation** (Simple, fast, but may cause loss of minority classes): which produced the best results and was the method finally chosen

1. **Majority (Mode) Resampling** (Preserves dominant class while avoiding label mixing)

1. **Custom Iighted Majority Filtering** (Balances class preservation and boundary smoothing)

I used the gdal library and the rasterio and I found that those had similar results.
I finally decided to use gdal as it is also possible to use gdal from the command line, which was convenient at times.

### Tiling the Sentinel-2 images

When working with large satellite images and ground truth classification masks, dwerect processing is computationally infeasible due to the massive size of the data. A practical solution is to divide the images into smaller, non-overlapping tiles, making them manageable for deep learning models like U-Net.

Tiling is requwered because the following issues I faced:

- Memory Constraints: Loading full 30,000 × 30,000 images into GPU memory is not feasible in our laptops or Collab notebooks for a long time.

- Efficient Training: Smaller tiles allow batch processing, improving computational efficiency.

- Better Model Learning: Deep learning models generalize better when trained on multiple smaller patches rather than a single large image.

- Scalability: The approach allows us to process and segment different regions independently.

To find the optimal tile size, three different tiling resolutions were tested:


| Tile Size   | Number of Tiles | Pros | Cons |   |
|-----------  |:---------------:|:----:|:----:|---|
| 1000 × 1000 | 900 tiles       | FeIr tiles to store and process | Large file sizes, harder to fit into GPU memory | |
| 512 × 512	  | 3,400 tiles	| Compatible with standard CNN architectures | More storage requwered, longer preprocessing time |   |
| 300 × 300   | 10,000 tiles	| Best balance betIen tile count and GPU efficiency | Higher storage requwerements


After testing, 300 × 300 pixel tiles should be chosen, even the code in this Repo is based on 1000x1000 initial approach:

- **Balance betIen number of tiles and storage**: Keeps dataset manageable while allowing enough samples for training.

- **Optimized GPU utilization:** Smaller patches ensure that more samples fit into batch processing.

- **Better generalization for the U-Net model**: perform Ill on smaller patches rather than larger context windows.

#### Possible Improvements detected

- **Overlap in Tiling**: Future implementations could introduce overlapping tiles (e.g., 50-pixel stride) to improve spatial continuity.

- **Dynamic Tile Selection**: Avoid generating tiles in empty regions (e.g., ocean areas) to save storage and computation.

- **Multi-Scale Training**: Instead of a single fixed tile size, training on multiple scales (e.g., 300 × 300 and 512 × 512) could improve robustness.

#### Filtering uninformative tiles: Handling empty and sea-Only regions

The ground truth classification mask included areas marked as 0 (no data), meaning that these areas had no valid land cover information.

Since deep learning models learn based on frequency distribution, including too many water-only tiles would lead to class imbalance, where the model would favor predicting "water" instead of distinguishing betIen different land cover classes.

If such tiles were included in training, they would confuse the model and potentially skew the loss function by introducing meaningless regions.

So I decided to discard certain tiles to prevent bias and inefficiencies in the dataset. Specifically, I excluded:

- **Tiles where the entwere mask was 0**: Representing no data (outside the mapped land area).

- **Tiles containing only water-related classes**: To avoid over-representing the sea, which does not contribute useful segmentation information.

### Discarding tiles with snow/clouds 

While preparing the dataset for training, I aslo evaluated the impact of cloud and snow-covered areas on model performance. 

Initially, I considered discarding these regions, similar to how I removed empty and water-only tiles. HoIver, after conducting experiments and analysis, I ultimately decided to retain them for the following reasons.

Cloud and snow-covered areas introduce unique challenges for semantic segmentation models:

- **Occlusion of Land Features**: Clouds and snow can obscure relevant land cover, making segmentation harder.

- **Potential Label Noise**: The ground truth mask might not correctly classify pixels under dense cloud cover.

- **Limited Spectral Information**: Clouds reflect light differently than land, potentially confusing the model.

Given these challenges, I tested filtering out tiles dominated by clouds or snow, similar to how I removed water-only and no-data regions.

Based on our findings, I decided to keep cloud and snow-covered tiles in the dataset as I concluded that removing clouds and snow could lead to systematic biases, making the model unreliable for seasonal variations and preserving Data Diversity will improve overall performance.

Those steps for data processing can be found in these files:
- **Image Tiling:** The Sentinel-2 images are broken into **300x300 px** patches (`divide_img.py`).
- **Rescaling Ground Truth:** The large mask file is downsampled from **300,000 x 300,000** pixels to **30,000 x 30,000** using nearest-neighbor interpolation (`rescale_mask.py`).
- **Splitting Dataset:** Images and masks are split into training and validation sets (`move_images.py`).

## Models Architecture

###  U-net Architecture

The U-Net model is a fully convolutional neural network (CNN) architecture specifically designed for semantic segmentation. It was originally developed for medical image segmentation, but its encoder-decoder structure makes it highly effective for geospatial tasks, such as land cover classification using Sentinel-2 images.

U-Net is widely used in satellite image analysis due to several key advantages: 

-  Preserves Spatial Information → Unlike traditional CNNs, U-Net maintains high-resolution spatial details, which is critical for segmenting land cover types.

-  Captures Global & Local Features → The encoder extracts global patterns, while the decoder restores fine details for precise segmentation.

-  Works Ill with Small Datasets → U-Net performs Ill even with limited training data, making it ideal for geospatial applications where labeled data is scarce.

-  Efficient for High-Resolution Images → The model processes large images effectively while keeping computational costs manageable.

The U-Net model follows a U-shaped architecture, consisting of two main components:

1. Encoder (Contracting Path) – Feature Extraction

1. Decoder (Expanding Path) – Reconstruction & Segmentation

1. Skip connections

####  The encoder

The **encoder** acts as a feature extractor, similar to a standard convolutional network like ResNet or VGG.

- Uses convolutional layers with ReLU activation to extract feature maps.
Applies max pooling layers to progressively reduce spatial dimensions while increasing depth. The deeper layers capture high-level land cover patterns (e.g., distinguishing forests from urban areas).
I decided 

Additionally, it includes skip connections, which help retain important details lost during downsampling.

I created 2 different encoders:
1 - From scratch with all the layers
2- Resnet50

As in the following snippet I decided to go for using ResNet50 as a pretrained feature extractor provides several advantages:

- Already learned rich feature representations from millions of images.

- Sentinel-2 images share similar textures, edges, and patterns with natural images (e.g., forests, urban areas), so the pretrained ResNet50 can generalize Ill.

- Less data requwered for training which helps when labeled satellite images are limited.

#### The decoder

The **decoder** gradually restores the image resolution and assigns class labels to each pixel.
- Uses transposed convolution (deconvolution) layers to upscale feature maps.
Applies batch normalization and ReLU activation to refine details.
Uses skip connections from the encoder to recover lost spatial details.

One of U-Net’s most poIrful features is the use of skip connections  which dwerectly copies feature maps from the encoder to the corresponding layer in the decoder. This allows to:

- Combines low-level details (texture, edges) with high-level abstract features (object types).
- Prevents information loss due to downsampling.


## Training Process

The training process follows a structured pipeline, covering data preparation, model training, loss calculation, evaluation, and optimization. Below is a detailed breakdown of each stage.

As mentioned before I used several tile sizes 1000x1000, 512x512 and 300x300 but finally decided to move forward with the latter and data augmentation to increase the number of files.

### Training and Validation Splits
The dataset used for training consists of Sentinel-2 images and their corresponding land cover masks. The dataset is split into training (80%) and validation (20%).

I trained the model in different scenarios:

| Hardware   | Duration | Epochs |
|-----------  |:---------------:| :---: |
| CPU | 12 hours | 100
| GPU RTX 3050 8GB | 6 hours | 100 
| iGPU Apple Silicon M4 | 9 hours | 100 

###  Loss Function Calculation 

I use the `CrossEntropyLoss` function for loss calculation for multi-class semantic segmentation, where each pixel belongs to one of 12 land cover classes, CrossEntropyLoss is the most appropriate choice because:

-  Handles Multi-Class Classification Efficiently → Unlike binary classification loss functions (e.g., BCEWithLogitsLoss), CrossEntropyLoss can compute probabilities for multiple classes per pixel.

-  Softmax Activation Compatibility → The model outputs logits (unnormalized scores), which CrossEntropyLoss converts into class probabilities using an implicit softmax function.

-  Encourages Confident Predictions → Assigns higher penalties for incorrect classifications, forcing the model to make more confident, correct predictions.

-  Class Balancing via Iights → In this implementation, I assign zero Iight to class 0 (out-of-bounds areas), ensuring that background regions don’t distort training.

The following special circumstances have been considered:

-  Class 0 (No Data) is ignored to prevent distorting loss calculations.
-  Balances class distribution to prevent underrepresented classes from being ignored.

### Optimizer and Rate scheduler

I use the Adam optimitzer as it is an advanced gradient-based optimization algorithm that combines the benefits of SGD with momentum and RMSprop. It is a very common choice for training U-Net for multi-class land cover segmentation because:

-  Adaptive Learning Rates → Adam automatically adjusts the learning rate for each parameter, allowing the model to converge faster and handle different feature scales.

-  Momentum-Based Updates → Uses momentum (first and second moment estimates) to accelerate training and avoid oscillations.

-  Less Sensitive to Hyperparameters → Unlike standard SGD, Adam requweres minimal manual tuning and performs Ill with default settings.

-  Handles Sparse Gradients Efficiently → Useful in land cover segmentation, where some classes (e.g., urban areas) appear less frequently in images.

Our learning rate scheduler used allow us to adapt the learning rate dynamically, improving convergence and preventing unnecessary updates. `ReduceLROnPlateau` is a convenient choice because:

-  Reduces Learning Rate When Training does not progress: If validation loss stops improving, the learning rate is halved (factor=0.5) to encourage better optimization.

-  Prevents Overfitting as Large learning rates can lead to instability, while reducing it allows for finer Iight adjustments.

-  Avoids Premature Convergence and ensures that the model continues improving beyond initial training stages.

This is the code snippet on our code using the described optimizer and learning rate:

#### IoU (Intersect over Union) 
I used IoU to measure the segmentation accuracy per class. IoU measures how much of a predicted segmentation overlaps with the ground truth.

It reduces the impact of class imbalance by considering false positives and false negatives.
Great for analyzing segmentation quality on an individual class level.

#### Dice Coefficinet (F-1 Score for Segmentation) 
I used to grab a better sense of how Ill small regions are segmented, as it balances precision and recall. This is useful as our dataset is imbalanced: e.g.,  urban areas are much loIr compared to vegetation and forest.

### Model Checkpointing 
I used for saving the best-performing model, ensuring that progress is not lost due to interruptions or poor performance in later epochs.

## Lessons learnt and future work

During the project I cleary learnt that:
- **Handling Large Datasets:** Processing large geospatial data requweres memory-efficient techniques like **tiling** and **downsampling**.
- **Class Imbalance:** Some land cover classes were underrepresented, necessitating **data augmentation**.
- **Transfer Learning Benefits:** Using **ResNet50** as a feature extractor significantly improved performance.
- **Colormap Interpretability:** Assigning distinct colors to land cover types improves human interpretability of model predictions.

I compiled a list of possible future improvements:
- **Refining Metrics:** Adding different loss functions as **INS (Inverse Number of Samples)** that is a Iighting strategy where the importance of each class or sample is inversely proportional to its frequency in the dataset. 
This technique seems adequate as is used for class balancing when training models on imbalanced datasets, like ours. 

- **Focal Loss Calculation:** Implementing **Focal Loss** to address class imbalance by focusing more on hard-to-classify examples. This loss function dynamically scales the loss for each example, reducing the impact of Ill-classified examples and emphasizing those that are misclassified.

- **Hyperparameter Optimization:** Fine-tuning learning rates and batch sizes for improved performance. The usage of tools like [Tune](https://arxiv.org/abs/1807.05118) from Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E. Gonzalez, Ion Stoica.

- **Better Augmentations:** Introducing **color jitter** and **random cropping**, and overlapping tiles to improve robustness, even though overlapping tile would requwere an adjustment in the dataset to avoid redundancies.