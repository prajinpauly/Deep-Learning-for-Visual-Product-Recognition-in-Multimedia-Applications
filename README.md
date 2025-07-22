# Deep Learning for Visual Product Recognition in Multimedia Applications

## Overview

This project focuses on building intelligent systems that can automatically identify and classify products in images using deep learning techniques. The models were trained and evaluated on a large and diverse product image dataset, tackling real-world challenges often seen in e-commerce, retail analytics, and warehouse management.

By implementing and comparing multiple architectures — including CNNs, ResNet50, and Vision Transformers — the goal was to achieve accurate recognition despite issues like poor image quality, class imbalance, and large-scale data volume.

---

## Objectives

- Develop deep learning models capable of identifying and classifying products based on customer-submitted images.
- Compare different architectures to determine which performs best under realistic data conditions.
- Apply techniques to improve generalization, reduce overfitting, and handle class imbalance.

---

## Real-World Relevance

Product recognition plays a critical role in modern multimedia applications:

- **E-commerce**: Automate the tagging and categorization process for faster and more accurate listings.
- **Retail Analytics**: Monitor shelf stock and optimize restocking without manual input.
- **Inventory Management**: Recognize and track products in warehouses to improve operational efficiency.
- **Counterfeit Detection**: Identify fake or low-quality items based on visual cues.

---

## Dataset: Products-10K

A large-scale open-source dataset containing:

- **~142,000 training images**
- **~55,000 test images**
- **Over 9,600 product categories**
- **Total size**: ~18GB

### Key Challenges

- **Class Imbalance**: Some categories had only 2–3 images, while others had 80+.
- **Image Quality**: Varying clarity, lighting, backgrounds, and occlusions impacted model learning.
- **High Computation Needs**: Training required GPU acceleration due to dataset volume and deep model complexity.

---

## Preprocessing Workflow

To ensure quality model training, the following steps were implemented:

- **Image Resizing**: Standardized image dimensions to 224×224 pixels.
- **Normalization**: Scaled pixel values between 0–1 to stabilize training.
- **Data Augmentation**: Introduced flipping, rotation, and color jitter to enhance diversity.
- **Imbalance Handling**:
  - Oversampling rare classes
  - Using class-weighted loss functions for fair learning
- **Train/Validation Split**: 80:20 ratio to balance training performance with reliable evaluation.

---

## Model Architectures

### 1. CNN with ResNet50 (Transfer Learning)

- Made use of a pre-trained ResNet50 model as a strong feature extractor.
- Added custom classification layers for the product dataset.
- Integrated dropout and early stopping to reduce overfitting.
- Applied weighted cross-entropy loss to handle class imbalance.

### 2. Vision Transformer (ViT)

- Processed image patches using self-attention instead of convolutions.
- Captured global image context more effectively for complex classifications.
- Customized classification head suited for the large number of categories.

### 3. Custom CNN with Augmentation & Oversampling

- Designed a lightweight 3-layer CNN from scratch with filters 12, 20, and 32.
- Applied batch normalization and dropout for performance tuning.
- Achieved the best overall results on training and test data.

---

## Evaluation Results

| Architecture           | Training Accuracy | Validation Accuracy | Test Accuracy |
|------------------------|-------------------|---------------------|----------------|
| CNN + ResNet50         | 5.02%             | 6.54%               | 4.99%          |
| Vision Transformer     | 9.26%             | 7.47%               | 4.35%          |
| Custom CNN             | **86.00%**        | **30.50%**          | **29.60%**     |

### Observations

- The **custom CNN** performed significantly better in both training and evaluation.
- ResNet50 and ViT architectures struggled with the high class imbalance and noisy data.
- Data augmentation and oversampling were critical for boosting performance on smaller classes.

---

## Key Insights

- Image quality and class balance play a major role in real-world model performance.
- Smaller, well-optimized models can outshine complex architectures when data is limited or noisy.
- CUDA-enabled GPUs and careful memory management were essential due to the dataset's scale.

---

## Future Work

- **CNN + LSTM**: Integrate temporal modeling for video or sequence-based recognition tasks.
- **Attention-based CNNs**: Use attention layers to focus on the most influential image regions.
- **Model Deployment**: Explore deploying the best-performing model as an API or web tool.

---

## Author

**Prajin Paul**  
📧 Email: [paulprajin2015@gmail.com](mailto:paulprajin2015@gmail.com)  
🔗 LinkedIn: [https://www.linkedin.com/in/prajin-paul-b64415247](https://www.linkedin.com/in/prajin-paul-b64415247)
