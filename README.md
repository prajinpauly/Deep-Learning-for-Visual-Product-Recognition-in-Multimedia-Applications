
# Visual Product Recognition for E-commerce Catalog Automation

## Overview
This project focuses on building a deep learning-based system for automated product recognition and categorization in e-commerce. The goal is to identify and classify products from customer-uploaded images, addressing real-world challenges such as class imbalance, noisy backgrounds, and large-scale data. The solution is built using PyTorch and compares several architectures, including CNNs, ResNet50, and Vision Transformers, trained on the large-scale Products-10K dataset.

The aim is to evaluate how different neural models perform in classifying products for applications like automated cataloging, duplicate detection, and inventory management in online marketplaces.

---

## Objectives
- Develop and evaluate deep learning models for product image classification.
- Extract robust visual features from diverse, user-submitted product images.
- Compare multiple architectures (CNN, ResNet50, Vision Transformer) to understand their strengths.
- Address challenges in class imbalance and noisy data to improve model generalization.

---

## Applications and Relevance
Visual product recognition has many important real-world applications, including:

- **Automated Cataloging:** Instantly categorize new product listings, reducing manual data entry and human error.
- **Duplicate & Counterfeit Detection:** Identify duplicate or fake listings by visually matching products.
- **Personalized Search & Recommendations:** Enhance search accuracy and product discovery by leveraging visual similarity.
- **Inventory Management:** Track and organize products in warehouses and online stores.

---

## Dataset: Products-10K
- **Total Samples:** ~197,000 images (142,000 training, 55,000 test)
- **Categories:** 9,600+ product types
- **Image Size:** Standardized to 224×224 pixels
- **Data Size:** ~18GB
- **Class Distribution:** Highly imbalanced, with some categories having only a few images and others having 80+

---

## Preprocessing & Feature Extraction

**Main Steps:**
- Image Resizing: All images resized to 224×224 pixels.
- Normalization: Pixel values scaled to [0, 1].
- Data Augmentation: Flipping, rotation, and color jitter to simulate real-world diversity.
- Imbalance Handling: Oversampling rare classes and using class-weighted loss functions.
- Train-Test Split: 80% training / 20% testing data.

---

## Model Architectures

1. **CNN with ResNet50 (Transfer Learning)**
   - Pre-trained on ImageNet, fine-tuned for e-commerce product categories.
   - Custom classification layers, dropout, and class-weighted loss for imbalance.

2. **Vision Transformer (ViT)**
   - Processes image patches with self-attention, capturing global context for complex product images.

3. **Custom Lightweight CNN**
   - 3-layer architecture optimized for e-commerce data.
   - Batch normalization, dropout, and aggressive augmentation for generalization.

---

## Implementation: PyTorch

- **Image Processing:** Using torchvision and PIL.
- **Model Definition:**
  - `torchvision.models.resnet50` for transfer learning
  - Custom `torch.nn.Module` for CNN and ViT
- **Loss Function:** Weighted CrossEntropyLoss (multi-class classification)
- **Optimizer:** Adam, with learning rate set at 0.001
- **Training Duration:** 30 epochs with early stopping

---

## Training and Performance

| Model Variant      | Training Accuracy | Validation Accuracy | Test Accuracy | Notes                                      |
|--------------------|------------------|--------------------|--------------|---------------------------------------------|
| CNN + ResNet50     | 5.02%            | 6.54%              | 4.99%        | Struggled with class imbalance and noise    |
| Vision Transformer | 9.26%            | 7.47%              | 4.35%        | Sensitive to data quality and imbalance     |
| Custom CNN         | 86.00%           | 30.50%             | 29.60%       | Best overall accuracy and generalization    |

Each model was trained using GPU acceleration. Data augmentation and oversampling were critical for boosting performance, especially for rare product categories.

---

## Key Learnings and Observations

- Class imbalance and image quality are major challenges in real-world e-commerce data.
- Custom, lightweight CNNs outperformed more complex models when data was noisy or imbalanced.
- Data augmentation and class-weighted loss functions were essential for fair learning.
- Efficient memory management and GPU usage were necessary due to dataset scale.

---

## Future Improvements

To further improve the system, future work can explore:

- Attention-based CNNs or transformers for better focus on key product features.
- Advanced augmentation and synthetic data generation to balance rare classes.
- Sequence modeling (e.g., CNN + LSTM) for video-based product listings.
- Deployment as a REST API for integration with e-commerce platforms.

---

## Conclusion

This project successfully implemented and compared multiple deep learning models for product image classification using the Products-10K dataset. Results demonstrated:

- Custom CNN achieved the highest test accuracy (29.60%) and best generalization.
- Data preparation and domain-specific architectures had a direct impact on training quality.
- The insights from this project can be applied to other visual recognition systems in e-commerce, retail, and inventory management.

---

Author  
Prajin Paul  
📧 Email: paulprajin2015@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/prajin-paul-b64415247  

