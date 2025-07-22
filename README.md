# Visual Product Recognition for E-commerce Catalog Automation

## Overview

This project focuses on building intelligent visual recognition systems to **automate product identification and categorization in e-commerce platforms**. Leveraging deep learning, the models are trained to accurately classify customer-submitted product images, addressing real-world challenges such as large-scale, diverse, and often noisy e-commerce data.

By implementing and comparing multiple architectures—including CNNs, ResNet50, and Vision Transformers—the project aims to streamline catalog management, improve listing accuracy, and reduce manual effort in online marketplaces.

---

## Objectives

- **Automate Product Tagging:** Reliably identify and classify products from user-uploaded images for faster, more accurate cataloging.
- **Architecture Benchmarking:** Compare deep learning approaches to determine the most effective solution for e-commerce image data.
- **Robustness to Real-World Data:** Apply techniques to handle class imbalance, poor image quality, and diverse backgrounds typical in customer photos.

---

## E-commerce Relevance

Product recognition is a cornerstone of modern e-commerce operations:

- **Automated Cataloging:** Instantly categorize new product listings, reducing manual data entry and human error.
- **Duplicate & Counterfeit Detection:** Identify duplicate or fake listings by visually matching products.
- **Personalized Search & Recommendations:** Enhance search accuracy and product discovery by leveraging visual similarity.
- **Quality Control:** Flag low-quality or miscategorized images before they go live.

---

## Dataset: Products-10K

- **Source:** Open-source, e-commerce-focused product image dataset.
- **Scale:** ~142,000 training images, ~55,000 test images, 9,600+ product categories.
- **Challenges:** Highly imbalanced classes, variable image quality, diverse backgrounds—mirroring real e-commerce submissions.

---

## Key Challenges

- **Class Imbalance:** Some product types are rare (e.g., niche accessories), while others are common (e.g., smartphones).
- **Image Quality:** User-uploaded images vary in lighting, focus, and background clutter.
- **Scalability:** Need for fast, scalable models to handle millions of listings.

---

## Preprocessing Workflow

- **Image Resizing:** Standardized to 224×224 pixels for model compatibility.
- **Normalization:** Pixel values scaled to [0, 1] for stable training.
- **Data Augmentation:** Flipping, rotation, and color jitter to simulate real-world listing diversity.
- **Imbalance Handling:** Oversampling rare classes and using class-weighted loss functions.
- **Train/Validation Split:** 80:20 for robust evaluation.

---

## Model Architectures

### 1. CNN with ResNet50 (Transfer Learning)
- Pre-trained on ImageNet, fine-tuned for e-commerce product categories.
- Custom classification layers, dropout, and class-weighted loss for imbalance.

### 2. Vision Transformer (ViT)
- Processes image patches with self-attention, capturing global context for complex product images.

### 3. Custom Lightweight CNN
- 3-layer architecture optimized for e-commerce data.
- Batch normalization, dropout, and aggressive augmentation for generalization.

---

## Evaluation Results

| Architecture      | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------------------|------------------|--------------------|--------------|
| CNN + ResNet50    | 5.02%            | 6.54%              | 4.99%        |
| Vision Transformer| 9.26%            | 7.47%              | 4.35%        |
| Custom CNN        | 86.00%           | 30.50%             | 29.60%       |

**Key Takeaway:**  
The custom CNN, tailored for e-commerce data, outperformed more complex models, highlighting the importance of domain-specific optimization and data augmentation.

---

## Insights for E-commerce

- **Data Quality is Critical:** User-generated images require robust preprocessing and augmentation.
- **Class Imbalance Must Be Addressed:** Rare product types need special handling to avoid bias.
- **Smaller, Efficient Models Work Best:** For noisy, diverse e-commerce data, lightweight models generalize better than large, generic architectures.

---

## Future Work

- **Sequence Modeling:** Integrate CNN + LSTM for video-based product listings (e.g., 360° views).
- **Attention Mechanisms:** Use attention layers to focus on key product features (e.g., logos, labels).
- **API Deployment:** Package the best model as a REST API for integration with e-commerce platforms.

---

## Author

**Prajin Paul**  
📧 paulprajin2015@gmail.com  
[LinkedIn](https://www.linkedin.com/in/prajin-paul-b64415247)

---

*Feel free to fork, contribute, or reach out for collaboration!*
