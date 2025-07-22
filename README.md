# 🤖 Deep Learning Project: Visual Product Recognition in Multimedia Applications

## 📌 Project Description

This repository contains deep learning models and supporting documentation for solving **visual product recognition** challenges using modern computer vision techniques in e-commerce and retail analytics. The project harnesses convolutional neural networks (CNNs), ResNet50, Vision Transformers (ViT), and hybrid models, designed and implemented with **PyTorch**.

The experiments focus on real-world issues such as data imbalance, low-image quality, and massive class diversity — leveraging powerful GPU-based systems to perform large-scale training on the **Products-10K** dataset (200K+ images).

---

## 🎯 Objective

To design and implement deep learning architectures that can accurately classify products in images under various real-world challenges (e.g. lighting, occlusion, orientation). Models were trained to distinguish visual patterns in customer-review images and enable scalable, automated product identification.

---

## 📌 Significance in Multimedia Processing

- **E-Commerce** ➜ Auto-tagging items from user-uploaded images for improved product matching and recommendations.
- **Inventory Optimization** ➜ Surveillance and shelf-tracking systems for retail stores.
- **Fake Product Prevention** ➜ Recognizing counterfeit products during quality control procedures.
- **Warehouse Management** ➜ Real-time categorization and tracking to enhance logistics efficiency.

---

## 📂 Dataset: Products-10K

- ✅ 141,931 training images  
- ✅ 55,375 testing images  
- ✅ ~9,690 classes  
- ✅ ~18GB dataset size  
- Source: Publicly available Products-10K image classification dataset

### 📉 Challenges:
- Skewed class distribution (some classes had only 2 images)
- Low-quality user submissions (blur, poor lighting, occlusions)
- Massive class count and metadata complexity
- High compute demand on local GPUs

---

## 🧪 Data Preprocessing Techniques

- **Image Resizing**: Standardized to 224×224 input size.
- **Normalization**: Scaling pixel values between 0–1.
- **Data Augmentation**: Flipping, color jitter, rotation to improve generalization.
- **Oversampling / Undersampling**: Balanced rare classes.
- **Weighted Loss**: Prioritized underrepresented classes using class-weighted cross-entropy.
- **Train/Validation Splits**: 80:20 ratio for training and evaluation.

---

## 🏗️ Model Design & Architecture

### 🔶 Model 1: CNN with ResNet50 (Transfer Learning)
- ResNet50 used as a feature extractor
- Dropout (p=0.5) to prevent overfitting
- Early stopping to avoid unnecessary epochs 
- Data augmentation incorporated

### 🔶 Model 2: Vision Transformer (ViT)
- 16×16 patch embedding
- Classification head customized for 9.6k classes
- Specialized for learning global feature representations
- Applied attention to obtain contextual image understanding

### 🔶 Model 3: CNN with Data Augmentation + Oversampling
- 3-layer custom CNN with filter sizes 12, 20, 32
- Batch normalization, dropout regularization
- Strong performance and generalization despite data noise

---

## 🚀 Future Models

### 📈 CNN + LSTM  
Captures spatial and sequential (temporal) patterns from image series or time-dependent data.

### 📌 Attention-based CNN  
Integrates spatial attention modules to focus on important regions of an image, improving classification for complex datasets.

---

## 🏁 Model Training & Performance

| 📍Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------|-------------------|---------------------|----------------|
| CNN (ResNet50) | 5.02% | 6.54% | 4.99% |
| Vision Transformer | 9.26% | 7.47% | 4.35% |
| Custom CNN | **86.00%** | **30.50%** | **29.60%** |

### ✅ Key Observations:

- Lower performance in Model 1 and Model 2 → Bottlenecks due to class imbalance, low image quality.
- Model 3 showed **best learning and generalization capacity** thanks to data balancing and architecture tuning.

---

## 📊 Comparative Insights

- Model 3 clearly outperformed others in accuracy and training stability.
- Training deeper models (e.g., ViT) required more tuning and compute, often underperforming with small metrics.
- Dataset quality and class imbalance were the biggest hurdles impacting overall model success.

---

## 🔍 Insights Gained

- **Data Quality Directly Impacts Accuracy**: Poor lighting, occlusion, and blur were major prediction hurdles.
- **Balanced Datasets Matter**: Handling class imbalance dramatically helped generalization in Model 3.
- **GPU Acceleration Was Critical**: Using CUDA on NVIDIA helped reduce training time and scale up experiments.
- **Model 3 = Best Performer**: Due to its simplicity, powerful augmentations, and class balancing strategy.

---

## ✅ Conclusion

This project explored and compared deep learning models for solving **Visual Product Recognition** from scratch. In particular:

- It demonstrated how CNNs, Transformers, and hybrid models behave under real-world multimedia challenges.
- It highlighted the **need for balanced data**, compute infrastructure, and proper model selection.
- The custom CNN with data augmentation and oversampling significantly outperformed other approaches in the task.

These findings are impactful for industries requiring scalable product recognition systems, such as e-commerce, logistics, and retail analytics. The project opened up many future directions for hybrid architectures and efficient model deployment.

---

## 📚 References

- Mieczysław Pawłowski. *Machine Learning-Based Product Classification for eCommerce*, May 2021.
- Fei Sun et al. *Product Classification with the Motivation of Target Consumers by Deep Learning*, June 2022.
- Leonidas Akritidis et al. *Effective Products Categorization with Importance Scores*, November 2018.

---

## 👤 Author

**Prajin Paul**  
📧 [paulprajin2015@gmail.com](mailto:paulprajin2015@gmail.com)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/prajin-paul-b64415247/)
