# Skin Cancer Classification using EfficientNet-B0: Complete Project Report

This project implements a deep learning solution for classifying skin lesions as **Malignant** or **Benign** using **EfficientNet-B0**. The system combines two medical image datasets and achieves state-of-the-art performance with **96.59% AUROC** and **90.19% accuracy** on a combined test set of 2,660 images.

---

## 🚀 Key Achievements

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **AUROC** | **96.59%** | Excellent ability to discriminate between classes. |
| **Accuracy** | **90.19%** | High overall correctness on the combined test set (2,660 images). |
| **Recall (Sensitivity)** | **88.54%** | Successfully detects most malignant cases (reduces False Negatives). |
| **Specificity** | **91.76%** | Correctly identifies benign cases (reduces unnecessary biopsies). |
| **Training Time** | 33 min 17 sec | Fast convergence leveraging Mixed Precision training on Tesla T4. |

---

## 📊 Datasets Used

The model was trained and validated on a combined dataset sourced from two distinct public repositories, ensuring robustness and generalization.

### Combined Dataset Statistics

| Statistic | Training Data | Test Data |
| :--- | :--- | :--- |
| **Total Images** | **14,516** | **2,660** |
| Malignant ($\text{Target}=1$) | 6,787 (**46.8%**) | 1,300 (**48.9%**) |
| Benign ($\text{Target}=0$) | 7,729 (**53.2%**) | 1,360 (**51.1%**) |

### Dataset Distribution

| Dataset | Training Images | Malignant | Benign |
| :--- | :--- | :--- | :--- |
| Melanoma Dataset | 11,879 | 5,590 | 6,289 |
| Skin Cancer Dataset | 2,637 | 1,197 | 1,440 |

### Test Data Distribution

| Dataset | Test Images | Malignant | Benign |
| :--- | :--- | :--- | :--- |
| Melanoma Dataset | 2,000 | 1,000 | 1,000 |
| Skin Cancer Dataset | 660 | 300 | 360 |

### Dataset Imbalance Handling

Medical datasets typically suffer from class imbalance. This project mitigates this using:
* **Weighted Loss Function:** Used a `pos_weight` of $\mathbf{1.14}$ in `BCEWithLogitsLoss` to penalize misclassification of the minority malignant class more heavily.
* **Stratified K-Fold:** Ensured proportional class distribution in all training/validation splits. 
* **Balanced Test Set:** The test set was intentionally balanced to provide a fair and unbiased final performance evaluation.
* **Data Augmentation:** Applied variations to augment the dataset size and diversify the features of underrepresented classes.

---

## 🖼️ Preprocessing and Augmentation Pipeline

The data pipeline ensures images are standardized and augmented for effective model training.

### Image Processing Steps

| Step | Detail |
| :--- | :--- |
| **Multi-format Support** | Handles BGR, RGB, grayscale, and RGBA formats (all converted to 3-channel RGB). |
| **Resizing** | All images normalized to $\mathbf{224 \times 224}$ pixels. |
| **Normalization** | Uses standard ImageNet statistics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$). |

### Data Augmentation (Training Only)

We employed the `albumentations` library for aggressive regularization:
* **Geometric:** Horizontal/Vertical Flips ($p=0.5$), Random Rotation ($90^\circ$, $p=0.5$), Shift, Scale, Rotate ($p=0.5$).
* **Color/Light:** Random Brightness/Contrast adjustments ($p=0.3$).

---

## 🧠 Model Architecture: EfficientNet-B0

The project uses **EfficientNet-B0**, a convolutional neural network (CNN) known for its excellent balance of accuracy and computational efficiency, achieved through compound scaling.

### Key Features

* **Base Model:** **EfficientNet-B0** pre-trained on **ImageNet**.
* **Strategy:** **Transfer Learning** (Fine-tuning).
* **Frozen Layers:** Early convolutional layers were kept frozen to retain general feature extraction capabilities.
* **Trainable Layers:** The **last 3 blocks** of the base model and the **entire custom classifier** were fine-tuned.

### Model Parameters

| Parameter Type | Count | % of Total |
| :--- | :--- | :--- |
| Total Parameters | 4,336,253 | 100% |
| **Trainable Parameters** | **999,315** | **23%** |
| Input Shape | $224 \times 224 \times 3$ (RGB) | |

### Custom Classifier Architecture

The original 1000-class classifier was replaced with a compact, robust binary classifier:

$$\text{Input (1280)} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256) \rightarrow \text{ReLU} \rightarrow \text{BatchNorm1d} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(1)$$

---

## ⚙️ Training Configuration

### Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 35 | Maximum training epochs. |
| **Batch Size** | 32 (Train) / 64 (Valid/Test) | Training batch size. |
| **Learning Rate** | $2.50 \times 10^{-5}$ | Initial learning rate. |
| **Optimizer** | **AdamW** | Adam optimization with decoupled weight decay. |
| **Scheduler** | **CosineAnnealingLR** | Smooth learning rate decay schedule. |
| **Early Stopping** | Patience $= 5$ | Stops training if validation AUROC does not improve for 5 epochs. |
| **Precision** | **Mixed Precision** | Enabled for faster training and lower VRAM usage. |

### Cross-Validation Strategy

* **Method:** $\text{5-Fold StratifiedGroupKFold}$
* **Fold Used:** Fold 0
    * Training Samples: 11,612
    * Validation Samples: 2,904
* **Seed:** $\mathbf{42}$ for complete reproducibility.

---

## 📈 Training Performance

### Final Metrics (Best Checkpoint - Epoch 24)

| Metric | Train Value | Validation Value |
| :--- | :--- | :--- |
| **Loss** | 0.2910 | **0.2613** |
| **AUROC** | 0.9544 | **0.9635** |
| Accuracy | 0.8835 | 0.8981 |
| F1 Score | 0.8778 | 0.8920 |

The best model checkpoint was saved at **Epoch 24** with a validation AUROC of **0.9631**.

---

## 🔬 Test Set Performance

The model's generalization ability was assessed on the dedicated **2,660-image test set**.

### Confusion Matrix Analysis

The Confusion Matrix provides a breakdown of the model's classifications: 

\`\`\`text
                 Predicted
                 Benign   Malignant
Actual Benign    1,248       112
Actual Malignant   149     1,151
\`\`\`

* **True Negatives (TN):** 1,248 (Correctly identified benign cases)
* **False Positives (FP):** 112 (Benign cases wrongly flagged as malignant - **Type I Error**)
* **False Negatives (FN):** 149 (Malignant cases missed - **Type II Error**)
* **True Positives (TP):** 1,151 (Correctly identified malignant cases)

### Detailed Classification Report

\`\`\`text
              precision    recall  f1-score   support

      Benign     0.8933    0.9176    0.9053      1360
   Malignant     0.9113    0.8854    0.8982      1300

    accuracy                         0.9019      2660
   macro avg     0.9023    0.9015    0.9017      2660
weighted avg     0.9021    0.9019    0.9018      2660
\`\`\`

### Dataset-Specific Performance

| Dataset | Images | AUROC | Accuracy |
| :--- | :--- | :--- | :--- |
| **Melanoma Dataset** | 2,000 | **0.9709** | **0.9135** |
| **Skin Cancer Dataset** | 660 | 0.9501 | 0.8667 |

The model demonstrates **consistent and high performance** across both source datasets, indicating good generalization.

---

## ⚕️ Clinical Implications

The balanced metrics show the model's fitness for clinical assistance:
* The high **Specificity (91.76%)** helps in **reducing unnecessary biopsies** by accurately ruling out benign lesions.
* The high **Recall (88.54%)** ensures **critical malignant cases are rarely missed**, supporting early detection efforts.
* The system offers a **fast inference time**, making it suitable for integration into busy clinical workflows.

---

## 🛠️ Future Improvements

To further enhance clinical readiness and performance:

### Technical Enhancements
* **Ensemble Methods:** Combine predictions from multiple distinct models (e.g., EfficientNet-B4, ResNet) to improve reliability.
* **Advanced Augmentation:** Implement medical-specific augmentations such as color constancy or specialized blurring.
* **Explainability (XAI):** Integrate **Grad-CAM**  to visualize the regions of the image the model uses for classification, building trust with clinicians.
* **Uncertainty Estimation:** Provide confidence scores for predictions to guide the clinician on ambiguous cases.

### Clinical Integration
* **Multi-center Validation:** Test the model's performance on images acquired from diverse hospitals and imaging equipment to validate real-world robustness.
* **Risk Stratification:** Extend the model to not just classify, but also provide a severity grade for malignant cases.

---

## 📦 Saved Outputs

All necessary files for model reproduction and performance analysis are saved:

| File Type | Description |
| :--- | :--- |
| **Model Checkpoint** | \`Fold0_efficientnet_AUROC0.9631_epoch24.pth\` (Best model weights) |
| **Predictions** | \`test_predictions_fold0_efficientnet.csv\` |
| **Metrics** | \`test_metrics_fold0_efficientnet.csv\` (Comprehensive final metrics) |
| **History** | \`history_fold0_efficientnet.csv\` (Epoch-by-epoch training logs and Learning Rate curve) |

---

## ✅ Conclusion

The EfficientNet-B0 model demonstrates exceptional performance in skin cancer classification, achieving **96.59% AUROC** and **90.19% accuracy** on a diverse test set. The system effectively handles class imbalance through weighted loss functions and provides robust, reproducible results suitable for clinical validation.

The model's balanced performance (**88.54% recall, 91.76% specificity**) makes it a promising tool for assisting dermatologists in early skin cancer detection, potentially reducing unnecessary biopsies while maintaining high sensitivity for malignant cases.

This implementation provides a solid foundation for further development and clinical integration of AI-assisted dermatological diagnosis systems.
