# 🫁 CHEST X-RAY PNEUMONIA CLASSIFICATION USING CNN 🫁

## 🏆 OVERVIEW  
🚀 This project detects **Pneumonia** from **Chest X-ray images** using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.  
🩺 Early diagnosis can support faster medical response and better patient outcomes.

---

## 📂 DATASET  
📸 Thousands of chest X-ray images categorized into two classes:  
✅ **NORMAL** – Healthy lungs  
✅ **PNEUMONIA** – Infected lungs (bacterial or viral)


📌 **Preprocessing Techniques**  
🔹 Image Resizing & Normalization – Standardizes input for model compatibility  
🔹 Data Augmentation – Improves model generalization using Keras `ImageDataGenerator`

---

## ⚡ MODELS USED & PERFORMANCE COMPARISON

### 🟢 1️⃣ CUSTOM CNN MODEL (BUILT FROM SCRATCH)  
🛠 **Architecture:**  
✔️ 3–4 Convolutional Layers (ReLU, MaxPooling)  
✔️ Dropout layers for regularization  
✔️ Dense Softmax Output Layer

🔹 **Optimizer:** Adam (LR = 0.0001)  
🎯 **Accuracy:** ~87%  

📌 **Insight:** Performs well on small datasets; learns from scratch but lacks robustness compared to pretrained models.

---

### 🔵 2️⃣ TRANSFER LEARNING – MobileNetV2 (Frozen Base)  
🛠 **Architecture:**  
✔️ Uses pretrained MobileNetV2 as a frozen feature extractor  
✔️ Custom dense layers on top for binary classification

🔹 **Optimizer:** SGD (LR = 0.001, Momentum = 0.9)  
🎯 **Accuracy:** ~91%

📌 **Insight:** Leverages pretrained features from ImageNet; generalizes better than a custom CNN.

---

### 🔴 3️⃣ TRANSFER LEARNING – MobileNetV2 (Fine-Tuned)  
🛠 **Architecture:**  
✔️ Same base as above, but unfreezes last few layers of MobileNetV2  
✔️ Fine-tunes with lower learning rate for task-specific adaptation

🔹 **Optimizer:** SGD (LR = 0.0005, Momentum = 0.9)  
🎯 **Accuracy:** ~94%

📌 **Insight:** Best-performing model due to targeted fine-tuning of pretrained weights for medical imaging.

---

## 📊 PERFORMANCE SUMMARY

| Model                              | Optimizer | Learning Rate | Accuracy |
|-----------------------------------|-----------|----------------|----------|
| 🟢 Custom CNN                     | Adam      | 0.0001         | ~87%     |
| 🔵 MobileNetV2 (Frozen)           | SGD       | 0.001          | ~91%     |
| 🔴 MobileNetV2 (Fine-Tuned)       | SGD       | 0.0005         | ~94%     |

---

## 🔬 EVALUATION METRICS  
📊 Confusion Matrix  
📈 Accuracy & Loss Curves  
🧮 Classification Report (Precision, Recall, F1-Score)

---

## 🔍 VISUALIZATIONS  
📉 Training & Validation Accuracy/Loss graphs  
🧾 Confusion matrix and performance metrics for interpretability

---

## 🛠 TOOLS & TECHNOLOGIES  
- TensorFlow / Keras  
- MobileNetV2  
- NumPy, Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook

---

## 📌 KEY TAKEAWAYS  
✅ Fine-tuning pretrained models like MobileNetV2 significantly improves performance  
✅ Transfer learning boosts accuracy even with limited data  
✅ Custom CNNs can be effective but are outperformed by transfer learning techniques

---

## 📈 FUTURE IMPROVEMENTS  
🚧 Experiment with ResNet50, EfficientNet, or Xception  
🚧 Apply Grad-CAM for visual explanations of model predictions  
🚧 Deploy a web app for real-time pneumonia diagnosis from chest X-rays

---

## 📝 LICENSE  
This project is open-source and available under the **MIT License**.
