🫁 CHEST X-RAY PNEUMONIA CLASSIFICATION USING CNN 🫁
🏆 OVERVIEW
🚀 This project detects Pneumonia from Chest X-ray images using Convolutional Neural Networks (CNNs).
🩺 Early diagnosis can support faster medical response and better patient outcomes.

📂 DATASET
📸 Thousands of chest X-ray images categorized into two classes:
✅ NORMAL – Healthy lungs
✅ PNEUMONIA – Infected lungs (bacterial or viral)
📌 Preprocessing Techniques
🔹 Image Resizing & Normalization – Standardizes input for model compatibility
🔹 Data Augmentation – Improves model generalization using Keras ImageDataGenerator

⚡ MODEL USED & PERFORMANCE
🟢 1️⃣ CUSTOM CNN MODEL (BUILT FROM SCRATCH)
🛠 Architecture:
✔️ 3–4 Convolutional Layers (ReLU, MaxPooling)
✔️ Dropout layers for regularization
✔️ Dense Softmax Output Layer

🔹 Optimizer: Adam (LR = 0.0001)
🎯 Accuracy: ~87%

📌 Insight: Performs well with basic image features, but may underperform with complex medical patterns.

🔬 EVALUATION METRICS
📊 Confusion Matrix
📈 Accuracy & Loss Curves
🧮 Classification Report (Precision, Recall, F1-Score)

🔍 VISUALIZATIONS
📉 Training & Validation Accuracy/Loss graphs
🧾 Performance metrics shown through plots and confusion matrix for better interpretability.

🛠 TOOLS & TECHNOLOGIES
TensorFlow / Keras

NumPy, Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

