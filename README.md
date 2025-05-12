# PCOS-Detection-Using-Deep-Learning
A deep learning-based system to detect Polycystic Ovary Syndrome from ultrasound images using VGG16 for feature extraction and XGBoost for classification, achieving 95.74% accuracy.
## Features
- Image classification into **PCOS** and **Non-PCOS**.
- Preprocessing, enhancement, and augmentation of ultrasound images.
- Feature extraction using **VGG16** pretrained on ImageNet.
- Compared Naïve Bayes, SVM, and XGBoost models.
- Developed UI using **Streamlit** (Flask version also available).

##  Dataset

- Source: Kaggle
- Total Images: 234 grayscale ultrasound images
  - Training: 164 (60 Non-PCOS, 104 PCOS)
  - Testing: 47 (27 Non-PCOS, 20 PCOS)
  - Validation: 23 (13 Non-PCOS, 10 PCOS)
- Annotations: YOLOv8 format
- Classes: 2 (PCOS, Non-PCOS)

##  Technologies Used

- Python, NumPy, Pandas, OpenCV
- TensorFlow/Keras, Scikit-learn, XGBoost
- Streamlit (for UI)
- Google Colab 

## 🧠 Model Performance

- VGG16 + XGBoost: **95.74% accuracy**
- Outperformed Naïve Bayes and SVM models
