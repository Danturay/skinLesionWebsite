# 🩺 Skin Lesion Checker

An AI-powered web application that helps users check skin lesions by uploading an image. The system analyses the lesion using a trained deep learning model and provides a prediction on whether the lesion is likely **benign** or if it may require **further medical follow-up**.  

⚠️ **Disclaimer**: This tool is **not a medical diagnostic system**. It is intended for educational and research purposes only. Always consult a qualified healthcare professional for medical concerns.  

---

## 🚀 Features

- Upload an image of a skin lesion  
- AI-powered lesion classification (Benign vs. Needs follow-up)  
- Web-based interface, accessible via browser  
- Interactive and user-friendly results page  
- 🏗Modular design for easy extension with new models  

---

## 🛠️ Tech Stack

- **Frontend:** React + TailwindCSS  
- **Backend:** FastAPI (Python)  
- **AI Model:** TensorFlow (Skin lesion classification CNN)  
- **Deployment:** (To be deployed)  

---

## 📂 Project Structure

skin-lesion-checker/
│── frontend/ # React frontend
│── backend/ # FastAPI backend
│── models/ # Trained AI models
│── data/ # Sample images/datasets (if any)
│── requirements.txt # Backend dependencies
│── README.md # Project documentation

---

## 🎯 Purpose

This project was developed as a **personal exploration into AI for healthcare applications**.  
It demonstrates how computer vision and deep learning can be applied to assist with early skin lesion screening in a user-friendly web environment.

---

## 🧠 Model Workflow & Preprocessing

The AI model follows a **structured workflow** to ensure accurate predictions:  

1. **Data Loading**  
   - Images are loaded from the HAM10000 dataset using their image IDs.  
   - Metadata (age, lesion type, etc.) is read from a CSV file. Missing ages are imputed with the mean value.  

2. **Label Encoding**  
   - Lesion categories (`dx`) are converted to numeric labels using `LabelEncoder`.  
   - Labels are one-hot encoded for multi-class classification.  

3. **Data Splitting**  
   - A **grouped split** ensures that lesions from the same patient are not shared across training, validation, and test sets.  
   - This avoids data leakage and ensures model generalisation.  

4. **Image Preprocessing**  
   - Images are resized to 240×240 pixels.  
   - Converted from BGR to RGB and normalised to the `[0,1]` range.  

5. **Data Augmentation**  
   - Applied random flips, rotations, brightness changes, and zooms during training.  
   - Validation data is kept unaugmented to measure true performance.  

6. **Model Architecture**  
   - **Base Model:** EfficientNet-B1 pre-trained on ImageNet (feature extractor).  
   - Added **Global Average Pooling**, **Dropout (0.3)**, and a **Dense softmax output** layer.  
   - Base model is frozen during initial training to leverage pretrained features.  

7. **Training**  
   - Optimiser: Adam  
   - Loss: Categorical Cross-Entropy  
   - Metrics: Accuracy and Recall  
   - **Class weights** are applied to handle imbalanced lesion categories.  
   - **Callbacks:** EarlyStopping and ReduceLROnPlateau for efficient training.  

8. **Evaluation**  
   - Model is tested on the held-out test set.  
   - Metrics include **accuracy, recall, classification report, and confusion matrix**.  
   - Visualisations of training curves and confusion matrices are generated using `matplotlib` and `seaborn`.  

This workflow ensures a robust, patient-level split and high-quality preprocessing to maximise model performance.  

---

## 👨‍💻 Author

Developed by **Dante Turay**  
