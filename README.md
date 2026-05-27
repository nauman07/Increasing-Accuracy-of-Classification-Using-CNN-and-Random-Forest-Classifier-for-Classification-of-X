# 🫁 Increasing Accuracy of X-Ray Classification Using CNN + Random Forest

A hybrid deep learning approach that combines **Convolutional Neural Networks (CNN)** with a **Random Forest Classifier** to improve classification accuracy for chest X-ray images.

---

## 📖 About

Standard CNN classifiers can sometimes plateau in accuracy. This project explores a hybrid methodology where CNN is used as a **feature extractor**, and the extracted deep features are passed into a **Random Forest Classifier** for the final classification decision — often resulting in improved performance over either method alone.

---

## ✨ Key Approach

```
Input X-Ray Image
        ↓
   CNN (Feature Extractor)
        ↓
Deep Feature Vector
        ↓
Random Forest Classifier
        ↓
   Classification Output
```

---

## 🎯 Classification Classes

- Normal
- Pneumonia
- COVID-19
- Lung Opacity

---

## 📦 Dataset

The dataset can be found on Kaggle (chest X-ray datasets). Modify the data loading paths as needed for your dataset.

🔗 Search: *"COVID-19 Radiography Database"* on [Kaggle](https://www.kaggle.com)

---

## 🛠️ Tech Stack

| Library | Purpose |
|--------|---------|
| `TensorFlow` / `Keras` | CNN architecture and feature extraction |
| `Scikit-learn` | Random Forest Classifier |
| `NumPy` | Array and feature vector operations |
| `Matplotlib` | Visualization of results |
| `OpenCV` | Image preprocessing |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow scikit-learn numpy matplotlib opencv-python
```

### Clone and Run

```bash
git clone https://github.com/nauman07/Increasing-Accuracy-of-Classification-Using-CNN-and-Random-Forest-Classifier-for-Classification-of-X.git
cd Increasing-Accuracy...
python train.py
```

---

## 📈 Why This Hybrid Approach?

- CNNs excel at learning spatial/visual features from images
- Random Forests handle tabular/feature data robustly
- Combining both leverages the strengths of each model
- Often reduces overfitting compared to deep CNN classifiers alone

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes** and should not be used for clinical diagnosis.

---

## 🤝 Contributing

Pull requests and suggestions are welcome. Please open an issue before submitting large changes.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
