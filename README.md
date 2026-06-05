# 🌸 Project 2 — Data Classification Using AI
**DecodeLabs Industrial Training | Batch 2026**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> Supervised Learning pipeline using **K-Nearest Neighbors (KNN)** on the classic **Iris dataset** — covering data loading, feature scaling, train-test split, model training, and evaluation with confusion matrix + F1 score.

---

## 📁 Project Structure

```
iris-knn-classifier/
├── iris_classifier.py      # Main Python script (run directly)
├── iris_classifier.ipynb   # Jupyter Notebook (interactive)
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/iris-knn-classifier.git
cd iris-knn-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python iris_classifier.py
```

### 4. Or open the notebook
```bash
jupyter notebook iris_classifier.ipynb
```

---

## 🧠 Pipeline (IPO Framework)

| Stage | Details |
|-------|---------|
| **Input** | Iris dataset (150 samples, 4 features, 3 classes) |
| **Preprocessing** | `StandardScaler` — Mean=0, Variance=1 |
| **Split** | 80% Train / 20% Test (shuffled) |
| **Algorithm** | K-Nearest Neighbors — optimal K via Elbow Method |
| **Output** | Confusion Matrix, F1 Score, Classification Report |

---

## 📊 Dataset — Iris Benchmark

| Property | Value |
|----------|-------|
| Samples | 150 (balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | Sepal Length, Sepal Width, Petal Length, Petal Width |
| Source | `sklearn.datasets.load_iris()` |

---

## 📈 Results

After running the script, `results.png` is generated containing:
- **Elbow Curve** — optimal K selection
- **Confusion Matrix** — TP, FP, FN, TN per class
- **Scatter Plot** — Petal Length vs Petal Width on test set

Expected accuracy: **~97%** | F1 Score: **~0.97**

---

## 🔑 Key Concepts Demonstrated

- **Supervised Learning** vs Heuristic/Rule-based approach
- **Feature Scaling** — why raw data biases distance-based algorithms
- **Train-Test Split** with shuffle to remove order bias
- **KNN — Proximity Principle**: similar things exist in close proximity
- **Choosing K** — Elbow Method to avoid overfitting (K=1) and underfitting (K=100)
- **Accuracy Mirage** — why F1 Score matters more than accuracy on imbalanced data
- **Confusion Matrix** — TP, FP (Type I), FN (Type II), TN

---

## 🚀 Skills Demonstrated

`Data Handling` · `Supervised Learning` · `Model Training` · `Model Evaluation` · `scikit-learn` · `matplotlib` · `seaborn`

---

*Built as part of DecodeLabs Industrial Training — Batch 2026*
