# ============================================================
# Project 2: Data Classification Using AI
# DecodeLabs Industrial Training - Batch 2026
# Algorithm: K-Nearest Neighbors on Iris Dataset
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# ── 1. LOAD DATASET ──────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

print("=" * 55)
print("  DecodeLabs | Project 2 — Iris KNN Classifier")
print("=" * 55)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {list(class_names)}")
print(f"Features      : {feature_names}\n")

# ── 2. FEATURE SCALING (StandardScaler) ──────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. TRAIN / TEST SPLIT (80 / 20) ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}\n")

# ── 4. FIND OPTIMAL K (Elbow Method) ─────────────────────────
error_rates = []
k_range     = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = int(np.argmin(error_rates)) + 1
print(f"Optimal K (lowest error) : {optimal_k}\n")

# ── 5. TRAIN FINAL MODEL ──────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ── 6. EVALUATION ─────────────────────────────────────────────
acc = accuracy_score(y_test, predictions)
f1  = f1_score(y_test, predictions, average="weighted")
cm  = confusion_matrix(y_test, predictions)

print(f"Accuracy : {acc * 100:.2f}%")
print(f"F1 Score : {f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

# ── 7. VISUALISATIONS ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "DecodeLabs | Project 2 — KNN Iris Classification",
    fontsize=14, fontweight="bold"
)

# — Plot A: Elbow Curve —
axes[0].plot(k_range, error_rates, marker="o", color="#1f4e79", linewidth=2)
axes[0].axvline(optimal_k, color="#e84118", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0].set_title("Elbow Curve — Choosing K")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Error Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# — Plot B: Confusion Matrix —
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# — Plot C: Feature Scatter (Petal Length vs Petal Width) —
colors = ["#1f4e79", "#e84118", "#27ae60"]
for idx, cls in enumerate(class_names):
    mask = y_test == idx
    axes[2].scatter(
        X_test[mask, 2], X_test[mask, 3],
        color=colors[idx], label=cls, alpha=0.8, edgecolors="white", s=80
    )
axes[2].set_title("Test Set — Petal Length vs Width (Scaled)")
axes[2].set_xlabel("Petal Length (scaled)")
axes[2].set_ylabel("Petal Width (scaled)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[✓] Visualisation saved to results.png")

# ── 8. QUICK PREDICTION DEMO ─────────────────────────────────
print("\n--- Sample Prediction Demo ---")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical Setosa
sample_scaled = scaler.transform(sample)
pred_class    = model.predict(sample_scaled)[0]
print(f"Input features : {sample[0]}")
print(f"Predicted class: {class_names[pred_class]}")
