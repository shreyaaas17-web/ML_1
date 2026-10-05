# eigenfaces_pca.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_lfw_people

# 1. Load real-world face dataset (Labeled Faces in the Wild)
faces = fetch_lfw_people(min_faces_per_person=60, resize=0.4)
X = faces.data          # each row = flattened face image pixels
y = faces.target
target_names = faces.target_names

print("Dataset shape:", X.shape, "-> each image has", X.shape[1], "pixel features")
print("Number of people (classes):", len(target_names))

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Reduce thousands of pixels -> 150 components (Eigenfaces)
n_components = 150
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nReduced from {X.shape[1]} pixels to {n_components} components")
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.3f}")

# 4. Visualize a few "eigenfaces" (the components themselves)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')
plt.suptitle("Top Eigenfaces (Principal Components)")
plt.show()

# 5. Train classifier on the REDUCED features (much faster than raw pixels)
model = SVC(kernel='rbf', class_weight='balanced', C=1000, gamma=0.001)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# 6. Show some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(faces.images[0].shape), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    ax.set_title(target_names[y_pred[i]].split()[-1], color=color, fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()