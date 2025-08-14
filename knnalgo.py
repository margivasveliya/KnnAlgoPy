import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)


print(" trying different value of K")

k_range = range(1, 11)
accuracy_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    print(f"K = {k:<2} → Accuracy = {acc:.4f}")


best_k_index = np.argmax(accuracy_list)
best_k = k_range[best_k_index]
print(f"\n Best value for K is {best_k} with accuracy of {accuracy_list[best_k_index]:.4f}")


final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
final_predictions = final_knn.predict(X_test)

print("\n the confuse matrix for the final model:")
cm = confusion_matrix(y_test, final_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()


print("\n Based on two features visualizing the document")

X_2d = X_scaled[:, :2]


X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, stratify=y, random_state=42
)

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_train_2d, y_train_2d)


h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel2, alpha=0.8)

scatter = plt.scatter(
    X_test_2d[:, 0], 
    X_test_2d[:, 1], 
    c=y_test_2d, 
    cmap=plt.cm.Set1, 
    edgecolor='k', 
    s=60, 
    marker='o'
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title(f"KNN Decision Boundary (K = {best_k})")
plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Class")
plt.grid(True)
plt.tight_layout()
plt.show()

