# Importing necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the Wine Dataset
wine = load_wine()
X, y = wine.data, wine.target

# Splitting the data for classification (70% training, 30% testing)
# This split ensures that the model is trained on 70% of the data and tested on the remaining 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Classification using Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Classification Results
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=wine.target_names)

print("Classification Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)

# Plotting the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, fontsize=10)
plt.title("Decision Tree for Wine Classification", fontsize=14)
plt.show()

# Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Calculating Silhouette Score
silhouette_avg = silhouette_score(X, clusters)

# Printing Clustering Results
print("Silhouette Score for Clustering:", silhouette_avg)
print("Cluster Distribution:", np.bincount(kmeans.labels_))

# Visualizing Clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(14, 8))
for cluster_id in range(3):
    plt.scatter(
        X_pca[clusters == cluster_id, 0], 
        X_pca[clusters == cluster_id, 1], 
        label=f"Cluster {cluster_id}"
    )
plt.scatter(
    pca.transform(kmeans.cluster_centers_)[:, 0], 
    pca.transform(kmeans.cluster_centers_)[:, 1], 
    color='black', 
    marker='x', 
    s=150, 
    label='Centroids'
)
plt.title("K-Means Clustering of Wine Dataset (PCA Reduced)", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Additional Graphics: Distribution of Features
plt.figure(figsize=(16, 10))
for i, feature_name in enumerate(wine.feature_names):
    plt.subplot(4, 4, i + 1)
    plt.hist(X[:, i], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(feature_name, fontsize=10)
    plt.tight_layout()
plt.suptitle("Distribution of Features in the Wine Dataset", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.show()

# Explanation of Training and Testing Split
# - The training set (70% of the data) is used to fit the Decision Tree model. These data points allow the model to learn patterns.
# - The testing set (30% of the data) is unseen by the model during training. It is used to evaluate the model's performance, ensuring it generalizes well to new data.
# - The splitting process ensures that the data remains stratified, maintaining class distributions across both sets.
