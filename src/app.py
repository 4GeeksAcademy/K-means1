import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load dataset
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv")
print(total_data.head())  # Display the first few rows of the dataset

# Select relevant features for clustering
X = total_data[["MedInc", "Latitude", "Longitude"]]
print(X.head())  # Display the first few rows of the selected features

# Split data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(X_train.head())  # Display the first few rows of the training set

# Train KMeans clustering model
model_unsup = KMeans(n_clusters=6, n_init="auto", random_state=42)
model_unsup.fit(X_train)

# Assign cluster labels to the training data
y_train = list(model_unsup.labels_)
X_train["cluster"] = y_train
print(X_train.head())  # Display the training data with cluster labels

# Plot training clusters
fig, axis = plt.subplots(1, 3, figsize=(15, 5))

sns.scatterplot(ax=axis[0], data=X_train, x="Latitude", y="Longitude", hue="cluster", palette="deep")
sns.scatterplot(ax=axis[1], data=X_train, x="Latitude", y="MedInc", hue="cluster", palette="deep")
sns.scatterplot(ax=axis[2], data=X_train, x="Longitude", y="MedInc", hue="cluster", palette="deep")

plt.tight_layout()
plt.show()

# Predict clusters for the test set
y_test = list(model_unsup.predict(X_test))
X_test["cluster"] = y_test
print(X_test.head())  # Display the test data with predicted cluster labels

# Overlay training and testing clusters on scatter plots
fig, axis = plt.subplots(1, 3, figsize=(15, 5))

# Plot training data with reduced opacity
sns.scatterplot(ax=axis[0], data=X_train, x="Latitude", y="Longitude", hue="cluster", palette="deep", alpha=0.1)
sns.scatterplot(ax=axis[1], data=X_train, x="Latitude", y="MedInc", hue="cluster", palette="deep", alpha=0.1)
sns.scatterplot(ax=axis[2], data=X_train, x="Longitude", y="MedInc", hue="cluster", palette="deep", alpha=0.1)

# Plot test data with markers
sns.scatterplot(ax=axis[0], data=X_test, x="Latitude", y="Longitude", hue="cluster", palette="deep", marker="+")
sns.scatterplot(ax=axis[1], data=X_test, x="Latitude", y="MedInc", hue="cluster", palette="deep", marker="+")
sns.scatterplot(ax=axis[2], data=X_test, x="Longitude", y="MedInc", hue="cluster", palette="deep", marker="+")

plt.tight_layout()

# Remove legends to declutter the plots
for ax in axis:
    ax.legend([], [], frameon=False)

plt.show()

# Train a Decision Tree Classifier on the clustered data
model_sup = DecisionTreeClassifier(random_state=42)
model_sup.fit(X_train, y_train)

# Visualize the decision tree
fig = plt.figure(figsize=(15, 15))
tree.plot_tree(
    model_sup,
    feature_names=list(X_train.columns),
    class_names=["0", "1", "2", "3", "4", "5"],
    filled=True
)
plt.show()

# Make predictions on the test set
y_pred = model_sup.predict(X_test)
print(y_pred)  # Display predictions

# Calculate and print the accuracy score
print(accuracy_score(y_test, y_pred))